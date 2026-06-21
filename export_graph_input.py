#!/usr/bin/env python3
"""
Create compact graph input for the C++/CUDA junction detector.
"""

from __future__ import annotations

import json
import sys
import time
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# ---------------- USER PARAMETERS ----------------
RAMP_ONLY = True
MAX_POINTS_PER_LINK = 64

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
# -------------------------------------------------

class LinkType:
    OTHER = 0
    PRIMARY = 1
    SECONDARY = 2

@dataclass
class DualGraph:
    link_ids: List[Any]
    row_offsets: List[int]
    col_indices: List[int]
    link_type: List[int]

    @property
    def n(self) -> int:
        return len(self.link_ids)

    @property
    def directed_edge_count(self) -> int:
        return len(self.col_indices)

def normalize_id(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, str):
        value = value.strip().strip("'").strip('"')
    try:
        return int(value)
    except Exception:
        return value

def jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    return value

def pick_column(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    lower_to_real = {str(c).lower(): str(c) for c in columns}
    for cand in candidates:
        if cand.lower() in lower_to_real:
            return lower_to_real[cand.lower()]
    return None

def find_link_id_column(df) -> str:
    col = pick_column(
        df.columns,
        ["link_id", "linkId", "link_local_ref", "linkLocalRef", "local_ref", "id"],
    )
    if col is None:
        raise RuntimeError(f"Could not find link id column. Columns: {list(df.columns)}")
    return col


def find_endpoint_columns(df) -> Tuple[str, str]:
    candidates = [
        ("start_node_id", "end_node_id"),
        ("startNodeId", "endNodeId"),
        ("start_node", "end_node"),
        ("from_node_id", "to_node_id"),
        ("fromNodeId", "toNodeId"),
        ("ref_node_id", "non_ref_node_id"),
        ("refNodeId", "nonRefNodeId"),
        ("node0", "node1"),
        ("source", "target"),
        ("u", "v"),
    ]

    for a, b in candidates:
        ca = pick_column(df.columns, [a])
        cb = pick_column(df.columns, [b])
        if ca is not None and cb is not None:
            return ca, cb

    raise RuntimeError(f"Could not find endpoint columns. Columns: {list(df.columns)}")


def road_links_from_raw_dict(road: Dict[str, Any]):
    rows = []

    for lid, rec in (road.get("links", {}) or {}).items():
        if not isinstance(rec, dict):
            continue

        u = rec.get("start_node_id", rec.get("u", rec.get("from_node_id")))
        v = rec.get("end_node_id", rec.get("v", rec.get("to_node_id")))

        if u is None or v is None:
            continue

        rows.append((normalize_id(lid), normalize_id(u), normalize_id(v)))

    return rows


# Return rows as (link_id, start_node, end_node).
def road_link_rows(road: Dict[str, Any], origin: Optional[Dict[str, Any]]):
    rows = road_links_from_raw_dict(road)
    if not rows:
        raise RuntimeError("Could not build road-link rows from road['links'].")
    return rows

def query_ids(routing: Dict[str, Any], expr: str) -> Set[Any]:
    from libs.road_topology import find_links_by_attr_query

    return {normalize_id(x) for x in find_links_by_attr_query(routing, expr)}


def classify_links(routing: Dict[str, Any], ramp_only: bool) -> Tuple[Set[Any], Set[Any]]:
    if ramp_only:
        secondary_expr = "is_ramp == true"
    else:
        secondary_expr = "is_ramp == true or is_within_interchange == true"

    primary_expr = (
        'functional_class == "FC_1" or functional_class == "FC_2" '
        'or functional_class == "FC_3"'
    )

    secondary_ids = query_ids(routing, secondary_expr)
    primary_ids = query_ids(routing, primary_expr) - secondary_ids

    return secondary_ids, primary_ids


def build_dual_graph(
    road: Dict[str, Any],
    secondary_ids: Set[Any],
    primary_ids: Set[Any],
    origin: Optional[Dict[str, Any]],
) -> DualGraph:
    secondary_ids = {normalize_id(x) for x in secondary_ids}
    primary_ids = {normalize_id(x) for x in primary_ids}
    valid_ids = secondary_ids | primary_ids

    rows = []
    for link_id, u, v in road_link_rows(road, origin):
        if normalize_id(link_id) in valid_ids:
            rows.append((normalize_id(link_id), normalize_id(u), normalize_id(v)))

    if not rows:
        raise RuntimeError("No road-link rows matched the classified primary/secondary ids.")

    link_ids = sorted({r[0] for r in rows}, key=lambda x: str(x))
    link_to_idx = {link_id: i for i, link_id in enumerate(link_ids)}

    node_to_links = defaultdict(list)
    for link_id, u, v in rows:
        idx = link_to_idx[link_id]
        node_to_links[u].append(idx)
        node_to_links[v].append(idx)

    adjacency = [set() for _ in link_ids]

    for incident in node_to_links.values():
        if len(incident) < 2:
            continue
        for i in incident:
            for j in incident:
                if i != j:
                    adjacency[i].add(j)

    row_offsets = [0]
    col_indices = []

    for neigh in adjacency:
        col_indices.extend(sorted(neigh))
        row_offsets.append(len(col_indices))

    link_type = []
    for link_id in link_ids:
        if link_id in secondary_ids:
            link_type.append(LinkType.SECONDARY)
        elif link_id in primary_ids:
            link_type.append(LinkType.PRIMARY)
        else:
            link_type.append(LinkType.OTHER)

    return DualGraph(
        link_ids=link_ids,
        row_offsets=row_offsets,
        col_indices=col_indices,
        link_type=link_type,
    )


def load_data(config_path: Path):
    from libs.lane_network import load_config
    from libs.road_topology import build_file_lists_from_config, build_from_tiles, build_from_tiles_attr

    cfg = load_config(str(config_path))
    _geom_files, _topo_files, _attr_files, ref_files, route_files = build_file_lists_from_config(cfg)

    road = build_from_tiles(list(ref_files))
    routing = build_from_tiles_attr(list(route_files))

    origin = {
        "lat0": float(road.get("origin_lat", cfg.get("OriginLat", 0.0)) or 0.0),
        "lon0": float(road.get("origin_lon", cfg.get("OriginLon", 0.0)) or 0.0),
        "h0": float(cfg.get("OriginAltCm", 0) or 0) / 100.0,
    }

    return cfg, road, routing, origin


def fill_geometry(config_path: Path, cfg: Dict[str, Any], road: Dict[str, Any]):
    from libs.road_topology import build_file_lists_from_config, fill_road_from_sxy
    from libs.lane_network import build_lane_network_enu

    geom_files, topo_files, attr_files, _ref_files, _route_files = build_file_lists_from_config(cfg)

    Sxy, _ST, _Graph, _SA, origin = build_lane_network_enu(
        list(geom_files),
        list(topo_files),
        OriginLat=road.get("origin_lat"),
        OriginLon=road.get("origin_lon"),
        OriginAltCm=cfg.get("OriginAltCm", 0),
        Undirected=cfg.get("Undirected", True),
        attrFiles=list(attr_files),
    )

    fill_road_from_sxy(road, Sxy, topo_geom_files=cfg.get("topoGeomFiles", []))

    return origin


def make_projection(origin: Dict[str, Any]):
    from libs.geometry_helpers import latlon_to_xy

    def projection(lats, lons):
        lats = np.asarray(lats, dtype=float)
        lons = np.asarray(lons, dtype=float)

        return latlon_to_xy(
            lats,
            lons,
            float(origin["lat0"]),
            float(origin["lon0"]),
            float(origin.get("h0", 0.0)),
            None,
        )

    return projection


def link_record(road: Dict[str, Any], link_id: Any) -> Optional[Dict[str, Any]]:
    links = road.get("links", {}) or {}

    if link_id in links:
        return links[link_id]

    s = str(link_id)
    if s in links:
        return links[s]

    try:
        i = int(link_id)
        if i in links:
            return links[i]
        if str(i) in links:
            return links[str(i)]
    except Exception:
        pass

    return None


def node_latlon(road: Dict[str, Any], node_id: Any):
    nodes = road.get("nodes", {}) or {}
    key = str(normalize_id(node_id))

    node_rec = nodes.get(key)
    if node_rec is None:
        try:
            node_rec = nodes.get(int(key))
        except Exception:
            node_rec = None

    if not isinstance(node_rec, dict):
        return None

    if "lat" not in node_rec or "lon" not in node_rec:
        return None

    return float(node_rec["lat"]), float(node_rec["lon"])


def link_xy_points(
    road: Dict[str, Any],
    origin: Dict[str, Any],
    link_id: Any,
    max_points_per_link: int,
):
    rec = link_record(road, link_id)
    if not isinstance(rec, dict):
        return np.empty((0, 2), dtype=float)

    projection = make_projection(origin)

    lats = np.asarray(rec.get("lat", []), dtype=float)
    lons = np.asarray(rec.get("lon", []), dtype=float)

    if lats.size >= 2 and lons.size >= 2 and lats.size == lons.size:
        xs, ys = projection(lats, lons)
        P = np.column_stack((xs, ys)).astype(float)
    else:
        u = rec.get("start_node_id", rec.get("u", rec.get("from_node_id")))
        v = rec.get("end_node_id", rec.get("v", rec.get("to_node_id")))

        endpoints = []
        for node_id in (u, v):
            ll = node_latlon(road, node_id)
            if ll is not None:
                endpoints.append(ll)

        if len(endpoints) < 2:
            return np.empty((0, 2), dtype=float)

        lats = np.asarray([p[0] for p in endpoints], dtype=float)
        lons = np.asarray([p[1] for p in endpoints], dtype=float)
        xs, ys = projection(lats, lons)
        P = np.column_stack((xs, ys)).astype(float)

    P = P[np.all(np.isfinite(P), axis=1)]

    if P.shape[0] <= max_points_per_link:
        return P

    idx = np.linspace(0, P.shape[0] - 1, max_points_per_link).round().astype(int)
    return P[idx]


def append_points(P, offsets, xs, ys):
    offsets.append(len(xs))

    if P is None:
        return

    for row in P:
        if len(row) < 2:
            continue
        xs.append(float(row[0]))
        ys.append(float(row[1]))


# Return two plotting points for a road link for ease of visualization.
def link_endpoint_xy(
    road: Dict[str, Any],
    origin: Dict[str, Any],
    link_id: Any,
    fallback_points=None,
):

    rec = link_record(road, link_id)
    if not isinstance(rec, dict):
        return None, None

    u = rec.get("start_node_id", rec.get("u", rec.get("from_node_id")))
    v = rec.get("end_node_id", rec.get("v", rec.get("to_node_id")))

    ll0 = node_latlon(road, u)
    ll1 = node_latlon(road, v)

    if ll0 is not None and ll1 is not None:
        projection = make_projection(origin)
        lats = np.asarray([ll0[0], ll1[0]], dtype=float)
        lons = np.asarray([ll0[1], ll1[1]], dtype=float)
        xs, ys = projection(lats, lons)
        p0 = (float(xs[0]), float(ys[0]))
        p1 = (float(xs[1]), float(ys[1]))
        return p0, p1

    if fallback_points is not None and len(fallback_points) >= 2:
        p0 = (float(fallback_points[0][0]), float(fallback_points[0][1]))
        p1 = (float(fallback_points[-1][0]), float(fallback_points[-1][1]))
        return p0, p1

    return None, None


def parse_config_id(argv: List[str]) -> int:
    if len(argv) != 2:
        raise SystemExit(
            "Example: python3 export_graph_input.py 12"
        )

    try:
        return int(argv[1])
    except ValueError as exc:
        raise SystemExit(f"Config number must be an integer, got: {argv[1]}") from exc


def main():
    t0 = time.perf_counter()

    config_id = parse_config_id(sys.argv)
    config_path = PROJECT_DIR / f"configs/config{config_id}.json"
    out_path = PROJECT_DIR / f"intermediate/graph_input_config{config_id}.json"

    cfg, road, routing, origin0 = load_data(config_path)

    secondary_ids, primary_ids = classify_links(routing, RAMP_ONLY)

    graph = build_dual_graph(
        road=road,
        secondary_ids=secondary_ids,
        primary_ids=primary_ids,
        origin=origin0,
    )

    origin = fill_geometry(config_path, cfg, road)

    point_offsets = []
    point_x = []
    point_y = []

    link_plot_x0 = []
    link_plot_y0 = []
    link_plot_x1 = []
    link_plot_y1 = []

    for link_id in graph.link_ids:
        P = link_xy_points(
            road=road,
            origin=origin,
            link_id=link_id,
            max_points_per_link=MAX_POINTS_PER_LINK,
        )
        append_points(P, point_offsets, point_x, point_y)

        p0, p1 = link_endpoint_xy(
            road=road,
            origin=origin,
            link_id=link_id,
            fallback_points=P,
        )
        if p0 is None or p1 is None:
            link_plot_x0.append(float("nan"))
            link_plot_y0.append(float("nan"))
            link_plot_x1.append(float("nan"))
            link_plot_y1.append(float("nan"))
        else:
            link_plot_x0.append(p0[0])
            link_plot_y0.append(p0[1])
            link_plot_x1.append(p1[0])
            link_plot_y1.append(p1[1])

    point_offsets.append(len(point_x))

    background_link_ids = []
    background_offsets = []
    background_x = []
    background_y = []

    background_plot_x0 = []
    background_plot_y0 = []
    background_plot_x1 = []
    background_plot_y1 = []

    for raw_link_id in (road.get("links", {}) or {}).keys():
        link_id = normalize_id(raw_link_id)
        P = link_xy_points(
            road=road,
            origin=origin,
            link_id=link_id,
            max_points_per_link=MAX_POINTS_PER_LINK,
        )
        p0, p1 = link_endpoint_xy(
            road=road,
            origin=origin,
            link_id=link_id,
            fallback_points=P,
        )

        if p0 is None or p1 is None:
            continue

        background_link_ids.append(link_id)
        append_points(P, background_offsets, background_x, background_y)

        background_plot_x0.append(p0[0])
        background_plot_y0.append(p0[1])
        background_plot_x1.append(p1[0])
        background_plot_y1.append(p1[1])

    background_offsets.append(len(background_x))

    payload = {
        "schema": "junction_graph_input_v1",
        "source_config": str(config_path),
        "classification": {
            "ramp_only": bool(RAMP_ONLY),
            "num_secondary_ids": len(secondary_ids),
            "num_primary_ids": len(primary_ids),
        },
        "num_links": graph.n,
        "link_ids": [jsonable(x) for x in graph.link_ids],
        "row_offsets": graph.row_offsets,
        "col_indices": graph.col_indices,
        "link_type": graph.link_type,
        "point_offsets": point_offsets,
        "point_x": point_x,
        "point_y": point_y,
        "link_plot_x0": link_plot_x0,
        "link_plot_y0": link_plot_y0,
        "link_plot_x1": link_plot_x1,
        "link_plot_y1": link_plot_y1,
        "background_link_ids": [jsonable(x) for x in background_link_ids],
        "background_point_offsets": background_offsets,
        "background_x": background_x,
        "background_y": background_y,
        "background_plot_x0": background_plot_x0,
        "background_plot_y0": background_plot_y0,
        "background_plot_x1": background_plot_x1,
        "background_plot_y1": background_plot_y1,
        "summary": {
            "num_dual_vertices": graph.n,
            "num_dual_directed_edges": graph.directed_edge_count,
            "num_background_links": len(background_link_ids),
            "export_time_s": time.perf_counter() - t0,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    print(f"Config: {config_path}")
    print(f"Ramp only: {RAMP_ONLY}")
    print(f"Classified {len(secondary_ids)} secondary links and {len(primary_ids)} primary links.")
    print(f"Built dual graph with {graph.n} vertices and {graph.directed_edge_count} directed adjacency entries.")
    print(f"Wrote intermediate C++ graph input: {out_path}")
    print(f"Export time: {time.perf_counter() - t0:.3f} s")


if __name__ == "__main__":
    main()