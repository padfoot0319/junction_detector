# road_multi.py
from __future__ import annotations
import os
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ---- use your geometry helpers only ----
from libs.geometry_helpers import (
    parse_uint64_decimal, morton_to_latlon, decode_stream, latlon_to_xy
)

# ---------------- utilities ----------------
def _canon_id(x) -> str:
    return "" if x is None else str(x).strip()

def _as_float(x) -> Optional[float]:
    try:
        return float(x) if x is not None else None
    except Exception:
        return None

def _load_json(path: str) -> dict:
    try:
        import orjson as _json
        with open(path, "rb") as f:
            return _json.loads(f.read())
    except Exception:
        import json as _json
        with open(path, "r", encoding="utf-8") as f:
            return _json.load(f)

# --------------- per-tile build ---------------
def build_from_tile(json_file: str) -> Dict[str, Any]:
    """
    Parse ONE tile:
      - decode nodes (XOR node morton with tile center morton)
      - decode links via decode_stream(geometry, origin_morton, origin_alt_cm)
      - record relations
    Returns a dict with nodes, links, and this tile's origin.
    """
    raw = _load_json(json_file)
    tid = _canon_id(raw.get("here_tile_id"))

    # --- origin (ENU ref & seed for decode_stream) ---
    tc2d = raw.get("tile_center_here_2d_coordinate")
    if not tc2d:
        # Pre-decoded format: no Morton code, no node/link geometry — return a
        # skeleton tile that still carries link_lane_group_references.
        print(f"[ROAD] {json_file}: tile_center_here_2d_coordinate missing; "
              "returning empty tile (pre-decoded format).")
        return {
            "tile_id": tid,
            "origin_lat": None, "origin_lon": None, "origin_alt_m": 0.0,
            "nodes": {}, "links": {},
            "link_lane_group_references": _extract_llgr_from_tile_json(raw),
        }
    origin_morton = np.uint64(parse_uint64_decimal(tc2d))
    origin_lat, origin_lon = morton_to_latlon(origin_morton)
    origin_alt_cm = int((raw.get("tile_center_here_3d_coordinate") or {}).get("cm_from_wgs84_ellipsoid", 0))
    origin_alt_m  = float(origin_alt_cm) / 100.0

    # --- nodes ---
    nodes: Dict[str, Dict[str, Any]] = {}
    node_morton_by_id: Dict[str, np.uint64] = {}

    nin = raw.get("nodes_in_tile") or []
    if not isinstance(nin, list): nin = [nin]

    for i, n in enumerate(nin, start=1):
        nid = _canon_id(n.get("node_id")) or f"UNKNOWN_NODE_{i}"
        geom = n.get("geometry") or {}
        cstr = geom.get("here_2d_coordinate")

        lat = lon = None
        nmorton_abs = None
        if cstr:
            # IMPORTANT: nodes store DIFF to tile center; recover absolute morton
            nmorton_abs = np.uint64(origin_morton) ^ np.uint64(parse_uint64_decimal(cstr))
            la, lo = morton_to_latlon(nmorton_abs)
            lat, lon = float(la), float(lo)
            node_morton_by_id[nid] = nmorton_abs

        nodes[nid] = {
            "node_id": nid, "tile_id": tid,
            "lat": lat, "lon": lon, "morton": nmorton_abs,
            "out_links": [], "in_links": []
        }

    # --- links (geometry like lane_path_geometry) ---
    links: Dict[str, Dict[str, Any]] = {}
    lin = raw.get("links_starting_in_tile") or []
    if not isinstance(lin, list): lin = [lin]

    for i, L in enumerate(lin, start=1):
        lid = _canon_id(L.get("link_id")) or f"UNKNOWN_LINK_{i}"
        sN  = _canon_id(L.get("start_node_id"))
        eR  = L.get("end_node_ref") or {}
        eN  = _canon_id(eR.get("node_id"))

        geom = L.get("geometry")
        if not geom:
            continue

        lat_seq, lon_seq, _ = decode_stream(geom, origin_morton, origin_alt_cm)

        links[lid] = {
            "link_id": lid, "tile_id": tid,
            "start_node_id": sN, "end_node_id": eN,
            "lat": np.asarray(lat_seq, float).tolist(),
            "lon": np.asarray(lon_seq, float).tolist(),
            "length_m": _as_float(L.get("link_length_meters")),
        }

        # relations (create stubs for endpoints not present yet)
        for nid, role in ((sN, "out_links"), (eN, "in_links")):
            if not nid: continue
            if nid not in nodes:
                nodes[nid] = {"node_id": nid, "tile_id": tid,
                              "lat": None, "lon": None, "morton": None,
                              "out_links": [], "in_links": []}
            nodes[nid][role].append(lid)

    return {
        "tile_id": tid,
        "origin_lat": float(origin_lat), "origin_lon": float(origin_lon), "origin_alt_m": float(origin_alt_m),
        "nodes": nodes, "links": links, "link_lane_group_references": _extract_llgr_from_tile_json(raw),
    }

# --------------- merge + post-fix ---------------
def build_from_tiles(files: List[str], workers: Optional[int] = None) -> Dict[str, Any]:
    """
    Multi-tile builder:
      - merges nodes/links across tiles
      - fills missing endpoint-node coordinates from link endpoints
      - chooses a global ENU origin (first tile by default)
    """
    fs = list(files or [])
    out = {
        "origin_lat": None, "origin_lon": None, "origin_alt_m": 0.0,
        "nodes": {}, "links": {}, "tiles": [],
        # NEW: keep a dict mapping link_id -> list of lane_group_reference dicts
        "link_lane_group_refs": {}       # { link_local_ref: [ lane_group_reference, ... ] }
    }
    if not fs: return out

    max_workers = workers or min(32, (os.cpu_count() or 2) * 2)
    tiles: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(build_from_tile, p): p for p in fs}
        for fut in as_completed(futs):
            try:
                t = fut.result()
            except Exception as ex:
                print(f"[ROAD] skip {futs[fut]}: {ex}")
                continue
            tiles.append(t)

    # choose ENU origin (first tile)
    if tiles:
        t0 = tiles[0]
        out["origin_lat"]   = t0["origin_lat"]
        out["origin_lon"]   = t0["origin_lon"]
        out["origin_alt_m"] = t0["origin_alt_m"]

    # merge nodes/links
    for t in tiles:
        out["tiles"].append(t["tile_id"])

        # nodes
        for nid, n in t["nodes"].items():
            if nid not in out["nodes"]:
                out["nodes"][nid] = n
            else:
                dst = out["nodes"][nid]
                # keep first known position; update relations
                if dst.get("lat") is None and n.get("lat") is not None:
                    dst["lat"], dst["lon"], dst["morton"] = n["lat"], n["lon"], n["morton"]
                dst["out_links"] = list(set(dst["out_links"]) | set(n["out_links"]))
                dst["in_links"]  = list(set(dst["in_links"])  | set(n["in_links"]))

        # links
        for lid, L in t["links"].items():
            if lid not in out["links"]:
                out["links"][lid] = L

    # merge nodes/links + collect LLGR
    for t in tiles:
        out["tiles"].append(t["tile_id"])

        # nodes (unchanged)
        for nid, n in t["nodes"].items():
            if nid not in out["nodes"]:
                out["nodes"][nid] = n
            else:
                dst = out["nodes"][nid]
                if dst.get("lat") is None and n.get("lat") is not None:
                    dst["lat"], dst["lon"], dst["morton"] = n["lat"], n["lon"], n["morton"]
                dst["out_links"] = list(set(dst["out_links"]) | set(n["out_links"]))
                dst["in_links"]  = list(set(dst["in_links"])  | set(n["in_links"]))

        # links (unchanged)
        for lid, L in t["links"].items():
            if lid not in out["links"]:
                out["links"][lid] = L

        # NEW: merge link_lane_group_references
        for row in t.get("link_lane_group_references", []):
            lid = int(row["link_local_ref"])
            out["link_lane_group_refs"].setdefault(lid, [])
            out["link_lane_group_refs"][lid].extend(row.get("lane_group_references", []))

    # --- post-pass: fill missing endpoint nodes using link endpoints
    for lid, L in out["links"].items():
        la = L.get("lat") or []; lo = L.get("lon") or []
        if len(la) < 1: continue
        sN = L.get("start_node_id"); eN = L.get("end_node_id")

        # start node from first vertex
        if sN:
            n = out["nodes"].setdefault(sN, {"node_id": sN, "tile_id": "", "lat": None, "lon": None,
                                             "morton": None, "out_links": [], "in_links": []})
            if n.get("lat") is None or n.get("lon") is None:
                n["lat"], n["lon"] = float(la[0]), float(lo[0])
            if lid not in n["out_links"]:
                n["out_links"].append(lid)

        # end node from last vertex
        if eN:
            n = out["nodes"].setdefault(eN, {"node_id": eN, "tile_id": "", "lat": None, "lon": None,
                                             "morton": None, "out_links": [], "in_links": []})
            if n.get("lat") is None or n.get("lon") is None:
                n["lat"], n["lon"] = float(la[-1]), float(lo[-1])
            if lid not in n["in_links"]:
                n["in_links"].append(lid)

    return out

def _build_road_from_topo_geom_files(files: List[str]) -> Dict[str, Any]:
    """Parse topology_geometry files (pre-decoded format) into road nodes and links."""
    nodes: Dict[str, Any] = {}
    links: Dict[str, Any] = {}

    for fpath in files:
        if not os.path.exists(fpath):
            continue
        with open(fpath, encoding="utf-8") as f:
            raw = json.load(f)

        tile_id = str(raw.get("here_tile_id", ""))

        for node in (raw.get("nodes") or []):
            nid = str(node["node_id"])
            if nid not in nodes:
                nodes[nid] = {
                    "node_id": nid,
                    "tile_id": str(node.get("tile_id", tile_id)),
                    "lat": float(node["lat"]),
                    "lon": float(node["lon"]),
                    "morton": node.get("morton"),
                    "out_links": [],
                    "in_links": [],
                }

        for link in (raw.get("links") or []):
            lid = str(link["link_id"])
            if lid in links:
                continue
            sn = str(link["start_node_id"])
            en = str(link["end_node_id"])
            links[lid] = {
                "link_id": lid,
                "tile_id": tile_id,
                "start_node_id": sn,
                "end_node_id": en,
                "lat": link["lat"],
                "lon": link["lon"],
                "length_m": link.get("length_m"),
            }
            if sn in nodes and lid not in nodes[sn]["out_links"]:
                nodes[sn]["out_links"].append(lid)
            if en in nodes and lid not in nodes[en]["in_links"]:
                nodes[en]["in_links"].append(lid)

    print(f"[ROAD] _build_road_from_topo_geom_files: {len(links)} links, {len(nodes)} nodes from {len(files)} file(s).")
    return {"nodes": nodes, "links": links}


def _snap_node(nodes: dict, lat: float, lon: float, fallback_id: str, snap_deg: float = 5e-5) -> str:
    """Return an existing node key if one is within snap_deg of (lat, lon), else create a new one."""
    for nid, n in nodes.items():
        if n.get("lat") is None:
            continue
        if abs(n["lat"] - lat) < snap_deg and abs(n["lon"] - lon) < snap_deg:
            return nid
    nodes[fallback_id] = {
        "node_id": fallback_id, "tile_id": "", "lat": lat, "lon": lon,
        "morton": None, "out_links": [], "in_links": []
    }
    return fallback_id


def fill_road_from_sxy(road: Dict[str, Any], Sxy: Dict[str, Any], topo_geom_files: List[str] = None) -> None:
    """
    Populate road['nodes'] and road['links'] when the ref tiles were pre-decoded
    (no node/link geometry in the ref files).

    Priority:
      1) topology_geometry files  — exact road geometry with real node IDs
      2) lane-group reference geometry + endpoint snapping  — fallback approximation
    Modifies road in-place; no-op if links are already populated.
    """
    if road.get("links"):
        return

    # --- Priority 1: topology_geometry files ---
    if topo_geom_files:
        tg = _build_road_from_topo_geom_files(topo_geom_files)
        if tg.get("links"):
            road.setdefault("nodes", {}).update(tg["nodes"])
            road.setdefault("links", {}).update(tg["links"])
            print(f"[ROAD] fill_road_from_sxy: used topology_geometry ({len(road['links'])} links, {len(road['nodes'])} nodes).")
            return

    link_lg_refs = road.get("link_lane_group_refs") or {}
    if not link_lg_refs:
        return

    # {lane_group_id_int → (lat_array, lon_array)}
    lg_geom: Dict[int, tuple] = {}
    for ref in Sxy.get("reference", []):
        lgid = int(ref["lane_group_ref"])
        lats = ref.get("lat")
        lons = ref.get("lon")
        if lgid not in lg_geom and lats is not None and len(lats) > 0:
            lg_geom[lgid] = (np.asarray(lats, float), np.asarray(lons, float))

    nodes: Dict[str, Any] = road.setdefault("nodes", {})
    links: Dict[str, Any] = road.setdefault("links", {})

    filled = 0
    for link_id, lg_ref_list in link_lg_refs.items():
        # Pick the lane group with the most geometry points for best road shape
        best_lat = best_lon = None
        best_n = 0
        for lgr in lg_ref_list:
            lg_ref = lgr.get("lane_group_ref")
            if isinstance(lg_ref, dict):
                lgid = lg_ref.get("lane_group_id")
            else:
                lgid = lg_ref
            if lgid is None:
                continue
            geom = lg_geom.get(int(lgid))
            if geom is not None and len(geom[0]) > best_n:
                best_lat, best_lon = geom
                best_n = len(geom[0])
        lat_arr, lon_arr = best_lat, best_lon

        if lat_arr is None or len(lat_arr) == 0:
            continue

        sN = _snap_node(nodes, float(lat_arr[0]),  float(lon_arr[0]),  f"ps_{link_id}_s", snap_deg=5e-5)
        eN = _snap_node(nodes, float(lat_arr[-1]), float(lon_arr[-1]), f"ps_{link_id}_e", snap_deg=5e-5)

        nodes[sN]["out_links"].append(str(link_id))
        nodes[eN]["in_links"].append(str(link_id))

        links[str(link_id)] = {
            "link_id": str(link_id), "tile_id": "",
            "start_node_id": sN, "end_node_id": eN,
            "lat": lat_arr.tolist(), "lon": lon_arr.tolist(),
            "length_m": None,
        }
        filled += 1

    print(f"[ROAD] fill_road_from_sxy: created {filled} pseudo-links from lane-group geometry.")


import json

def _extract_llgr_from_tile_json(tile_json):
    """Return list[ {link_local_ref:int, lane_group_references:list[...]} ]."""
    return tile_json.get("link_lane_group_references", []) or []

import json, os, math, itertools
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------- helpers --------------------

def _as_frozendict(obj: Any) -> Tuple:
    """Turn nested dict/list into a hashable tuple for de-duplication."""
    if isinstance(obj, dict):
        return tuple(sorted((k, _as_frozendict(v)) for k, v in obj.items()))
    if isinstance(obj, list):
        return tuple(_as_frozendict(v) for v in obj)
    return obj  # primitives

def _dedup_list_of_dicts(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for itm in items:
        key = _as_frozendict(itm)
        if key in seen:
            continue
        seen.add(key)
        out.append(itm)
    return out

_FC_INT_MAP = {1: "FC_1", 2: "FC_2", 3: "FC_3", 4: "FC_4", 5: "FC_5"}

def _normalize_predecoded_link_attrs(d: dict) -> dict:
    """Normalize a pre-decoded link_parametric_attribution dict to standard attr format."""
    out = dict(d)
    fc = out.get("functional_class")
    if isinstance(fc, int):
        out["functional_class"] = _FC_INT_MAP.get(fc, f"FC_{fc}")
    return out


def _flatten_param_list(param_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    HERE parametric attribution is a list of single-key dicts.
    We flatten them into a single dict (keys stay distinct).
    If a key appears multiple times with incompatible values, we keep a list.
    """
    flat: Dict[str, Any] = {}
    for d in param_list or []:
        if not isinstance(d, dict) or len(d) != 1:
            # Keep as-is under a generic bucket
            flat.setdefault("_raw", []).append(d)
            continue
        k, v = next(iter(d.items()))
        if k not in flat:
            flat[k] = v
        else:
            # Merge collisions into a list, dedup lightly
            cur = flat[k]
            if isinstance(cur, list):
                if _as_frozendict(v) not in {_as_frozendict(x) for x in cur}:
                    cur.append(v)
            else:
                if _as_frozendict(v) != _as_frozendict(cur):
                    flat[k] = [cur, v]
    return flat

def _normalize_range(r: Dict[str, Any]) -> Dict[str, float]:
    """
    Normalize range fields into explicit floats in [0,1] when present.
    Keys observed in HERE attribution:
      - range_offset_from_start
      - range_offset_from_end
    """
    norm = {}
    if not r:
        return norm
    if "range_offset_from_start" in r:
        try:
            norm["from_start"] = float(r["range_offset_from_start"])
        except Exception:
            pass
    if "range_offset_from_end" in r:
        try:
            norm["from_end"] = float(r["range_offset_from_end"])
        except Exception:
            pass
    return norm

# -------------------- per-tile reader --------------------

def build_from_tile_attr(path: str) -> Dict[str, Any]:
    """
    Read a single HERE attribution tile (structure with link_attribution / strand_attribution)
    and convert to a normalized dict:
      {
        "tile_id": <int or str>,
        "origin_lat": None, "origin_lon": None, "origin_alt_m": 0.0,
        "links": {
           <link_id>: {
              "link_id": <int>,
              "tile_id": <tile>,
              "parametric": [ {direction, range, attrs} ],
              "point":      [ {direction, attrs} ],
           },
           ...
        },
        "strands": [
           {
             "strand_id": <int>,
             "first_link_id": <int>,
             "first_link_orientation": "FORWARD"|"BACKWARD",
             "first_link_start": {location_offset_from_start?: float},
             "additional_links": [ {link_id, tile_id, orientation} ],
             "last_link_end": {location_offset_from_start?: float},
             "attrs": { ... flattened ... }
           },
           ...
        ]
      }
    """
    with open(path, "r", encoding="utf-8") as f:
        J = json.load(f)

    tile_id = J.get("here_tile_id")
    out = {
        "tile_id": tile_id,
        "origin_lat": None, "origin_lon": None, "origin_alt_m": 0.0,
        "links": {},
        "strands": [],
    }

    # --- link_attribution
    for la in J.get("link_attribution", []) or []:
        link_id = la.get("link_local_ref")
        if link_id is None:
            continue
        L = out["links"].setdefault(
            int(link_id),
            {"link_id": int(link_id), "tile_id": tile_id, "parametric": [], "point": []},
        )

        for pa in la.get("parametric_attribution", []) or []:
            direction = pa.get("applies_to_direction")  # BOTH | FORWARD | BACKWARD
            rng = _normalize_range(pa.get("applies_to_range") or {})
            lpa_raw = pa.get("link_parametric_attribution") or []
            if isinstance(lpa_raw, dict):
                attrs = _normalize_predecoded_link_attrs(lpa_raw)
            else:
                attrs = _flatten_param_list(lpa_raw)
            L["parametric"].append({"direction": direction, "range": rng, "attrs": attrs})

        for pt in la.get("point_attribution", []) or []:
            direction = pt.get("applies_to_direction")
            lpa_raw = pt.get("link_point_attribution") or []
            if isinstance(lpa_raw, dict):
                attrs = _normalize_predecoded_link_attrs(lpa_raw)
            else:
                attrs = _flatten_param_list(lpa_raw)
            L["point"].append({"direction": direction, "attrs": attrs})

    # --- strand_attribution
    for sa in J.get("strand_attribution", []) or []:
        sid_obj = sa.get("strand_attribution_id") or {}
        sid = sid_obj.get("strand_attribution_id")
        first_link_id = sa.get("first_link_id")
        first_link_orientation = sa.get("first_link_orientation_in_strand")
        first_link_start = {}
        if isinstance(sa.get("first_link_start"), dict):
            first_link_start = {
                k: float(v) for k, v in sa["first_link_start"].items() if isinstance(v, (int, float))
            }

        last_link_end = {}
        if isinstance(sa.get("last_link_end"), dict):
            last_link_end = {
                k: float(v) for k, v in sa["last_link_end"].items() if isinstance(v, (int, float))
            }

        add_links = []
        for add in sa.get("additional_link_refs", []) or []:
            refs = add.get("additional_link_refs") or {}
            add_links.append({
                "link_id": refs.get("link_id"),
                "tile_id": refs.get("link_here_tile_id"),
                "orientation": add.get("link_orientation_in_strand"),
            })

        attrs = _flatten_param_list(sa.get("strand_attribution") or [])

        out["strands"].append({
            "strand_id": sid,
            "first_link_id": first_link_id,
            "first_link_orientation": first_link_orientation,
            "first_link_start": first_link_start,
            "additional_links": add_links,
            "last_link_end": last_link_end,
            "attrs": attrs,
        })

    # de-dup parametric/point lists per link
    for L in out["links"].values():
        L["parametric"] = _dedup_list_of_dicts(L["parametric"])
        L["point"] = _dedup_list_of_dicts(L["point"])

    return out

# -------------------- multi-tile merge --------------------

def build_from_tiles_attr(files: List[str], workers: Optional[int] = None) -> Dict[str, Any]:
    """
    Multi-tile attribution builder:
      - merges link parametric/point attribution across tiles (union + dedup)
      - merges strand_attribution as a flat list (dedup)
      - preserves HERE tile ids in `tiles`
      - keeps the same outer shape as your geometry builder for pipeline symmetry
    NOTE: Attribution tiles usually don't carry geometry or node coordinates;
          'nodes' are left empty and no ENU origin is derived.
    """
    fs = list(files or [])
    out = {"origin_lat": None, "origin_lon": None, "origin_alt_m": 0.0,
           "nodes": {}, "links": {}, "tiles": [], "strands": []}
    if not fs:
        return out

    max_workers = workers or min(32, (os.cpu_count() or 2) * 2)
    tiles: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(build_from_tile_attr, p): p for p in fs}
        for fut in as_completed(futs):
            try:
                t = fut.result()
            except Exception as ex:
                print(f"[ATTR] skip {futs[fut]}: {ex}")
                continue
            tiles.append(t)

    # merge
    for t in tiles:
        if t.get("tile_id") is not None:
            out["tiles"].append(t["tile_id"])

        # links: union + dedup of parametric / point
        for lid, L in t.get("links", {}).items():
            D = out["links"].setdefault(
                lid, {"link_id": lid, "tile_id": None, "parametric": [], "point": []}
            )
            # tile_id is not unique across tiles for a link seen in many tiles; keep first
            if D["tile_id"] is None:
                D["tile_id"] = L.get("tile_id")

            D["parametric"].extend(L.get("parametric", []))
            D["point"].extend(L.get("point", []))

        # strands: append; dedup later
        out["strands"].extend(t.get("strands", []))

    # global dedup after merge
    for D in out["links"].values():
        D["parametric"] = _dedup_list_of_dicts(D["parametric"])
        D["point"] = _dedup_list_of_dicts(D["point"])

    out["strands"] = _dedup_list_of_dicts(out["strands"])

    return out


# --------------- plotting: GEO + ENU ---------------
def plot_geo(road: dict,
             *, show_nodes=True, show_links=True,
             show_node_ids=False, show_link_ids=False,
             figsize=(7, 7),
             window_rect: Optional[Tuple[float, float, float, float]] = None,
             match_node_link_colors: bool = False,
             single_color: Optional[Any] = None):
    """
    Plot in geo coordinates.

    window_rect: (lat_ul, lon_ul, lat_lr, lon_lr) in degrees.
    match_node_link_colors: if True, color node endpoints to match their links.
    single_color: if given, use the same color for all links and nodes.
    """
    fig, ax = plt.subplots(figsize=figsize)

    links_dict = road.get("links") or {}
    nodes_dict = road.get("nodes") or {}

    view_links, view_nodes = _select_view_ids(road, window_rect)

    # Colors per link
    link_color = _link_color_map(ax, view_links, single_color=single_color)

    # --- links
    if show_links:
        for lid in view_links:
            L = links_dict.get(lid)
            if not L: continue
            la, lo = L.get("lat") or [], L.get("lon") or []
            if len(la) < 2: continue
            c = link_color[lid]
            ax.plot(lo, la, linewidth=1.2, alpha=0.9, color=c)
            if show_link_ids:
                j = len(lo)//2
                ax.text(lo[j], la[j], str(lid), fontsize=7, ha="center", va="center",
                        alpha=0.85, color=c)

    # --- nodes
    if show_nodes:
        if match_node_link_colors:
            # draw endpoints in their link colors
            drawn = set()
            for lid in view_links:
                L = links_dict.get(lid)
                if not L: continue
                c = link_color[lid]
                for nid in (L.get("start_node_id"), L.get("end_node_id")):
                    if not nid or nid in drawn or nid not in nodes_dict: continue
                    n = nodes_dict[nid]
                    la, lo = n.get("lat"), n.get("lon")
                    if la is None or lo is None: continue
                    if view_nodes and (nid not in view_nodes): continue
                    ax.scatter([lo], [la], s=18.0, alpha=0.95, zorder=3, color=c, edgecolors='none')
                    if show_node_ids:
                        ax.text(lo, la, str(nid), fontsize=7, ha="left", va="bottom",
                                alpha=0.9, color=c)
                    drawn.add(nid)
        else:
            # single or default color for all nodes
            xs, ys, labels = [], [], []
            for nid in view_nodes:
                n = nodes_dict.get(nid)
                if not n: continue
                la, lo = n.get("lat"), n.get("lon")
                if la is None or lo is None: continue
                xs.append(lo); ys.append(la); labels.append(str(nid))
            if xs:
                ax.scatter(xs, ys, s=8.0, alpha=0.95, zorder=3,
                           color=(single_color if single_color is not None else None),
                           edgecolors='none')
                if show_node_ids:
                    for nid, x0, y0 in zip(labels, xs, ys):
                        ax.text(x0, y0, nid, fontsize=7, ha="left", va="bottom", alpha=0.9,
                                color=(single_color if single_color is not None else None))

    ax.set_xlabel("Longitude [deg]"); ax.set_ylabel("Latitude [deg]")
    ax.set_title("Road Topology (Geo debug)")
    ax.grid(True, alpha=0.25); ax.set_aspect("equal", adjustable="box")
    return ax

def plot_enu(road: dict,
             *, show_nodes=True, show_links=True,
             show_node_ids=False, show_link_ids=False,
             figsize=(8, 8),
             origin=None,
             link_style="--",
             node_style="open",
             link_color="k",
             highlight_links=None,
             highlight_color="r",
             window_rect=None,
             draw_rect=True,
             # NEW: optionally restrict nodes shown
             restrict_nodes=None):
    """
    ENU plot of road links & nodes.
    Now supports BOTH:
      - node/link geometry stored as ENU x/y
      - node/link geometry stored as lat/lon (converted using origin)
    """

    import numpy as np
    import matplotlib.pyplot as plt

    lat0 = float(road["origin_lat"]) if origin is None else float(origin[0])
    lon0 = float(road["origin_lon"]) if origin is None else float(origin[1])
    h0   = float(road["origin_alt_m"]) if origin is None else float(origin[2])

    fig, ax = plt.subplots(figsize=figsize)

    # --- normalize highlight list to strings ---
    highlight_links_str = {str(l) for l in (highlight_links or [])}

    # --- normalize nodes/links containers (dict expected) ---
    nodes_all = road.get("nodes") or {}
    links_all = road.get("links") or {}

    # if nodes_all is a list -> convert to dict keyed by node_id
    if isinstance(nodes_all, list):
        nodes_all = {int(n["node_id"]): n for n in nodes_all if isinstance(n, dict) and "node_id" in n}
    if isinstance(links_all, list):
        # if ever happens
        links_all = {str(L.get("link_id", i)): L for i, L in enumerate(links_all)}

    # --- helpers: ENU fetch with lat/lon fallback ---
    def _node_xy(n):
        # preferred: already ENU
        if isinstance(n, dict) and ("x" in n) and ("y" in n) and (n["x"] is not None) and (n["y"] is not None):
            return float(n["x"]), float(n["y"])
        # fallback: lat/lon
        la, lo = (n.get("lat"), n.get("lon")) if isinstance(n, dict) else (None, None)
        if la is None or lo is None:
            return None
        xx, yy = latlon_to_xy(np.asarray([la], float), np.asarray([lo], float), lat0, lon0, h0, None)
        return float(xx[0]), float(yy[0])

    def _link_xy(L):
        # preferred: already ENU polyline
        if isinstance(L, dict) and ("x" in L) and ("y" in L):
            x = L.get("x") or []
            y = L.get("y") or []
            if len(x) >= 2 and len(y) >= 2:
                return np.asarray(x, float), np.asarray(y, float)
        # fallback: lat/lon polyline
        la, lo = (L.get("lat") or [], L.get("lon") or [])
        if len(la) < 2 or len(lo) < 2:
            return None
        x, y = latlon_to_xy(np.asarray(la, float), np.asarray(lo, float), lat0, lon0, h0, None)
        return x, y

    # --- view sets ---
    view_links = set(links_all.keys())
    view_nodes = set(nodes_all.keys())

    # optional restriction (NEW): show only these nodes
    if restrict_nodes is not None:
        restrict_nodes = {str(v).strip().strip("'").strip('"') for v in restrict_nodes}
        view_nodes = {str(v) for v in view_nodes}.intersection(restrict_nodes)


    # window_rect logic (still GEO-based) — keep it if you still use GEO windows.
    # If you now window in ENU, handle that outside and pass restrict_nodes / highlight_links.
    # (Leaving your existing window_rect code unchanged is OK if you still provide GEO coords.)

    # --- links ---
    if show_links:
        for lid in view_links:
            L = links_all.get(lid)
            if not L:
                continue
            xy = _link_xy(L)
            if xy is None:
                continue
            x, y = xy

            lid_str = str(lid)
            this_color = highlight_color if lid_str in highlight_links_str else link_color
            ax.plot(x, y, link_style, linewidth=1.2, alpha=0.9, color=this_color)

            if show_link_ids:
                j = len(x) // 2
                ax.text(float(x[j]), float(y[j]), str(lid), fontsize=7,
                        ha="center", va="center", alpha=0.85, color=this_color)

    # --- nodes ---
    if show_nodes:
        xs, ys, labels = [], [], []
        for nid in view_nodes:
            n = nodes_all.get(nid)
            if not n:
                continue
            p = _node_xy(n)
            if p is None:
                continue
            xs.append(p[0]); ys.append(p[1]); labels.append(str(nid))

        if xs:
            if node_style == "open":
                ax.scatter(xs, ys, s=28, facecolors="none", edgecolors=link_color, zorder=3, linewidths=1.0)
            else:
                ax.scatter(xs, ys, s=28, c=link_color, zorder=3)
            if show_node_ids:
                for nid, x0, y0 in zip(labels, xs, ys):
                    ax.text(x0, y0, nid, fontsize=7, ha="left", va="bottom", alpha=0.9, color=link_color)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]")
    ax.set_title("Road Topology (ENU)")
    ax.grid(True, alpha=0.25)
    return ax




# ---------------- helpers for filtering & colors ----------------
def _in_rect(lat: Optional[float], lon: Optional[float],
             rect: Optional[Tuple[float, float, float, float]]) -> bool:
    """
    rect = (lat_ul, lon_ul, lat_lr, lon_lr) in geo degrees.
    Returns True if (lat, lon) is inside (inclusive). Robust to unordered inputs.
    """
    if rect is None or lat is None or lon is None:
        return False
    lat1, lon1, lat2, lon2 = rect
    lat_min, lat_max = min(lat1, lat2), max(lat1, lat2)
    lon_min, lon_max = min(lon1, lon2), max(lon1, lon2)
    return (lat_min <= lat <= lat_max) and (lon_min <= lon <= lon_max)


def _select_view_ids(road: dict,
                     window_rect: Optional[Tuple[float, float, float, float]]):
    """
    Decide which link IDs and node IDs are included in the view.

    Rule: include a link if its start OR end node is inside the rectangle.
    If window_rect is None -> include all.
    Nodes included are endpoints of the included links (even if outside rect).
    """
    links = road.get("links") or {}
    nodes = road.get("nodes") or {}

    if not window_rect:
        # Full view
        return set(links.keys()), set(nodes.keys())

    # Map nid -> (lat, lon)
    node_pos = {
        nid: (n.get("lat"), n.get("lon"))
        for nid, n in nodes.items()
    }

    include_links = set()
    include_nodes = set()

    for lid, L in links.items():
        sN = L.get("start_node_id")
        eN = L.get("end_node_id")
        inside = False

        if sN and sN in node_pos:
            la, lo = node_pos[sN]
            if _in_rect(la, lo, window_rect):
                inside = True
        if not inside and eN and eN in node_pos:
            la, lo = node_pos[eN]
            if _in_rect(la, lo, window_rect):
                inside = True

        if inside:
            include_links.add(lid)
            if sN: include_nodes.add(sN)
            if eN: include_nodes.add(eN)

    return include_links, include_nodes


def _link_color_map(ax, link_ids, *, single_color=None):
    """
    Build a consistent color mapping for link ids.
    - If single_color is given: use it for all.
    - Else: use the current prop cycle.
    """
    cmap = {}
    if single_color is not None:
        for lid in link_ids:
            cmap[lid] = single_color
        return cmap

    # pull cycle colors deterministically
    # we rely on the axes prop_cycle, and loop over link_ids
    prop_cycle = plt.rcParams.get('axes.prop_cycle', None)
    colors = []
    if prop_cycle is not None:
        colors = list(prop_cycle.by_key().get('color', []))
    if not colors:
        colors = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9']

    for k, lid in enumerate(sorted(link_ids)):
        cmap[lid] = colors[k % len(colors)]
    return cmap

import ast
from typing import Any, Dict, List, Set

# ---------- tiny, safe expression compiler ----------

class _PredCompiler(ast.NodeVisitor):
    """Compile a boolean expression into a Python callable(attrs_dict)->bool, safely."""
    ALLOWED_CMPOPS = (ast.Eq, ast.NotEq, ast.In, ast.NotIn)
    ALLOWED_BOOLOPS = (ast.And, ast.Or)

    def __init__(self):
        super().__init__()

    def compile(self, expr: str):
        tree = ast.parse(expr, mode="eval")
        code = self._emit(tree.body)

        def predicate(attrs: Dict[str, Any]) -> bool:
            try:
                return bool(code(attrs))
            except Exception:
                return False
        return predicate

    # ---- helpers ----

    @staticmethod
    def _get_nested(attrs: Dict[str, Any], dotted: str) -> Any:
        """Fetch attrs['a']['b'] via 'a.b'; return None if missing at any level."""
        cur: Any = attrs
        for part in dotted.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return None
        return cur

    # ---- emitters for safe AST nodes ----

    def _emit(self, node):
        if isinstance(node, ast.BoolOp) and isinstance(node.op, self.ALLOWED_BOOLOPS):
            parts = [self._emit(v) for v in node.values]
            if isinstance(node.op, ast.And):
                return lambda attrs: all(p(attrs) for p in parts)
            else:  # Or
                return lambda attrs: any(p(attrs) for p in parts)

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            inner = self._emit(node.operand)
            return lambda attrs: not inner(attrs)

        if isinstance(node, ast.Compare):
            # Only support simple a [op] b (no chained a<b<c)
            if len(node.ops) != 1 or len(node.comparators) != 1:
                raise ValueError("Only single comparisons are supported")
            op, rhs_node = node.ops[0], node.comparators[0]
            left = self._emit(node.left)
            right = self._emit(rhs_node)

            if isinstance(op, ast.Eq):
                return lambda attrs: left(attrs) == right(attrs)
            if isinstance(op, ast.NotEq):
                return lambda attrs: left(attrs) != right(attrs)
            if isinstance(op, ast.In):
                return lambda attrs: left(attrs) in right(attrs)
            if isinstance(op, ast.NotIn):
                return lambda attrs: left(attrs) not in right(attrs)
            raise ValueError(f"Operator {type(op).__name__} not allowed")

        if isinstance(node, ast.Constant):
            return lambda attrs, v=node.value: v

        if isinstance(node, ast.Name):
            name = node.id
            # allow bare true/false/null style names
            if name in ("true", "True"):
                return lambda attrs: True
            if name in ("false", "False"):
                return lambda attrs: False
            if name in ("none", "None", "null"):
                return lambda attrs: None
            # otherwise treat as top-level key
            return lambda attrs, k=name: attrs.get(k, None)

        if isinstance(node, ast.Call):
            # disallow function calls for safety
            raise ValueError("Function calls are not allowed")

        if isinstance(node, ast.Attribute) or isinstance(node, ast.Subscript):
            # We'll normalize dotted access via Attribute into a string and read via _get_nested
            path = self._to_dotted(node)
            return lambda attrs, p=path: self._get_nested(attrs, p)

        if isinstance(node, ast.BinOp):
            # disallow arithmetic; not needed for attribute queries
            raise ValueError("Arithmetic is not allowed")

        if isinstance(node, ast.Expr):
            return self._emit(node.value)

        # Also allow strings like "accessible_by.pedestrians"
        if isinstance(node, ast.JoinedStr):
            raise ValueError("f-strings are not allowed")

        raise ValueError(f"Unsupported expression fragment: {ast.dump(node)}")

    def _to_dotted(self, node) -> str:
        """Turn nested Attribute/Subscript chain into dotted.name segments if possible."""
        parts: List[str] = []

        def walk(n):
            if isinstance(n, ast.Attribute):
                walk(n.value)
                parts.append(n.attr)
            elif isinstance(n, ast.Name):
                parts.append(n.id)
            elif isinstance(n, ast.Subscript):
                walk(n.value)
                # Only allow constant string keys: obj['key']
                if isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str):
                    parts.append(n.slice.value)
                else:
                    raise ValueError("Only string literal subscripts are allowed (e.g., a['b'])")
            else:
                raise ValueError("Unsupported nested access")
        walk(node)
        return ".".join(parts)

# assumes _PredCompiler and _flatten_param_list from before are in scope

from typing import Any, Dict, Iterable, Set

# --- helpers you already have or similar ---
# _PredCompiler: compiles safe boolean expr -> callable(attrs)->bool
# _flatten_param_list: flattens HERE's list of single-key dicts into one dict

def _flatten_nested(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Recursively flatten nested dicts into dotted keys."""
    out: Dict[str, Any] = {}
    for k, v in (d or {}).items():
        kk = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_nested(v, kk))
        else:
            out[kk] = v
    return out

def _candidate_attr_dicts(item: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """
    Yield all plausible attribution dicts under a parametric item:
      - item['attrs'] (already-flattened by your tile reader)
      - item['link_parametric_attribution'] (raw list -> flatten)
      - item['additional_link_parametric_attribution'] (raw list -> flatten)
    """
    a = item.get("attrs")
    if isinstance(a, dict) and a:
        yield a
    for k in ("link_parametric_attribution", "additional_link_parametric_attribution"):
        if k in item and isinstance(item[k], list) and item[k]:
            flat = _flatten_param_list(item[k])
            if flat:
                yield flat

def _merge_attrs_union(acc: dict, src: dict) -> None:
    """Union-merge src into acc.
    - scalars: keep if equal, else store {a, b}
    - sets: union
    - lists: extend with dedup (works for list-of-dicts too)
    - dicts: recursive merge
    - mixed types: normalize to list and dedup
    """
    def _to_list(x):
        if x is None:
            return []
        if isinstance(x, list):
            return x
        if isinstance(x, set):
            return list(x)
        return [x]

    for k, v in (src or {}).items():
        if v is None:
            continue

        if k not in acc:
            # first occurrence: deep copy lists/dicts so we don't alias
            acc[k] = v.copy() if isinstance(v, dict) else (v[:] if isinstance(v, list) else v)
            continue

        a = acc[k]

        # same simple scalar -> keep
        if isinstance(a, (int, float, str, bool)) and isinstance(v, (int, float, str, bool)):
            if a != v:
                acc[k] = {a, v}
            continue

        # set ∪ set
        if isinstance(a, set) and isinstance(v, set):
            a |= v
            acc[k] = a
            continue

        # dict ⇔ dict: recursive
        if isinstance(a, dict) and isinstance(v, dict):
            _merge_attrs_union(a, v)
            acc[k] = a
            continue

        # list ⇔ list: extend with dedup (supports list-of-dicts)
        if isinstance(a, list) and isinstance(v, list):
            for item in v:
                if item not in a:   # list equality works for dict elements
                    a.append(item)
            acc[k] = a
            continue

        # mixed types: normalize both to lists and dedup
        la = _to_list(a)
        lv = _to_list(v)
        for item in lv:
            if item not in la:
                la.append(item)
        acc[k] = la


def find_links_by_attr_query(routing: Dict[str, Any], expr: str, mode: str = "link_union") -> Set[int]:
    """
    Find link_ids that satisfy `expr`.

    Modes:
      - "any_item": predicate must be true for at least one parametric item independently
      - "link_union" (default): predicate evaluated on link-level union of all parametric attrs
        (so attributes can come from different parametric sub-ranges)

    Example:
      functional_class == "FC_1" and (is_ramp == true or is_within_interchange == true)
    """
    pred = _PredCompiler().compile(expr)
    hits: Set[int] = set()

    for lid, L in (routing.get("links") or {}).items():
        params = L.get("parametric") or []

        if mode == "any_item":
            matched = False
            for item in params:
                for cand in _candidate_attr_dicts(item):
                    if pred(cand):
                        hits.add(int(lid))
                        matched = True
                        break
                if matched:
                    break
            continue

        # mode == "link_union"
        union_attrs: Dict[str, Any] = {}
        for item in params:
            for cand in _candidate_attr_dicts(item):
                _merge_attrs_union(union_attrs, cand)

        if union_attrs and pred(union_attrs):
            hits.add(int(lid))

    return hits

def build_file_lists_from_config(cfg):
    """
    Normalizes config so that we always end up with:
        cfg["geomFiles"], cfg["topoFiles"],
        cfg["attrFiles"], cfg["refFiles"],
        cfg["routeFiles"]

    Supports two modes:
    1) Explicit lists: geomFiles/topoFiles/... already present.
    2) Prefix + tiles: geomPrefix/topoPrefix/... + tiles[]
    """

    # --- Case 1: old style, explicit file lists already given ---
    if "geomFiles" in cfg and "topoFiles" in cfg and "attrFiles" in cfg \
       and "refFiles" in cfg and "routeFiles" in cfg:
        # Ensure they exist as lists and just return them
        geomFiles  = cfg["geomFiles"]
        topoFiles  = cfg["topoFiles"]
        attrFiles  = cfg["attrFiles"]
        refFiles   = cfg["refFiles"]
        routeFiles = cfg["routeFiles"]
        return geomFiles, topoFiles, attrFiles, refFiles, routeFiles

    # --- Case 2: new style, prefix + tiles ---
    tiles = cfg.get("tiles")
    if tiles is None:
        raise ValueError(
            "Config must either contain explicit '*Files' lists or a 'tiles' array "
            "together with *Prefix entries."
        )

    def make_file_list(prefix_key: str):
        prefix = cfg.get(prefix_key)
        if prefix is None:
            # You can choose to raise here instead of silently returning []
            return []
        return [f"{prefix}{tile}.json" for tile in tiles]

    geomFiles     = make_file_list("geomPrefix")
    topoFiles     = make_file_list("topoPrefix")
    attrFiles     = make_file_list("attrPrefix")
    refFiles      = make_file_list("refPrefix")
    routeFiles    = make_file_list("routePrefix")
    topoGeomFiles = make_file_list("topoGeomPrefix")

    # For downstream code that expects the old keys:
    cfg["geomFiles"]     = geomFiles
    cfg["topoFiles"]     = topoFiles
    cfg["attrFiles"]     = attrFiles
    cfg["refFiles"]      = refFiles
    cfg["routeFiles"]    = routeFiles
    cfg["topoGeomFiles"] = topoGeomFiles

    return geomFiles, topoFiles, attrFiles, refFiles, routeFiles

def _pick_admin_area(attrs: Dict[str, Any]) -> (Optional[int], Optional[str]):
    rel = attrs.get("administrative_area_relationship", {})
    within = rel.get("within_administrative_area", {})
    ref = within.get("administrative_area_ref", {})
    return ref.get("administrative_area_id"), ref.get("administrative_area_partition_id")


def summarize_routing_link(r: Dict[str, Any]) -> Dict[str, Any]:
    """
    r is routing['links'][link_id] entry:
      {'link_id':..., 'tile_id':..., 'parametric':[{'direction':..., 'range':..., 'attrs':{...}}, ...], 'point':[...]}

    Returns a flat summary + retains parametric list.
    """
    out = {
        "link_id": int(r["link_id"]),
        "tile_id": int(r.get("tile_id", 0)),
        "functional_class": None,
        "accessible_by": None,
        "is_ramp": False,
        "is_within_interchange": False,
        "is_urban": False,
        "built_up_area_road": False,
        "admin_area_id": None,
        "admin_area_partition_id": None,
        "routing_parametric": r.get("parametric", []),  # keep raw
    }

    param = r.get("parametric", []) or []

    # Prefer "BOTH" + empty range for global properties
    def score(entry: Dict[str, Any]) -> int:
        d = entry.get("direction")
        rg = entry.get("range") or {}
        return int(d == "BOTH") * 2 + int(rg == {}) * 1

    param_sorted = sorted(param, key=score, reverse=True)

    for p in param_sorted:
        attrs = p.get("attrs", {}) or {}

        if out["functional_class"] is None and "functional_class" in attrs:
            out["functional_class"] = attrs.get("functional_class")

        if out["accessible_by"] is None and "accessible_by" in attrs:
            out["accessible_by"] = attrs.get("accessible_by")

        if attrs.get("is_ramp") is True:
            out["is_ramp"] = True

        if attrs.get("is_within_interchange") is True:
            out["is_within_interchange"] = True

        if attrs.get("is_urban") is True:
            out["is_urban"] = True

        # built_up_area_road can be dict like {"is_built_up_area_road": True} or {"is_verified": True}
        if "built_up_area_road" in attrs:
            bu = attrs["built_up_area_road"] or {}
            if isinstance(bu, dict):
                out["built_up_area_road"] = out["built_up_area_road"] or bool(
                    bu.get("is_built_up_area_road", False) or bu.get("is_verified", False)
                )
            elif isinstance(bu, bool):
                out["built_up_area_road"] = out["built_up_area_road"] or bu

        # admin area: keep first preferred one (often BOTH/{} wins)
        if out["admin_area_id"] is None:
            aid, apart = _pick_admin_area(attrs)
            if aid is not None:
                out["admin_area_id"] = aid
                out["admin_area_partition_id"] = apart

    return out


def build_routing_link_summary_map(routing: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """
    Returns: link_id -> summary dict
    """
    out = {}
    for lid, entry in (routing.get("links", {}) or {}).items():
        # lid may already be int, but entry has link_id too
        s = summarize_routing_link(entry)
        out[s["link_id"]] = s
    return out

def attach_routing_to_links_df(here_link_df: pd.DataFrame, routing_summary: Dict[int, Dict[str, Any]]) -> pd.DataFrame:
    add_df = pd.DataFrame(list(routing_summary.values()))
    # ensure join key exists
    assert "link_id" in here_link_df.columns, "here_link_df must have 'link_id' to join routing attrs"
    merged = here_link_df.merge(add_df, on="link_id", how="left", suffixes=("", "_routing"))
    return merged