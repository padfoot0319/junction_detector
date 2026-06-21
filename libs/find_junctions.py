from __future__ import annotations
import math
import numbers
from dataclasses import dataclass, field
from itertools import combinations
from typing import List, Tuple, Set, Dict, Iterable, Optional, Any

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from math import atan2, pi
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
from math import atan2, pi
from collections import defaultdict

from libs.lane_group_helpers import (
    reorient_lane_groups_result_to_graph,
    prepare_lane_groups_for_link,
    compute_boring_road_nodes,
)

@dataclass
class JunctionNX:
    jid: int
    itype: str
    n_legs: int
    centroid: Tuple[float, float]
    seed_node_id: int
    node_ids: List[int]          # CP node ids in CCW order
    edge_ids: List[int]
    area: float
    min_angle_deg: float
    max_angle_deg: float
    radius_m: float
    directed_ok: bool = True
    lanes: List[Dict[str, Any]] = field(default_factory=list)
    cp_adj: np.ndarray = field(default_factory=lambda: np.zeros((0,0), dtype=object))

from math import pi
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple

def build_junctions_from_C(
    C: pd.DataFrame,
    Edges: pd.DataFrame,
    *,
    dead_ends: set[int] | None = None,
    junctions: set[int] | None = None,
    road_node_col: str = "road_node_id",
    road_x_col: str   = "road_x",
    road_y_col: str   = "road_y",
    cp_id_col: str    = "lane_node_id",
    cp_x_col: str     = "lane_x",
    cp_y_col: str     = "lane_y",
    d_node_col: str   = "d_node",
    link_id_col: str  = "link_id",
    link_end_col: str = "link_end",          # NEW (optional but preferred)
    cp_type_col: str  = "cp_type",           # NEW (optional; used to detect dead-ends)
    keep_lanes_blob: bool = True,

    edge_id_col: str  = "E",
    from_col: str     = "FromN",
    to_col: str       = "ToN",

    adj_value: str    = "id",
    multi_policy: str = "first",

    compute_in_out: bool = True,
) -> List["JunctionNX"]:

    if C is None or C.empty:
        return []

    need_C = [road_node_col, road_x_col, road_y_col, cp_id_col, cp_x_col, cp_y_col, d_node_col]
    miss_C = [c for c in need_C if c not in C.columns]
    if miss_C:
        raise KeyError(f"Missing required columns in C: {miss_C}")

    need_E = [from_col, to_col, edge_id_col]
    miss_E = [c for c in need_E if c not in Edges.columns]
    if miss_E:
        raise KeyError(f"Missing required columns in Edges: {miss_E}")

    # --------- directed edge lookup ----------
    pair_to_idxs: Dict[Tuple[int, int], List[int]] = {}
    pair_to_ids:  Dict[Tuple[int, int], List[Any]] = {}

    cols = Edges[[from_col, to_col, edge_id_col]].to_numpy()
    df_index = Edges.index.to_numpy()

    incoming_srcs: Dict[int, List[Tuple[int, Any]]] = {}
    outgoing_dsts: Dict[int, List[Tuple[int, Any]]] = {}

    for pos in range(cols.shape[0]):
        u, v, eid = cols[pos]
        if pd.isna(u) or pd.isna(v):
            continue
        u = int(u); v = int(v)
        idx = int(df_index[pos])

        pair_to_idxs.setdefault((u, v), []).append(idx)
        pair_to_ids.setdefault((u, v), []).append(eid)

        ref = (idx if adj_value == "index" else eid)
        incoming_srcs.setdefault(v, []).append((u, ref))
        outgoing_dsts.setdefault(u, []).append((v, ref))

    def _pick(lst_idx, lst_id):
        if not lst_idx:
            return -1
        if multi_policy == "all":
            return lst_idx if adj_value == "index" else lst_id
        return (lst_idx[0] if adj_value == "index" else lst_id[0])

    def lookup_dir(u: int, v: int):
        return _pick(pair_to_idxs.get((u, v), []), pair_to_ids.get((u, v), []))
    # --- helper: cp -> sorted unique gids from C ---
    def _cp2gids_from_C(Crid: pd.DataFrame) -> dict[int, list[int]]:
        cp2 = {}
        for cp, lg in Crid[["lane_node_id", "lane_groups"]].itertuples(index=False, name=None):
            try:
                cp_i = int(cp)
            except Exception:
                continue
            gids = []
            if isinstance(lg, (list, tuple, set, np.ndarray)):
                for g in lg:
                    try:
                        gids.append(int(g))
                    except Exception:
                        pass
            cp2.setdefault(cp_i, set()).update(gids)
        return {cp: sorted(gs) for cp, gs in cp2.items()}
    
    J_out: List["JunctionNX"] = []

    # --------- per road-node ----------
    for rid, G in C.groupby(road_node_col, sort=False):
        rid_i = int(rid)
        if dead_ends is not None and rid_i in dead_ends:
            junction_type = "deadend" 
        elif junctions is not None and rid_i in junctions:
            junction_type = "junction"
        else:
            continue

        rx = float(G[road_x_col].iloc[0])
        ry = float(G[road_y_col].iloc[0])

        # dedup CPs: keep nearest instance of each CP id
        G1 = (G.sort_values(d_node_col, ascending=True)
                .drop_duplicates(subset=[cp_id_col], keep="first")
                .reset_index(drop=True))

        cp_ids = G1[cp_id_col].astype(int).to_numpy()
        cp_xy  = G1[[cp_x_col, cp_y_col]].to_numpy(dtype=float)
        d_node = G1[d_node_col].to_numpy(dtype=float)
        # --------- ordering & geometry stats ----------
        if junction_type == "deadend":
            # no CCW, just stable nearest-first ordering
            order = np.argsort(d_node) if len(d_node) else np.array([], dtype=int)
            cp_ids_ord = cp_ids[order].tolist()
            G_ord = G1.iloc[order].reset_index(drop=True)

            min_angle_deg = 0.0
            max_angle_deg = 0.0
            radius_m = float(d_node[order].max()) if len(order) else 0.0
            area = 0.0

            n_legs = 1
            itype = "DeadEnd"

        else:
            # normal junction: CCW
            ang = np.mod(np.arctan2(cp_xy[:, 1] - ry, cp_xy[:, 0] - rx), 2 * pi)
            order = np.argsort(ang)
            cp_ids_ord = cp_ids[order].tolist()
            ang_ord = ang[order]
            d_ord = d_node[order]
            G_ord = G1.iloc[order].reset_index(drop=True)

            if len(ang_ord) >= 2:
                diffs = np.mod(np.diff(np.r_[ang_ord, ang_ord[0] + 2 * pi]), 2 * pi)
                min_angle_deg = float(np.degrees(diffs.min())) if len(diffs) else 0.0
                max_angle_deg = float(np.degrees(diffs.max())) if len(diffs) else 0.0
            else:
                min_angle_deg = max_angle_deg = 0.0

            radius_m = float(d_ord.max()) if len(d_ord) else 0.0

            # IMPORTANT FIX: count legs by LINK-ENDS if available
            if (link_id_col in G_ord.columns) and (link_end_col in G_ord.columns):
                # unique (link_id, link_end) pairs
                lid = G_ord[link_id_col].astype(str)
                lend = G_ord[link_end_col].astype(str)
                n_legs = int(pd.DataFrame({"lid": lid, "lend": lend}).drop_duplicates().shape[0])
            elif link_id_col in G_ord.columns:
                n_legs = count_legs_by_link_angle_clusters(G_ord, ang_ord, link_id_col, gap_deg=50.0)
            else:
                n_legs = len(cp_ids_ord)

            itype = {3: "T3-unknown", 4: "X4-unknown"}.get(n_legs, "Unknown")

            # polygon area (shoelace)
            if len(cp_ids_ord) >= 3:
                X = G_ord[cp_x_col].to_numpy()
                Y = G_ord[cp_y_col].to_numpy()
                area = 0.5 * float(abs(np.dot(X, np.roll(Y, -1)) - np.dot(Y, np.roll(X, -1))))
            else:
                area = 0.0

        # --------- directed adjacency among CPs ----------
        n = len(cp_ids_ord)
        adj = np.full((n, n), -1, dtype=object)
        for i in range(n):
            ui = int(cp_ids_ord[i])
            for j in range(n):
                if i == j:
                    continue
                vj = int(cp_ids_ord[j])
                adj[i, j] = lookup_dir(ui, vj)

        lanes_blob = (G_ord.to_dict(orient="records") if keep_lanes_blob else [])

        # --------- in/out CP computation ----------
        in_cps: List[int] = []
        out_cps: List[int] = []
        in_mask = [False] * n
        out_mask = [False] * n
        in_from: Dict[int, List[Tuple[int, Any]]] = {}
        out_to:  Dict[int, List[Tuple[int, Any]]] = {}

        if compute_in_out and n > 0:
            S = set(int(x) for x in cp_ids_ord)
            for idx_cp, cp in enumerate(cp_ids_ord):
                cp = int(cp)
                inc_edges = [(u, ref) for (u, ref) in incoming_srcs.get(cp, []) if int(u) not in S]
                out_edges = [(v, ref) for (v, ref) in outgoing_dsts.get(cp, []) if int(v) not in S]

                is_in = (len(inc_edges) > 0)
                is_out = (len(out_edges) > 0)

                in_mask[idx_cp] = is_in
                out_mask[idx_cp] = is_out

                if is_in:
                    in_cps.append(cp)
                    in_from[cp] = inc_edges
                if is_out:
                    out_cps.append(cp)
                    out_to[cp] = out_edges

        # --------- assemble JunctionNX ----------
        J = JunctionNX(
            jid=int(rid_i),
            itype=itype,
            n_legs=int(n_legs),
            centroid=(rx, ry),
            seed_node_id=int(rid_i),
            node_ids=cp_ids_ord,
            edge_ids=[-1] * int(n_legs),
            area=float(area),
            min_angle_deg=float(min_angle_deg),
            max_angle_deg=float(max_angle_deg),
            radius_m=float(radius_m),
            directed_ok=True,
            lanes=lanes_blob,
            cp_adj=adj
        )

        # NEW: kind tag (don’t mix with itype)
        J.junction_type = junction_type   # "junction" | "deadend"

        # Handle in/out points of deadend nodes
        if J.junction_type == "deadend":
            cp2gids = _cp2gids_from_C(G_ord)
            # keep only those actually present in this G_ord cp set (safety)
            J.in_points  = {cp: cp2gids.get(cp, []) for cp in in_cps}
            J.out_points = {cp: cp2gids.get(cp, []) for cp in out_cps}

        # existing attachments
        J.in_cps = in_cps
        J.out_cps = out_cps
        J.in_mask = in_mask
        J.out_mask = out_mask
        J.in_from = in_from
        J.out_to  = out_to

        J_out.append(J)

    return J_out

from collections import defaultdict
import pandas as pd

from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd

def attach_link_lane_group_refs(
    road: Dict[str, Any],
    here_link_df: pd.DataFrame,
    *,
    link_id_col: str = "link_id",
    link_lane_group_refs_key: str = "link_lane_group_refs",  # existing in road
    out_link_to_lane_groups_key: str = "link_to_lane_groups",  # NEW in road
    out_lane_group_to_links_key: str = "lane_group_to_links",  # NEW in road
    lane_group_ref_field: str = "lane_group_ref",
    coerce_lane_group_ref_to: str = "str",  # "str" or "int"
    only_links_in_df: bool = True,
) -> Dict[str, Any]:
    """
    Attach:
      road[out_link_to_lane_groups_key] : {link_id -> [lane_group_ref, ...]}
      road[out_lane_group_to_links_key] : {lane_group_ref -> [link_id, ...]}

    Uses road[link_lane_group_refs_key][link_id][*][lane_group_ref_field].

    Parameters
    ----------
    only_links_in_df:
        If True, considers only link_ids present in here_link_df.
        If False, considers all link_ids in road[link_lane_group_refs_key].

    Returns
    -------
    road (same dict, mutated) for convenience.
    """

    if link_lane_group_refs_key not in road:
        raise KeyError(f"road does not contain '{link_lane_group_refs_key}'")

    llgr = road.get(link_lane_group_refs_key, {}) or {}

    # which link_ids to consider
    if only_links_in_df:
        link_ids = here_link_df[link_id_col].astype("int64").unique().tolist()
    else:
        link_ids = list(llgr.keys())

    def _cast_lg(x):
        if coerce_lane_group_ref_to == "int":
            return int(x)
        return str(x)

    link_to_lane_groups: Dict[int, List[Any]] = {}
    lane_group_to_links: Dict[Any, set] = defaultdict(set)

    for lid in link_ids:
        lid_int = int(lid)
        recs = llgr.get(lid_int, []) or []
        lgs = []
        for r in recs:
            if not isinstance(r, dict):
                continue
            lgref = r.get(lane_group_ref_field, None)
            if lgref is None:
                continue
            lgref = _cast_lg(lgref)
            lgs.append(lgref)
            lane_group_to_links[lgref].add(lid_int)

        # unique lane groups per link (stable order)
        seen = set()
        lgs_uniq = [x for x in lgs if not (x in seen or seen.add(x))]
        link_to_lane_groups[lid_int] = lgs_uniq

    # finalize inverse map with sorted link lists
    lane_group_to_links_out = {lg: sorted(list(lids)) for lg, lids in lane_group_to_links.items()}

    # attach to road
    road[out_link_to_lane_groups_key] = link_to_lane_groups
    road[out_lane_group_to_links_key] = lane_group_to_links_out

    return road


def count_legs_by_link_angle_clusters(G_ccw, ang_ccw, link_id_col, gap_deg=50.0, min_pts=1):
    """
    Count legs by splitting each link_id into one or more angular clusters.
    gap_deg: split threshold between consecutive angles (circular) in degrees.
    min_pts: clusters smaller than this are ignored (optional noise suppression).
    """
    if link_id_col not in G_ccw.columns or len(ang_ccw) == 0:
        return len(G_ccw)

    gap = np.deg2rad(gap_deg)
    legs = 0

    # Angles already correspond to G_ccw order
    link_ids = G_ccw[link_id_col].to_numpy()

    for lid in pd.unique(link_ids):
        idx = np.where(link_ids == lid)[0]
        if idx.size == 0:
            continue
        a = np.sort(ang_ccw[idx])

        if a.size == 1:
            legs += 1
            continue

        # circular gaps
        diffs = np.diff(a)
        wrap  = (a[0] + 2*np.pi) - a[-1]
        gaps  = np.r_[diffs, wrap]

        # number of clusters = number of "big gaps"
        n_clusters = int(np.sum(gaps > gap)) + 1

        # optional: ignore tiny clusters if you want (needs actual split, keep simple for now)
        legs += n_clusters

    return int(legs)

# -----------------------------------------------------------------------------
# Geometry helpers (de-duplicated)
# -----------------------------------------------------------------------------
def _poly_area(P: np.ndarray) -> float:
    x, y = P[:, 0], P[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def _poly_angles_ccw(P: np.ndarray) -> np.ndarray:
    """Interior angles of a CCW-ordered simple polygon."""
    n = len(P)
    ang = np.empty(n)
    for i in range(n):
        p_prev = P[(i - 1) % n]
        p = P[i]
        p_next = P[(i + 1) % n]
        v1 = p_prev - p
        v2 = p_next - p
        v1 /= (np.linalg.norm(v1) + 1e-12)
        v2 /= (np.linalg.norm(v2) + 1e-12)
        ang[i] = np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1, 1)))
    return ang

def _convex_hull_ccw(points: np.ndarray) -> List[int]:
    """Andrew’s monotone chain; return indices of hull vertices in CCW order."""
    pts = np.asarray(points, dtype=float)
    idx = np.lexsort((pts[:, 1], pts[:, 0]))  # sort by x then y
    P = pts[idx]
    I = idx.tolist()

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, lower_i = [], []
    for p, i in zip(P, I):
        while len(lower) >= 2 and cross(pts[lower_i[-2]], pts[lower_i[-1]], pts[i]) <= 0:
            lower.pop()
            lower_i.pop()
        lower.append(p)
        lower_i.append(i)

    upper, upper_i = [], []
    for p, i in zip(P[::-1], I[::-1]):
        while len(upper) >= 2 and cross(pts[upper_i[-2]], pts[upper_i[-1]], pts[i]) <= 0:
            upper.pop()
            upper_i.pop()
        upper.append(p)
        upper_i.append(i)

    # CCW without duplicate endpoints
    return lower_i[:-1] + upper_i[:-1]

# -----------------------------------------------------------------------------
# Lane endpoint utilities
# -----------------------------------------------------------------------------
def _lane_pairs_from_endpoint_map(lane_endpoint_map: pd.DataFrame) -> Set[Tuple[int, int]]:
    """Undirected lane connections purely from endpoints (no direction filtering)."""
    piv = (
        lane_endpoint_map[["GroupID", "LaneNo", "Endpoint", "node_id"]]
        .pivot_table(index=["GroupID", "LaneNo"], columns="Endpoint", values="node_id", aggfunc="first")
    )
    piv = piv[(~piv["start"].isna()) & (~piv["end"].isna())]
    pairs = [(int(a), int(b)) for a, b in zip(piv["start"].astype(int), piv["end"].astype(int))]
    return set(tuple(sorted(p)) for p in pairs)

def _lane_boundary_map_from_Sxy(Sxy: Dict[str, Any]) -> Dict[Tuple[str,int], Tuple[Optional[int], Optional[int]]]:
    """
    Build (GroupID, LaneNo) -> (left_bno, right_bno) from Sxy['lanes'].
    Accepts both HERE-style keys: lane_group_ref / lane_number / lane_index_within_group.
    """
    out: Dict[Tuple[str,int], Tuple[Optional[int], Optional[int]]] = {}
    if not Sxy or "lanes" not in Sxy:
        return out
    for rec in Sxy.get("lanes", []):
        gid = str(rec.get("lane_group_ref"))
        if gid is None or gid == "None":
            continue
        # lane number may come as 'lane_number' (preferred) or 'lane_index_within_group'
        ln = rec.get("lane_number")
        if ln is None:
            ln = rec.get("lane_index_within_group")
        if ln is None:
            continue
        ln = int(ln)
        lbno = rec.get("left_lane_boundary_number")
        rbno = rec.get("right_lane_boundary_number")
        out[(gid, ln)] = (
            int(lbno) if lbno is not None else None,
            int(rbno) if rbno is not None else None,
        )
    return out


# -----------------------------------------------------------------------------
# Road nodes → ENU DataFrame (moved out of main)
# -----------------------------------------------------------------------------
def road_nodes_dict_to_df_enu(
    nodes_obj: dict,
    origin: dict,                        # keys: lat0, lon0, h0  (from build_lane_network_enu)
    id_key_candidates=("node_id", "id", "node_id"),
    lat_key="lat",
    lon_key="lon",
    alt_key_candidates=("h", "alt", "alt_m", "alt_cm"),
):
    """
    Accepts:
      - nodes_obj: dict like { 'nodes': {id: {...}}, ... } or directly {id: {...} }
      - origin: {'lat0': .., 'lon0': .., 'h0': ..}
    Returns:
      DataFrame with columns ['node_id','X','Y'] (N=int, X=float, Y=float)
    """
    from libs.geometry_helpers import geodetic_to_enu  # local import to avoid circulars

    # 1) get the mapping id -> record
    if isinstance(nodes_obj, dict) and "nodes" in nodes_obj and isinstance(nodes_obj["nodes"], dict):
        items = nodes_obj["nodes"].items()
    elif isinstance(nodes_obj, dict) and all(isinstance(v, dict) for v in nodes_obj.values()):
        items = nodes_obj.items()
    else:
        raise ValueError("nodes_obj must be a dict mapping id -> {lat, lon, ...} or contain a 'nodes' dict.")

    lat0, lon0, h0 = float(origin["lat0"]), float(origin["lon0"]), float(origin.get("h0", 0.0))

    rows = []
    for k, rec in items:
        # id
        nid = None
        for kk in id_key_candidates:
            if kk in rec:
                nid = rec[kk]
                break
        if nid is None:
            nid = k
        try:
            nid = int(nid)
        except (ValueError, TypeError):
            continue  # skip pseudo-nodes with non-integer IDs

        # lat/lon
        if lat_key not in rec or lon_key not in rec:
            continue
        lat = float(rec[lat_key])
        lon = float(rec[lon_key])

        # altitude (optional)
        alt = 0.0
        for ak in alt_key_candidates:
            if ak in rec:
                v = rec[ak]
                alt = float(v) / 100.0 if ak.endswith("_cm") else float(v)
                break

        # ENU
        x, y, _z = geodetic_to_enu(lat, lon, alt, lat0, lon0, h0)
        rows.append({"node_id": nid, "X": float(x), "Y": float(y)})

    if not rows:
        print("[WARN] road_nodes_dict_to_df_enu: no nodes with lat/lon found; returning empty DataFrame.")
        return pd.DataFrame(columns=["node_id", "X", "Y"])

    df = pd.DataFrame(rows, columns=["node_id", "X", "Y"])
    df["node_id"] = df["node_id"].astype(int)
    df["X"] = df["X"].astype(float)
    df["Y"] = df["Y"].astype(float)
    return df

# -----------------------------------------------------------------------------
# 4-leg finder (geometry-only): convex hull of 8-closest points within R_node
# -----------------------------------------------------------------------------
from typing import Any

def find_X4_by_hull(
    RoadNodes_df: pd.DataFrame,        # ['node_id','X','Y'] seeds (road/intersection center nodes)
    Nodes: pd.DataFrame,               # lane connection points ['node_id','X','Y'] in ENU
    Edges: pd.DataFrame,               # <- NEW (same table you use elsewhere)
    lane_endpoint_map: pd.DataFrame,   # <- NEW (not strictly needed here, but consistent)
    *,
    Sxy: Optional[Dict[str, Any]] = None,  # <- NEW (to fetch boundary numbers)
    R_node: float = 28.0,
    k_closest: int = 8,
    angle_min_deg: float = 100.0,
    angle_max_deg: float = 170.0,
    min_area: float = 15.0,
    max_area_factor: float = 3.5
) -> List[JunctionNX]:

    # --- precompute lookups ---
    lane_ids = Nodes["node_id"].to_numpy(int)
    lane_xy  = Nodes[["X", "Y"]].to_numpy(float)
    # (GroupID, LaneNo) -> (lbno, rbno)
    bmap = _lane_boundary_map_from_Sxy(Sxy or {})

    # map (undirected) endpoint pair -> list of edge ids
    edges_multi: Dict[frozenset, List[int]] = {}
    if len(Edges):
        for f, t, e in Edges[["FromN", "ToN", "E"]].itertuples(index=False):
            edges_multi.setdefault(frozenset((int(f), int(t))), []).append(int(e))
        Eidx = Edges.set_index("E", drop=False)
    else:
        Eidx = Edges

    out: List[JunctionNX] = []
    jid = 1
    circle_max_area = math.pi * (R_node**2)

    for _, seed in RoadNodes_df.iterrows():
        sx, sy, sid = float(seed["X"]), float(seed["Y"]), int(seed["node_id"])

        # take all points in circle, then keep the 8 closest
        d = np.hypot(lane_xy[:, 0] - sx, lane_xy[:, 1] - sy)
        idx_in = np.where(d <= R_node)[0]
        if idx_in.size < k_closest:
            continue

        idx_sorted = idx_in[np.argsort(d[idx_in])]
        idx8 = idx_sorted[:k_closest]
        pts8 = lane_xy[idx8]
        ids8 = lane_ids[idx8].tolist()

        hull_idx = _convex_hull_ccw(pts8)
        if len(hull_idx) != 8:
            continue

        cyc_ids = [ids8[i] for i in hull_idx]
        P = Nodes.set_index("node_id").loc[cyc_ids][["X", "Y"]].to_numpy(float)

        A = _poly_area(P)
        if A < min_area or A > max_area_factor * circle_max_area:
            continue
        ang = _poly_angles_ccw(P)
        if ang.min() < angle_min_deg or ang.max() > angle_max_deg:
            continue

        cx, cy = P.mean(axis=0)
        r_est = float(np.mean(np.hypot(P[:, 0] - cx, P[:, 1] - cy)))

        # ---- perimeter edge ids (diagonal-aware, like T3) ----
        perim_eids: List[int] = []
        lanes_payload: List[Dict[str, Any]] = []
        Np = len(cyc_ids)

        for i, (a, b) in enumerate(zip(cyc_ids, cyc_ids[1:] + cyc_ids[:1])):
            key = frozenset((int(a), int(b)))
            ids = edges_multi.get(key, [])
            picked = ids[0] if ids else -1

            # fallback: allow short diagonals (±2, ±3 hops)
            if picked < 0:
                for hop in (2, 3):
                    c = int(cyc_ids[(i + hop) % Np])     # forward diagonal
                    ids_ac = edges_multi.get(frozenset((int(a), c)), [])
                    if ids_ac:
                        picked = ids_ac[0]; break
                    dN = int(cyc_ids[(i - hop) % Np])     # backward diagonal
                    ids_db = edges_multi.get(frozenset((int(dN), int(b))), [])
                    if ids_db:
                        picked = ids_db[0]; break

            perim_eids.append(picked)

            # lane record for export (only when we have a valid edge)
            if picked >= 0 and picked in Eidx.index:
                r = Eidx.loc[picked]
                lanes_payload.append(_lane_payload_from_edge_row(r, bmap))

        out.append(
            JunctionNX(
                jid=jid,
                itype="X4-classic",
                n_legs=4,
                centroid=(float(cx), float(cy)),
                seed_node_id=sid,
                node_ids=cyc_ids,
                edge_ids=perim_eids,
                area=float(A),
                min_angle_deg=float(ang.min()),
                max_angle_deg=float(ang.max()),
                radius_m=r_est,
                directed_ok=False,
                lanes=lanes_payload,   # <- now attached with boundary numbers
            )
        )
        jid += 1

    return out


# -----------------------------------------------------------------------------
# 3-leg finder (seeded by road nodes), with optional exclusion by prior X4
# -----------------------------------------------------------------------------
def find_T3_seeded_by_road_nodes(
    RoadNodes: pd.DataFrame,
    Nodes: pd.DataFrame,
    Edges: pd.DataFrame,
    lane_endpoint_map: pd.DataFrame,
    *,
    Sxy: Optional[Dict[str, Any]] = None,   # <— NEW
    R_node: float = 26.0,
    top_m: int = 10,
    max_combos: int = 600,
    min_area: float = 12.0,
    max_area_factor: float = 3.0,
    min_angle: float = 38.0,
    max_angle: float = 172.0,
    exclude_intersections: Optional[Iterable[JunctionNX]] = None,
    exclude_center_tol: float = 15.0,
) -> List[JunctionNX]:

    # ------------------------------------------------------------------
    # lookups
    # ------------------------------------------------------------------
    LXY = Nodes[["X", "Y"]].to_numpy(float)
    LID = Nodes["node_id"].to_numpy(int)
    N_to_xy = Nodes.set_index("node_id")[["X", "Y"]]
    pair_set = _lane_pairs_from_endpoint_map(lane_endpoint_map)
    bmap = _lane_boundary_map_from_Sxy(Sxy or {})   # <— NEW

    # --- multi-edge index for all (FromN, ToN) pairs (undirected) ---
    edges_multi: Dict[frozenset, List[int]] = {}
    for f, t, e in Edges[["FromN", "ToN", "E"]].itertuples(index=False):
        key = frozenset((int(f), int(t)))
        edges_multi.setdefault(key, []).append(int(e))

    # --- exclusion vs. X4 ---
    x4_sets: List[Tuple[Set[int], Tuple[float, float]]] = []
    if exclude_intersections:
        for j in exclude_intersections:
            x4_sets.append((set(j.node_ids), j.centroid))

    def _is_excluded_t3(cyc_nodes: List[int], center: Tuple[float, float]) -> bool:
        S = set(cyc_nodes)
        cx, cy = center
        for Sx4, (x, y) in x4_sets:
            if S.issubset(Sx4) and math.hypot(cx - x, cy - y) <= exclude_center_tol:
                return True
        return False

    out: List[JunctionNX] = []
    seen: Set[frozenset] = set()
    jid = 1

    # ------------------------------------------------------------------
    # main loop over road nodes
    # ------------------------------------------------------------------
    for _, rn in RoadNodes.iterrows():
        cx, cy = float(rn["X"]), float(rn["Y"])
        rid = int(rn["node_id"])

        # endpoints within radius, sorted by distance
        d = np.hypot(LXY[:, 0] - cx, LXY[:, 1] - cy)
        idx = np.where(d <= R_node)[0]
        if idx.size < 6:
            continue
        order = np.argsort(d[idx])
        cand_idx = idx[order][:max(6, min(top_m, idx.size))]
        cand_nodes = LID[cand_idx].tolist()

        # enumerate candidate 6-sets
        candidates = [tuple(cand_nodes[:6])]
        if len(cand_nodes) > 6:
            ccount = 0
            for comb in combinations(cand_nodes, 6):
                if comb != tuple(cand_nodes[:6]):
                    candidates.append(comb)
                ccount += 1
                if ccount >= max_combos:
                    break

        best = None
        best_score = -1e9

        for six in candidates:
            S6 = set(six)
            pairs_in = [(a, b) for (a, b) in pair_set if (a in S6) and (b in S6)]
            if len(pairs_in) < 6:
                continue

            G = nx.Graph()
            G.add_nodes_from(S6)
            G.add_edges_from(pairs_in)

            # require a 6-cycle
            cycles = [c for c in nx.cycle_basis(G) if len(c) == 6]
            if not cycles:
                continue
            cyc = cycles[0]

            # keep adjacency order, only reverse to make CCW
            P = N_to_xy.loc[cyc].to_numpy(float)
            signed_area = 0.5 * (
                np.dot(P[:, 0], np.roll(P[:, 1], -1))
                - np.dot(P[:, 1], np.roll(P[:, 0], -1))
            )
            if signed_area < 0:  # clockwise → reverse
                cyc = cyc[::-1]
                P = P[::-1, :]

            A = _poly_area(P)
            if A < min_area or A > max_area_factor * math.pi * (R_node**2):
                continue
            angs = _poly_angles_ccw(P)
            if angs.min() < min_angle or angs.max() > max_angle:
                continue

            center = tuple(P.mean(axis=0))
            if _is_excluded_t3(cyc, center):
                continue

            dist_to_seed = math.hypot(center[0] - cx, center[1] - cy)
            score = (
                (A / (math.pi * R_node * R_node))
                + 0.4 * (min(angs) - min_angle) / 30.0
                - 0.1 * dist_to_seed / R_node
            )

            if score > best_score:
                best = (cyc, P, A, angs, center)
                best_score = score

        if best is None:
            continue

        cyc, P, A, angs, center = best
        key = frozenset(cyc)
        if key in seen:
            continue
        seen.add(key)

        # ------------------------------------------------------------------
        # NEW: diagonal-aware perimeter edge mapping
        # ------------------------------------------------------------------
        perim_eids: List[int] = []
        Np = len(cyc)
        for i, (a, b) in enumerate(zip(cyc, cyc[1:] + cyc[:1])):
            key = frozenset((int(a), int(b)))
            ids = edges_multi.get(key, [])
            if ids:
                perim_eids.append(ids[0])
                continue

            # fallback: look for diagonals up to ±3 hops
            picked = -1
            for hop in (2, 3):
                c = int(cyc[(i + hop) % Np])     # forward diagonal
                ids_ac = edges_multi.get(frozenset((int(a), c)), [])
                if ids_ac:
                    picked = ids_ac[0]; break
                d = int(cyc[(i - hop) % Np])     # backward diagonal
                ids_db = edges_multi.get(frozenset((int(d), int(b))), [])
                if ids_db:
                    picked = ids_db[0]; break
            perim_eids.append(picked)

        # Assemble exported lane records for this junction (perimeter edges)
        if len(Edges):
            Eidx = Edges.set_index("E", drop=False)  # keep 'E' as a column
        else:
            Eidx = Edges  # empty
        lanes_payload: List[Dict[str, Any]] = []
        for e in perim_eids:
            if e is None or e < 0 or not len(Edges) or e not in Eidx.index:
                continue
            r = Eidx.loc[e]
            lanes_payload.append(_lane_payload_from_edge_row(r, bmap))
        # ------------------------------------------------------------------
        # output junction
        # ------------------------------------------------------------------
        out.append(
            JunctionNX(
                jid=jid,
                itype="T3-classic",
                n_legs=3,
                centroid=(float(center[0]), float(center[1])),
                seed_node_id=rid,
                node_ids=cyc,
                edge_ids=perim_eids,
                area=float(A),
                min_angle_deg=float(angs.min()),
                max_angle_deg=float(angs.max()),
                radius_m=R_node,
                directed_ok=False,
                lanes=lanes_payload,   # <— NEW
            )
        )
        jid += 1

    return out


# -----------------------------------------------------------------------------
# Plot utilities (generic for any n-legs)
# -----------------------------------------------------------------------------
def plot_all_connection_points(
    ax,
    Nodes: pd.DataFrame,
    Edges: Optional[pd.DataFrame] = None,
    *,
    color_by_degree: bool = True,
    show_ids: bool = False,
    ms: float = 4.0,
    alpha: float = 0.9,
):
    """
    Overlay all lane connection points (from `Nodes`) on the current axes.

    - If `color_by_degree` and `Edges` provided:
        deg==1  → light blue (dead-ends / tile edges)
        deg==2  → black     (typical interior connection points)
        deg>=3  → orange    (junction candidates)
    """
    X = Nodes["X"].to_numpy(float)
    Y = Nodes["Y"].to_numpy(float)

    if color_by_degree and Edges is not None and len(Edges):
        deg = pd.concat([Edges["FromN"], Edges["ToN"]]).value_counts()
        d = Nodes["node_id"].map(deg).fillna(0).astype(int).to_numpy()

        mask1 = d == 1
        mask2 = d == 2
        mask3 = d >= 3

        ax.plot(X[mask2], Y[mask2], "o", ms=ms, mfc="k", mec="k", alpha=alpha, label="deg=2")
        ax.plot(X[mask1], Y[mask1], "o", ms=ms, mfc="#69b3f2", mec="#69b3f2", alpha=alpha, label="deg=1")
        ax.plot(X[mask3], Y[mask3], "o", ms=ms + 0.5, mfc="#ff8c00", mec="#ff8c00", alpha=alpha, label="deg≥3")
    else:
        ax.plot(X, Y, "o", ms=ms, mfc="k", mec="k", alpha=alpha, label="all endpoints")

    if show_ids:
        for n, x, y in Nodes[["node_id", "X", "Y"]].itertuples(index=False):
            ax.text(
                x,
                y+0.5,
                str(int(n)),
                fontsize=7,
                color="dimgray",
                ha="center",
                va="center",
                bbox=dict(fc="white", ec="none", alpha=0.6),
            )

    ax.set_aspect("equal", "box")

import numpy as np
import pandas as pd
from matplotlib.patches import Circle

def plot_junctions_with_angles(
    ax,
    J,
    Nodes: pd.DataFrame,
    *,
    Sxy=None,
    lane_lw: float = 0.9,
    lane_alpha: float = 0.25,
    lane_color: str = "gray",
    hull_color: str = "crimson",
    hull_lw: float = 2.0,
    show_circle: bool = True,
    circle_color: str = "royalblue",
    circle_ls: str = "--",
    circle_lw: float = 1.2,
    show_angles: bool = True,
    angle_fontsize: int = 8,
    angle_fmt: str = "{:.0f}°",
    show_type: bool = True,
    type_fontsize: int = 9,
    show_node_ids: bool = False,
    node_id_fontsize: int = 7,
    show_missing_edges: bool = True,     # NEW: draw dashed edge where eid==-1
    missing_ls: str = ":",               # NEW
    missing_alpha: float = 0.8,          # NEW
    annotate_lanes_per_leg: bool = True, # NEW: write (#lanes) near each vertex if present
    lanes_fontsize: int = 8              # NEW
):
    """
    Robust plotting for perfect/imperfect junctions.

    Supports:
      - dict or object junction records
      - j.hull_node_ids (preferred) or j.node_ids (fallback)
      - j.hull_edge_ids with -1 for missing segments
      - j.legs[k]['lanes_count'] to annotate lanes per leg (optional)
      - j.centroid (x,y) and j.radius_m for circle
    """
    def _get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _order_ccw(P, center=None):
        """Return indices that sort points CCW around center (or mean)."""
        if P.shape[0] <= 2:
            return list(range(P.shape[0]))
        c = np.mean(P, axis=0) if center is None else np.asarray(center, float)
        ang = _angles_ccw(c, P)
        return list(np.argsort(ang))

    XY = Nodes.set_index("node_id")[["X", "Y"]]

    # --- lane overlay for context ---
    if Sxy and isinstance(Sxy, dict) and "lanes" in Sxy:
        for rec in Sxy.get("lanes", []):
            x = np.asarray(rec.get("x", []), float).ravel()
            y = np.asarray(rec.get("y", []), float).ravel()
            if x.size >= 2:
                ax.plot(x, y, lw=lane_lw, alpha=lane_alpha, color=lane_color, zorder=0)

    for j in J:
        # Prefer hull reps if present (general finder), else raw node_ids
        hull_ids = _get(j, "hull_node_ids")
        node_ids = hull_ids if hull_ids else _get(j, "node_ids", [])
        if not node_ids:
            continue

        # Positions (safe reindex)
        P = XY.loc[node_ids].to_numpy(float)

        # Centroid & radius (if present)
        centroid = _get(j, "centroid")
        radius_m = float(_get(j, "radius_m", 0.0) or 0.0)
        if centroid is None:
            cx, cy = P.mean(axis=0)
            centroid = (float(cx), float(cy))

        # CCW order (works for 2,3,4+ points)
        ord_idx = _order_ccw(P, center=centroid)
        P = P[ord_idx]
        ids_ord = [node_ids[i] for i in ord_idx]

        # Draw perimeter (polygon for >=3, simple line for 2)
        if len(P) >= 3:
            Pc = np.vstack([P, P[0]])
            ax.plot(Pc[:, 0], Pc[:, 1], color=hull_color, lw=hull_lw, zorder=2)
            # Missing edge visualization if we have hull_edge_ids
            eids = _get(j, "hull_edge_ids")
            if show_missing_edges and eids and len(eids) == len(P):
                for k in range(len(P)):
                    if int(eids[k]) == -1:
                        a = P[k]
                        b = P[(k+1) % len(P)]
                        ax.plot([a[0], b[0]], [a[1], b[1]],
                                color=hull_color, lw=max(1.0, hull_lw-0.6),
                                ls=missing_ls, alpha=missing_alpha, zorder=2.1)
        elif len(P) == 2:
            # Just draw the segment
            ax.plot(P[:, 0], P[:, 1], color=hull_color, lw=hull_lw, zorder=2)

        # Circle (search radius estimate)
        if show_circle and radius_m > 0:
            ax.add_patch(Circle(centroid, radius_m, fill=False, ec=circle_color,
                                ls=circle_ls, lw=circle_lw, zorder=1))

        # Angle annotations (for >=3 points)
        if show_angles and len(P) >= 3:
            ang = _poly_angles_ccw(P)
            for (xv, yv), a in zip(P, ang):
                ax.text(xv, yv, angle_fmt.format(a), fontsize=angle_fontsize,
                        ha="center", va="bottom", color=hull_color, zorder=3)

        # Type label
        itype = _get(j, "itype")
        if show_type and itype:
            ax.text(centroid[0], centroid[1], f"{itype}",
                    fontsize=type_fontsize, color=hull_color,
                    ha="center", va="center", zorder=3)

        # Node ids
        if show_node_ids:
            for nid, (xv, yv) in zip(ids_ord, P):
                ax.text(xv, yv, str(nid), fontsize=node_id_fontsize,
                        ha="left", va="bottom", zorder=3)

        # Lanes per leg (if legs[] present and aligned with hull order)
        legs = _get(j, "legs")
        if annotate_lanes_per_leg and isinstance(legs, (list, tuple)) and len(legs) == len(P):
            for (xv, yv), leg in zip(P, legs):
                k = leg.get("lanes_count") if isinstance(leg, dict) else getattr(leg, "lanes_count", None)
                if k is not None:
                    ax.text(xv, yv, f"({k})", fontsize=lanes_fontsize,
                            ha="center", va="top", color=hull_color, zorder=3)

    ax.set_aspect("equal", adjustable="box")


def _lane_payload_from_edge_row(r: pd.Series,
                                bmap: Dict[Tuple[str,int], Tuple[Optional[int], Optional[int]]]) -> Dict[str, object]:
    gid = str(r["GroupID"])
    ln  = int(r["LaneNo"])

    # Prefer numbers already in Edges, else fall back to Sxy map
    lb = r["left_lane_boundary_number"] if "left_lane_boundary_number" in r else np.nan
    rb = r["right_lane_boundary_number"] if "right_lane_boundary_number" in r else np.nan

    if (pd.isna(lb) or lb is None) or (pd.isna(rb) or rb is None):
        lb2, rb2 = bmap.get((gid, ln), (None, None))
        if (pd.isna(lb) or lb is None) and lb2 is not None: lb = lb2
        if (pd.isna(rb) or rb is None) and rb2 is not None: rb = rb2

    return {
        "E":        int(r["E"]),
        "FromN":    int(r["FromN"]),
        "ToN":      int(r["ToN"]),
        "GroupID":  gid,
        "LaneNo":   ln,
        "Dir":      str(r["Dir"]),
        "Weight":   float(r["Weight"]),
        "left_lane_boundary_number":  None if (pd.isna(lb) or lb is None) else int(lb),
        "right_lane_boundary_number": None if (pd.isna(rb) or rb is None) else int(rb),
    }

import math, numpy as np, pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Iterable

# --- small helpers (pure numpy) ---

def _angles_ccw(pts: np.ndarray, center: Tuple[float,float]) -> np.ndarray:
    vx, vy = pts[:,0]-center[0], pts[:,1]-center[1]
    ang = np.degrees(np.arctan2(vy, vx)) % 360.0
    return ang


import math, numpy as np, pandas as pd
from typing import List, Dict, Any, Tuple, Optional


import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# ---- utilities --------------------------------------------------------------

def _to_angle(vx, vy):
    """Return heading angle in radians in [0, 2π)."""
    a = math.atan2(vy, vx)
    return a if a >= 0 else a + 2*math.pi

def _merge_angles(angles_deg, tol_merge_deg=15.0, bidir_merge_deg=15.0):
    """
    Cluster angles into legs.
    - First, merge angles within tol_merge_deg (same-direction lanes).
    - Then, merge clusters that are ~180° apart into the same leg (two-way).
    Returns list of leg angles (deg) and labels (len=angles).
    """
    if len(angles_deg) == 0:
        return [], np.array([], dtype=int)

    ang = np.sort(np.mod(angles_deg, 360.0))
    # 1) same-direction merging (1D clustering on circle)
    clusters = [[ang[0]]]
    for a in ang[1:]:
        if min(abs(a - np.mean(clusters[-1])), 360-abs(a - np.mean(clusters[-1]))) <= tol_merge_deg:
            clusters[-1].append(a)
        else:
            clusters.append([a])
    centers = np.array([np.mean(c) % 360.0 for c in clusters])

    # 2) bidirectional merge (≈ 180° apart)
    used = np.zeros(len(centers), dtype=bool)
    leg_centers = []
    for i, ci in enumerate(centers):
        if used[i]: 
            continue
        used[i] = True
        # find partner near 180°
        diffs = np.mod(centers - (ci + 180.0), 360.0)
        j = np.argmin(np.minimum(diffs, 360.0 - diffs))
        if (not used[j]) and min(abs(((centers[j]-ci+180)%360)-180), 
                                 abs(((ci-centers[j]+180)%360)-180)) <= bidir_merge_deg:
            used[j] = True
            leg_centers.append(ci)  # arbitrary representative
        else:
            leg_centers.append(ci)
    # create labels by nearest leg center or its opposite
    labels = np.zeros(len(angles_deg), dtype=int)
    leg_centers = np.array(leg_centers) % 360.0
    for k, a in enumerate(angles_deg):
        # distance to each leg or its opposite
        d = []
        for c in leg_centers:
            d.append(min(abs(a-c), 360-abs(a-c), abs((a-(c+180))%360), 360-abs((a-(c+180))%360)))
        labels[k] = int(np.argmin(d))
    return leg_centers.tolist(), labels

# ---- main: find seed nodes --------------------------------------------------

def find_seed_nodes(Nodes, conn_pts, *, origin=None, rc=12.0,  # meters
                    min_legs=3, theta_min_deg=30.0,
                    tol_merge_deg=15.0, bidir_merge_deg=15.0):
    """
    Parameters
    ----------
    Nodes     : DataFrame with columns ['node_id', 'x','y'] in ENU
                (or 'lat','lon' if you choose to pre-convert outside)
    conn_pts  : DataFrame with columns ['x','y'] (ENU) and optional ['lane_id','group_id','dir']
    rc        : search radius around each node (m)
    min_legs  : minimum number of angular legs to accept as a seed
    theta_min_deg : minimum angular separation between any two legs
    Returns
    -------
    seeds_df : DataFrame with one row per seed node:
               ['node_id','x','y','n_conn','n_legs','leg_angles_deg']
    also provides a dict node_id -> indices of conn_pts used
    """
    # Ensure ENU in Nodes
    if not {'X','Y'}.issubset(Nodes.columns):
        raise ValueError("Nodes must contain ENU columns 'X','Y'. Convert before calling.")

    node_xy = Nodes[['node_id','X','Y']].to_numpy()
    P = conn_pts[['x','y']].to_numpy()

    rows = []
    idx_map = {}  # node_id -> indices of conn_pts used

    # grid index for speed (simple uniform binning)
    if len(P) == 0:
        return pd.DataFrame(columns=['node_id','x','y','n_conn','n_legs','leg_angles_deg']), {}

    cell = max(rc, 1.0)
    minx, miny = P.min(axis=0)
    maxx, maxy = P.max(axis=0)
    nx = max(1, int((maxx-minx)/cell)+1)
    ny = max(1, int((maxy-miny)/cell)+1)
    buckets = {}
    for i,(px,py) in enumerate(P):
        ix = int((px-minx)/cell); iy = int((py-miny)/cell)
        buckets.setdefault((ix,iy), []).append(i)

    def candidates_around(x,y):
        ix = int((x-minx)/cell); iy = int((y-miny)/cell)
        cand = []
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                cand.extend(buckets.get((ix+dx,iy+dy), []))
        return cand

    for nid, nx_, ny_ in node_xy:
        cand_idx = candidates_around(nx_, ny_)
        if not cand_idx:
            continue
        d2 = (P[cand_idx,0]-nx_)**2 + (P[cand_idx,1]-ny_)**2
        in_rc = [cand_idx[i] for i in np.where(d2 <= rc*rc)[0]]
        if len(in_rc) < 3:
            continue

        # angles from node to connection points
        A = []
        for j in in_rc:
            vx, vy = P[j,0]-nx_, P[j,1]-ny_
            if vx == 0 and vy == 0: 
                continue
            A.append(math.degrees(_to_angle(vx, vy)))
        if len(A) < 3:
            continue

        leg_angles, labels = _merge_angles(A, tol_merge_deg, bidir_merge_deg)

        # reject if legs too close
        ok_sep = True
        if len(leg_angles) >= 2:
            la = np.sort(np.array(leg_angles))
            gaps = np.diff(np.r_[la, la[0]+360.0])
            if np.any(gaps < theta_min_deg):
                ok_sep = False

        if len(leg_angles) >= min_legs and ok_sep:
            rows.append({
                'node_id': int(nid), 'x': float(nx_), 'y': float(ny_),
                'n_conn': int(len(in_rc)),
                'n_legs': int(len(leg_angles)),
                'leg_angles_deg': [float(a) for a in leg_angles],
            })
            idx_map[int(nid)] = in_rc

    seeds_df = pd.DataFrame(rows).sort_values('n_legs', ascending=False).reset_index(drop=True)
    return seeds_df, idx_map

# ---- quick plotter ----------------------------------------------------------

def plot_seed_nodes(ax, Nodes, conn_pts, seeds_df, idx_map, 
                    node_color='k', cp_color='orange', seed_edge='r'):
    """
    Plot all connection points and highlight seed nodes with red-edged circles.
    """
    # connection points
    ax.plot(conn_pts['x'], conn_pts['y'], '.', ms=6, color=cp_color, alpha=0.9)
    # all nodes (light)
    ax.plot(Nodes['X'], Nodes['Y'], '.', ms=3, color='0.7', alpha=0.6)
    # seeds
    for _, r in seeds_df.iterrows():
        ax.plot(r['x'], r['y'], 'o', ms=8, mfc='white', mec=seed_edge, mew=1.8)
        # optional: draw small spokes for leg directions
        for a in r['leg_angles_deg']:
            L = 8.0
            rad = math.radians(a)
            ax.plot([r['x'], r['x']+L*math.cos(rad)],
                    [r['y'], r['y']+L*math.sin(rad)],
                    '--', lw=0.8, color=seed_edge, alpha=0.7)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)

import pandas as pd
import matplotlib.pyplot as plt

# -- 1) normalize to a common schema: ['node_id','x','y'] --------------------
def normalize_road_nodes(RoadNodes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts the output of road_nodes_dict_to_df_enu (columns ['node_id','X','Y']).
    Returns DataFrame with ['node_id','x','y'] for the rest of the pipeline.
    """
    if not {'node_id','X','Y'}.issubset(RoadNodes_df.columns):
        raise ValueError("Expected RoadNodes_df with columns ['node_id','X','Y'].")
    nodes = RoadNodes_df.rename(columns={'node_id':'node_id','X':'x','Y':'y'})[['node_id','x','y']].copy()
    nodes['node_id'] = nodes['node_id'].astype(int)
    nodes['x'] = nodes['x'].astype(float)
    nodes['y'] = nodes['y'].astype(float)
    return nodes

# -- 2) simple plotter: plot all road nodes as 'seed' circles ----------------
def plot_road_node_seeds(nodes_df: pd.DataFrame, *, ax=None, title="Road-node seeds"):
    """
    Plots all road nodes (treating them as initial seeds). No filtering yet.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(nodes_df['x'], nodes_df['y'], 'o', mfc='white', mec='red', mew=1.5, ms=6)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    return ax

import pandas as pd
import matplotlib.pyplot as plt

# --- 3) seed selection from LINKS ONLY --------------------------------------
def seeds_from_links(RoadNodes_df, Links_df,
                     *, min_neighbors: int = 3,
                     collapse_parallel: bool = True,
                     ignore_self_loops: bool = True) -> pd.DataFrame:
    """
    A road node is a seed if it connects to at least `min_neighbors` DISTINCT neighbors via links.
    - collapse_parallel=True: multiple links between the same pair (slip lanes) count once.
    - ignore_self_loops=True: links with u==v are ignored.
    Returns: DataFrame ['node_id','x','y','deg','neighbors']
    """
    nodes = normalize_road_nodes(RoadNodes_df)
    L = Links_df.copy()

    if ignore_self_loops:
        L = L[L['u'] != L['v']]

    # undirected adjacency (intersection-ness doesn’t care about one-way direction)
    E = L[['u','v']].copy()
    E2 = E.rename(columns={'u':'v','v':'u'})
    A = pd.concat([E, E2], ignore_index=True)

    if collapse_parallel:
        A = A.drop_duplicates(['u','v'])

    # neighbors per node
    neigh = (A.groupby('u')['v']
               .apply(lambda s: sorted(set(int(x) for x in s)))
               .rename('neighbors'))

    deg = neigh.apply(len).rename('deg')

    seeds = (nodes.set_index('node_id')
                  .join(pd.concat([deg, neigh], axis=1))
                  .dropna(subset=['deg'])
                  .query('deg >= @min_neighbors')
                  .reset_index())

    return seeds[['node_id','x','y','deg','neighbors']].sort_values('deg', ascending=False)

# --- 4) plot -----------------------------------------------------------------
def plot_link_seeds(ax, RoadNodes_df, seeds_df):
    nodes = normalize_road_nodes(RoadNodes_df)
    ax.plot(nodes['x'], nodes['y'], '.', color='0.75', ms=3, alpha=0.7)
    ax.plot(seeds_df['x'], seeds_df['y'], 'o', mfc='white', mec='red', mew=1.8, ms=8)
    for _, r in seeds_df.iterrows():
        ax.text(r['x'], r['y'], str(int(r['deg'])), color='red',
                fontsize=9, ha='center', va='bottom')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

    import numpy as np
import pandas as pd
import math


def _angles_deg(px, py, Q):  # Q: Nx2
    V = Q - np.array([px, py])
    return (np.degrees(np.arctan2(V[:,1], V[:,0])) + 360.0) % 360.0

def _angular_span(angles_deg):
    if len(angles_deg) == 0: return 0.0
    a = np.sort(angles_deg)
    wrap = np.r_[a, a[0] + 360.0]
    gaps = np.diff(wrap)
    # covered span = 360 - largest empty gap
    return 360.0 - float(np.max(gaps))

def filter_seeds_with_lane_coverage(
    seeds_df: pd.DataFrame,
    lane_points_df: pd.DataFrame,
    *,
    rc: float = 15.0,          # search radius (m)
    min_pts: int = 3,          # need at least this many lane points near the seed
    min_span_deg: float = 200, # require points around (not one-sided)
    n_bins: int = 8,           # optional sector check
    min_bins_hit: int = 3
):
    """
    Inputs
    ------
    seeds_df: ['node_id','x','y', ...]  (from seeds_from_links)
    lane_points_df: any table with ENU columns X,Y or x,y (e.g., lane Nodes)

    Returns
    -------
    keep_seeds: filtered seeds_df with columns:
      ['node_id','x','y','deg','neighbors','n_lane_pts','ang_span','bins_hit']
    plus: a boolean mask and diagnostics columns.
    """
    S = seeds_df.copy()
    P = _to_xy(lane_points_df)

    # lightweight uniform grid for radius queries
    cell = max(rc, 1.0)
    minx, miny = P.min(0); maxx, maxy = P.max(0)
    buckets = {}
    for i,(x,y) in enumerate(P):
        ix = int((x-minx)/cell); iy = int((y-miny)/cell)
        buckets.setdefault((ix,iy), []).append(i)

    def neighbors_in_radius(x,y):
        ix = int((x-minx)/cell); iy = int((y-miny)/cell)
        cand = []
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                cand += buckets.get((ix+dx, iy+dy), [])
        if not cand: return np.array([], dtype=int)
        d2 = (P[cand,0]-x)**2 + (P[cand,1]-y)**2
        return np.array([cand[i] for i in np.where(d2 <= rc*rc)[0]], dtype=int)

    n_pts_list, span_list, bins_hit_list = [], [], []
    for _, r in S.iterrows():
        idx = neighbors_in_radius(r['x'], r['y'])
        n_pts = int(len(idx))
        if n_pts:
            ang = _angles_deg(r['x'], r['y'], P[idx])
            span = _angular_span(ang)
            # sector occupancy
            bins = np.floor((ang/360.0) * n_bins).astype(int) % n_bins
            bins_hit = int(len(np.unique(bins)))
        else:
            span, bins_hit = 0.0, 0
        n_pts_list.append(n_pts); span_list.append(span); bins_hit_list.append(bins_hit)

    S['n_lane_pts'] = n_pts_list
    S['ang_span']   = span_list
    S['bins_hit']   = bins_hit_list

    mask = (S['n_lane_pts'] >= min_pts) & ((S['ang_span'] >= min_span_deg) | (S['bins_hit'] >= min_bins_hit))
    return S.loc[mask].reset_index(drop=True), mask, S

import numpy as np
import pandas as pd
import math

def _to_xy(df):
    if {'x','y'}.issubset(df.columns): return df[['x','y']].to_numpy(float)
    if {'X','Y'}.issubset(df.columns): return df[['X','Y']].to_numpy(float)
    raise ValueError("Need columns X/Y (or x/y) for connection points.")

def _angles_deg(cx, cy, Q):  # Q: Nx2 points
    V = Q - np.array([cx, cy])
    return (np.degrees(np.arctan2(V[:,1], V[:,0])) + 360.0) % 360.0

def _angular_span(angles_deg):
    if len(angles_deg) == 0: return 0.0
    a = np.sort(angles_deg)
    wrap = np.r_[a, a[0] + 360.0]
    gaps = np.diff(wrap)
    # span = 360 - largest empty gap (i.e., covered arc)
    return 360.0 - float(np.max(gaps))

def filter_seeds_by_vector_spread(
    seeds_df: pd.DataFrame,
    conn_pts_df: pd.DataFrame,   # lane connection points (orange dots) with X/Y (or x/y)
    *,
    rc: float = 12.0,            # radius around seed (m)
    min_pts: int = 3,            # need at least this many CPs
    min_span_deg: float = 120.0, # require at least this angular spread
    require_two_sides: bool = True,   # enforce points on both sides of a line
    side_margin_deg: float = 10.0,    # >180+margin for “two sides”
    n_bins: int = 8,                 # optional robustness: sector occupancy
    min_bins_hit: int = 2
):
    """
    Keep seeds only if nearby connection-point vectors are not all aligned.
    Conditions:
      1) >= min_pts points within rc
      2) angular span >= min_span_deg
      3) (optional) points occupy both sides of some dividing line:
         span >= 180 + side_margin_deg
      4) (optional) hit at least min_bins_hit angular sectors
    Returns: filtered_seeds, mask, diagnostics_df
    """
    S = seeds_df.copy()
    P = _to_xy(conn_pts_df)

    # simple uniform grid for fast radius queries
    cell = max(rc, 1.0)
    minx, miny = P.min(0); maxx, maxy = P.max(0)
    buckets = {}
    for i,(x,y) in enumerate(P):
        ix = int((x-minx)/cell); iy = int((y-miny)/cell)
        buckets.setdefault((ix,iy), []).append(i)

    def in_radius(x,y):
        ix = int((x-minx)/cell); iy = int((y-miny)/cell)
        cand = []
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                cand += buckets.get((ix+dx, iy+dy), [])
        if not cand: return np.array([], dtype=int)
        d2 = (P[cand,0]-x)**2 + (P[cand,1]-y)**2
        return np.array([cand[i] for i in np.where(d2 <= rc*rc)[0]], dtype=int)

    n_list, span_list, bins_list, two_side_list = [], [], [], []
    for _, r in S.iterrows():
        idx = in_radius(r['x'], r['y'])
        n = int(len(idx))
        if n:
            ang = _angles_deg(r['x'], r['y'], P[idx])
            span = _angular_span(ang)
            bins = np.floor((ang/360.0)*n_bins).astype(int) % n_bins
            bins_hit = len(np.unique(bins))
            two_sides = span >= (180.0 + side_margin_deg) if require_two_sides else True
        else:
            span, bins_hit, two_sides = 0.0, 0, False
        n_list.append(n); span_list.append(span); bins_list.append(bins_hit); two_side_list.append(two_sides)

    S['n_cp'] = n_list
    S['ang_span'] = span_list
    S['bins_hit'] = bins_list
    S['two_sides'] = two_side_list

    mask = (S['n_cp'] >= min_pts) & (S['ang_span'] >= min_span_deg) & S['two_sides'] & (S['bins_hit'] >= min_bins_hit)
    return S.loc[mask].reset_index(drop=True), mask.to_numpy(bool), S

import numpy as np
import pandas as pd
import math

def _to_xy(df):
    if {'x','y'}.issubset(df.columns): return df[['x','y']].to_numpy(float)
    if {'X','Y'}.issubset(df.columns): return df[['X','Y']].to_numpy(float)
    raise ValueError("Need X/Y (or x/y).")

def _bearing(p, q):
    return (math.degrees(math.atan2(q[1]-p[1], q[0]-p[0])) + 360.0) % 360.0

def _angular_span(ang):
    if len(ang) == 0: return 0.0
    a = np.sort(ang)
    wrap = np.r_[a, a[0] + 360.0]
    gaps = np.diff(wrap)
    return 360.0 - float(np.max(gaps))

def _incident_bearings(seed_id, nodes_df, links_df):
    # returns list of bearings (deg) from the seed to each distinct neighbor node
    xys = nodes_df.set_index('node_id')[['X','Y']]
    p = xys.loc[seed_id].to_numpy().ravel()
    neigh = set()
    for _,e in links_df.iterrows():
        u,v = int(e.u), int(e.v)
        if u == seed_id: neigh.add(v)
        elif v == seed_id: neigh.add(u)
    bears = []
    for nb in neigh:
        q = xys.loc[nb].to_numpy().ravel()
        bears.append(_bearing(p, q))
    return np.array(bears, float)

def _build_grid(P, cell):
    minx, miny = P.min(0)
    buckets = {}
    for i,(x,y) in enumerate(P):
        ix = int((x-minx)/cell); iy = int((y-miny)/cell)
        buckets.setdefault((ix,iy), []).append(i)
    return buckets, minx, miny

def _in_radius(P, buckets, minx, miny, x, y, rc, cell):
    ix = int((x-minx)/cell); iy = int((y-miny)/cell)
    cand = []
    for dx in (-1,0,1):
        for dy in (-1,0,1):
            cand += buckets.get((ix+dx, iy+dy), [])
    if not cand: return np.array([], int)
    d2 = (P[cand,0]-x)**2 + (P[cand,1]-y)**2
    return np.array([cand[i] for i in np.where(d2 <= rc*rc)[0]], int)

def _nearest_per_bearing(sx, sy, Pidx, P, bearings, tol):
    """For each bearing (deg), keep index of the nearest point within ±tol deg."""
    if len(Pidx) == 0 or len(bearings) == 0:
        return np.array([], int)
    V = P[Pidx] - np.array([sx, sy])
    ang = (np.degrees(np.arctan2(V[:,1], V[:,0])) + 360.0) % 360.0
    dist = np.hypot(V[:,0], V[:,1])
    keep = []
    for b in bearings:
        d_ang = np.minimum(np.abs(ang - b), 360.0 - np.abs(ang - b))
        mask = d_ang <= tol
        if np.any(mask):
            k = np.argmin(dist[mask])
            keep.append(Pidx[np.where(mask)[0][k]])
    return np.array(sorted(set(keep)), int)

def _nearest_per_sector(sx, sy, Pidx, P, n_bins):
    """Partition 360° into n_bins and keep nearest point per bin."""
    if len(Pidx) == 0:
        return np.array([], int)
    V = P[Pidx] - np.array([sx, sy])
    ang = (np.degrees(np.arctan2(V[:,1], V[:,0])) + 360.0) % 360.0
    dist = np.hypot(V[:,0], V[:,1])
    bins = (np.floor(ang/360.0 * n_bins).astype(int)) % n_bins
    keep = []
    for b in range(n_bins):
        m = np.where(bins == b)[0]
        if len(m):
            keep.append(Pidx[m[np.argmin(dist[m])]])
    return np.array(sorted(set(keep)), int)

def filter_seeds_directional_nearest(
    seeds_df,
    conn_pts_df,            # lane connection points (X/Y or x/y)
    nodes_df,               # road nodes ['node_id','x','y']
    links_df=None,          # road links ['u','v']; optional but recommended
    *,
    rcap=25.0,              # loose cap (m) to get candidate points (pre-thinning)
    rc_local=10.0,          # default local radius if you don't compute adaptive
    use_adaptive_rc=True,   # adapt rc to local link length
    min_pts=2,              # require at least this many selected points
    bearing_tol_deg=25.0,   # cone half-width around each incident bearing
    n_bins=8,               # if no links: sector count
    max_ratio=2.5,          # drop selected points farther than max_ratio * nearest
    span_keep_deg=140.0,    # accept if angular span of selected points ≥ this
    require_two_sides=False # set True if you want ≥ ~180° coverage
):
    P = _to_xy(conn_pts_df)
    cell = max(rc_local, 1.0)
    buckets, minx, miny = _build_grid(P, cell)

    # pre for adaptive radius
    nodes_idx = nodes_df.set_index('node_id')[['X','Y']]
    link_uv = None
    if links_df is not None:
        link_uv = links_df[['u','v']].to_numpy(int)

    def local_rc(seed_id):
        if not use_adaptive_rc or link_uv is None:
            return rc_local
        p = nodes_idx.loc[seed_id].to_numpy().ravel()
        Ls = []
        for u,v in link_uv:
            if u==seed_id or v==seed_id:
                q = nodes_idx.loc[v if u==seed_id else u].to_numpy().ravel()
                Ls.append(np.linalg.norm(q - p))
        if not Ls:
            return rc_local
        return min(rcap, max(rc_local, 0.35*min(Ls)))  # ~1/3 of shortest incident link

    kept_idx = []
    diag = []
    for i, s in seeds_df.reset_index().iterrows():
        sid, sx, sy = int(s['node_id']), float(s['x']), float(s['y'])
        rc = local_rc(sid)

        # broad pickup (loose cap), then directional thinning inside rc
        cand = _in_radius(P, buckets, minx, miny, sx, sy, rcap, cell)
        if len(cand) == 0:
            diag.append((0, 0, rc, 0.0)); continue
        # keep only those within rc for thinning
        near = cand[np.where((P[cand,0]-sx)**2 + (P[cand,1]-sy)**2 <= rc*rc)[0]]

        if links_df is not None:
            bears = _incident_bearings(sid, nodes_df, links_df)
            sel = _nearest_per_bearing(sx, sy, near, P, bears, bearing_tol_deg)
        else:
            sel = _nearest_per_sector(sx, sy, near, P, n_bins)

        if len(sel) < min_pts:
            diag.append((len(sel), 0, rc, 0.0)); continue

        V = P[sel] - np.array([sx, sy])
        d = np.hypot(V[:,0], V[:,1]); a = (np.degrees(np.arctan2(V[:,1], V[:,0])) + 360.0) % 360.0

        # distance-based pruning: keep only points not much farther than nearest
        dmin = d.min()
        ok = sel[np.where(d <= max_ratio * dmin)[0]]
        if len(ok) < min_pts:
            diag.append((len(ok), 0, rc, 0.0)); continue

        a_ok = (np.degrees(np.arctan2(P[ok,1]-sy, P[ok,0]-sx)) + 360.0) % 360.0
        span = _angular_span(a_ok)
        two_sides = span >= 180.0 if require_two_sides else True

        if (span >= span_keep_deg) and two_sides:
            kept_idx.append(int(s['index']))
            diag.append((len(ok), len(ok), rc, span))
        else:
            # also accept if we touched ≥2 distinct incident bearings (if links known)
            if links_df is not None:
                bears = _incident_bearings(sid, nodes_df, links_df)
                hits = 0
                for b in np.unique(bears):
                    d_ang = np.minimum(np.abs(a_ok - b), 360.0 - np.abs(a_ok - b))
                    if np.any(d_ang <= bearing_tol_deg):
                        hits += 1
                if hits >= 2:
                    kept_idx.append(int(s['index']))
            diag.append((len(ok), 0 if links_df is None else hits, rc, span))

    out = seeds_df.loc[kept_idx].copy().reset_index(drop=True)
    return out  # add diag if you want to inspect


import numpy as np
import pandas as pd
import math

# --- geodesy helper: lat/lon list -> ENU arrays using your origin ------------
def _ll_to_enu(lat_list, lon_list, origin):
    from libs.geometry_helpers import geodetic_to_enu
    enu = [geodetic_to_enu(lat, lon, 0.0, origin['lat0'], origin['lon0'], origin.get('h0', 0.0))
           for lat, lon in zip(lat_list, lon_list)]
    xy = np.array([(x, y) for x, y, _ in enu], float)
    return xy

import numpy as np
import pandas as pd

def _poly_arclen_xy(XY: np.ndarray) -> np.ndarray:
    """Cumulative arclength for ENU polyline XY (Nx2). Returns length N, S[0]=0."""
    if XY.shape[0] == 0:
        return np.array([0.0], dtype=float)
    if XY.shape[0] == 1:
        return np.array([0.0], dtype=float)
    d = np.hypot(np.diff(XY[:,0]), np.diff(XY[:,1]))
    return np.concatenate(([0.0], np.cumsum(d)))

def _clean_polyline(XY: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Remove NaNs and zero-length consecutive duplicates (within eps)."""
    XY = np.asarray(XY, float)
    # drop NaN rows
    m = np.isfinite(XY).all(axis=1)
    XY = XY[m]
    if XY.shape[0] <= 1:
        return XY
    keep = [0]
    for i in range(1, XY.shape[0]):
        if np.hypot(*(XY[i] - XY[keep[-1]])) > eps:
            keep.append(i)
    return XY[keep]

def road_links_dict_to_df(road: dict, origin: dict, *, with_arclen: bool = True) -> pd.DataFrame:
    """
    Build a tidy links DataFrame with full polylines in ENU + end-oriented views.
    Columns:
      ['link_id','u','v',
       'sx0','sy0','sx1','sy1','ex0','ey0','ex1','ey1',   # compatibility
       'x_all','y_all',                                   # full ENU polyline (lists)
       'u_x','u_y','v_x','v_y',                           # end-oriented polylines (lists)
       'u_s','v_s',                                       # optional arclengths (lists; start at 0)
       'length_m']                                        # total length
    Notes:
      - Input geometry from road['links'][lid]['lat'] / ['lon'] (WGS84 arrays).
      - Uses your _ll_to_enu(lat, lon, origin) -> Nx2.
    """
    rows = []
    links = road.get('links', {}) or {}
    for lid, rec in links.items():
        u = int(rec['start_node_id'])
        v = int(rec['end_node_id'])
        lat = rec.get('lat'); lon = rec.get('lon')
        if lat is None or lon is None or len(lat) < 2 or len(lon) < 2:
            continue

        # WGS84 -> ENU, then clean
        XY = _ll_to_enu(lat, lon, origin)  # (N,2)
        XY = _clean_polyline(XY)
        if XY.shape[0] < 2:
            continue

        x_all = XY[:,0]
        y_all = XY[:,1]

        # tiny local segments kept for backward compatibility
        s0, s1 = XY[0], XY[min(1, len(XY)-1)]
        e0, e1 = XY[-1], XY[-2]  # segment at the end node, oriented outward

        # end-oriented polylines
        u_x, u_y = x_all, y_all                # from u toward v
        v_x, v_y = x_all[::-1], y_all[::-1]    # from v toward u

        # arclengths
        if with_arclen:
            S_u = _poly_arclen_xy(XY)                # along u->v
            S_v = _poly_arclen_xy(XY[::-1])          # along v->u
            length_m = float(S_u[-1])
            u_s = S_u
            v_s = S_v
        else:
            u_s = v_s = None
            length_m = float(np.hypot(*(XY[-1] - XY[0])))  # fallback (chord)

        rows.append(dict(
            link_id=str(lid),
            u=u, v=v,
            sx0=float(s0[0]), sy0=float(s0[1]),
            sx1=float(s1[0]), sy1=float(s1[1]),
            ex0=float(e0[0]), ey0=float(e0[1]),
            ex1=float(e1[0]), ey1=float(e1[1]),
            x_all=list(map(float, x_all)),
            y_all=list(map(float, y_all)),
            u_x=list(map(float, u_x)),
            u_y=list(map(float, u_y)),
            v_x=list(map(float, v_x)),
            v_y=list(map(float, v_y)),
            u_s=(list(map(float, u_s)) if with_arclen else None),
            v_s=(list(map(float, v_s)) if with_arclen else None),
            length_m=length_m
        ))

    L = pd.DataFrame(rows)

    # Set object dtype explicitly for list columns (helps some pandas versions)
    for col in ['x_all','y_all','u_x','u_y','v_x','v_y','u_s','v_s']:
        if col in L.columns:
            L[col] = L[col].astype('object')

    return L


def road_nodes_df_from_dict(road: dict, origin: dict) -> pd.DataFrame:
    """Shortcut to get ENU nodes as ['node_id','x','y'] (wraps your function)."""
    df = road_nodes_dict_to_df_enu(road, origin)  # -> ['node_id','X','Y']
    return df.rename(columns={'node_id':'node_id','X':'x','Y':'y'})[['node_id','x','y']]




import numpy as np
import pandas as pd

def build_geo_rect_enu(geo_rect: dict, origin: dict):
    """
    geo_rect = {'min_lat':..,'min_lon':..,'max_lat':..,'max_lon':..}
    Returns (xmin, xmax, ymin, ymax) as plain floats.
    """
    from libs.geometry_helpers import geodetic_to_enu

    def _enu(lat, lon):
        x, y, _ = geodetic_to_enu(float(lat), float(lon),
                                  0.0, origin['lat0'], origin['lon0'], origin.get('h0', 0.0))
        return float(x), float(y)

    x0, y0 = _enu(geo_rect['min_lat'], geo_rect['min_lon'])
    x1, y1 = _enu(geo_rect['max_lat'], geo_rect['max_lon'])

    xmin, xmax = (min(x0, x1), max(x0, x1))
    ymin, ymax = (min(y0, y1), max(y0, y1))
    # Ensure scalars (not 0-d arrays)
    return float(xmin), float(xmax), float(ymin), float(ymax)

def mask_in_rect_xy(df: pd.DataFrame, xmin, xmax, ymin, ymax, xcol='x', ycol='y'):
    """
    Safe rectangle mask:
    - accepts X/Y or x/y
    - coerces bounds and columns to numeric scalars/Series
    """
    # normalize columns
    if xcol not in df.columns and xcol == 'x' and 'X' in df.columns:
        xcol = 'X'
    if ycol not in df.columns and ycol == 'y' and 'Y' in df.columns:
        ycol = 'Y'

    # coerce bounds to plain floats (handles np.array([val]) cases)
    xmin = float(np.asarray(xmin).ravel()[0])
    xmax = float(np.asarray(xmax).ravel()[0])
    ymin = float(np.asarray(ymin).ravel()[0])
    ymax = float(np.asarray(ymax).ravel()[0])

    # coerce columns to numeric (NaNs will be excluded by comparisons)
    x = pd.to_numeric(df[xcol], errors='coerce')
    y = pd.to_numeric(df[ycol], errors='coerce')

    return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)


def seeds_from_links_in_rect(nodes_df, links_df, geo_rect_enu=None, min_neighbors=3):
    nodes = nodes_df.copy()
    if geo_rect_enu is not None:
        xmin,xmax,ymin,ymax = geo_rect_enu
        nodes = nodes[mask_in_rect_xy(nodes, xmin,xmax,ymin,ymax)].copy()

    # undirected adjacency (collapse parallel)
    E = links_df[['u','v']].astype(int)
    A = pd.concat([E, E.rename(columns={'u':'v','v':'u'})], ignore_index=True).drop_duplicates(['u','v'])
    neigh = (A.groupby('u')['v'].apply(lambda s: sorted(set(int(x) for x in s))).rename('neighbors'))
    deg = neigh.apply(len).rename('deg')

    seeds = (nodes.set_index('node_id').join(pd.concat([deg,neigh], axis=1))
             .dropna(subset=['deg']).query('deg >= @min_neighbors').reset_index())
    return seeds[['node_id','x','y','deg','neighbors']]

def _to_xy(df):
    if {'x','y'}.issubset(df.columns): return df[['x','y']].to_numpy(float)
    if {'X','Y'}.issubset(df.columns): return df[['X','Y']].to_numpy(float)
    raise ValueError("Need X/Y (or x/y).")

def _point_to_seg_metrics(px, py, x0,y0,x1,y1):
    """Return (dist, t) where dist is perpendicular distance (m), t is projection in [0,1] ideally."""
    vx, vy = x1-x0, y1-y0
    wx, wy = px-x0, py-y0
    L2 = vx*vx + vy*vy
    if L2 == 0.0:
        t = 0.0
        dist = math.hypot(px-x0, py-y0)
    else:
        t = (wx*vx + wy*vy) / L2
        # clamp t so distance is to segment not infinite line
        t_clamped = max(0.0, min(1.0, t))
        projx, projy = x0 + t_clamped*vx, y0 + t_clamped*vy
        dist = math.hypot(px-projx, py-projy)
    return dist, t

def filter_seeds_by_link_aligned_cps(
    seeds_df, links_df, nodes_df, conn_pts_df,
    *,
    rc=20.0,              # candidate search radius around seed (m)
    dist_max=5.0,         # max perpendicular distance (m) to accept a CP as aligned
    min_supported_links=2,
    geo_rect_enu=None
):
    """
    Keep seeds that have lane connection points (CPs) close to the *local link line*
    at the seed end for at least `min_supported_links` distinct incident links.
    Returns: kept_seeds, diagnostics (per seed: supported link_ids, counts)
    """
    seeds = seeds_df.copy()
    if geo_rect_enu is not None:
        xmin,xmax,ymin,ymax = geo_rect_enu
        seeds = seeds[mask_in_rect_xy(seeds, xmin,xmax,ymin,ymax)].copy()

    # CP coordinates (optionally crop for speed)
    P = conn_pts_df.copy()
    if geo_rect_enu is not None:
        if {'x','y'}.issubset(P.columns):
            P = P[mask_in_rect_xy(P, xmin,xmax,ymin,ymax, 'x','y')]
        else:
            P = P[mask_in_rect_xy(P, xmin,xmax,ymin,ymax, 'X','Y')]
    Pxy = _to_xy(P)

    # simple grid for radius queries
    cell = max(rc, 1.0)
    minx, miny = Pxy.min(0) if len(Pxy) else (0.0,0.0)
    buckets = {}
    for i,(x,y) in enumerate(Pxy):
        ix=int((x-minx)/cell); iy=int((y-miny)/cell)
        buckets.setdefault((ix,iy), []).append(i)

    def in_radius(x,y):
        ix=int((x-minx)/cell); iy=int((y-miny)/cell)
        cand=[]
        for dx in(-1,0,1):
            for dy in(-1,0,1):
                cand += buckets.get((ix+dx,iy+dy), [])
        if not cand: return np.array([], int)
        d2=(Pxy[cand,0]-x)**2 + (Pxy[cand,1]-y)**2
        return np.array([cand[i] for i in np.where(d2 <= rc*rc)[0]], int)

    # index links by node end
    links = links_df.copy()
    links[['u','v']] = links[['u','v']].astype(int)
    by_u = links.groupby('u').indices
    by_v = links.groupby('v').indices

    kept_rows = []
    diags = []

    for _, s in seeds.iterrows():
        sid, sx, sy = int(s['node_id']), float(s['x']), float(s['y'])
        # incident links at this node
        idxs = list(by_u.get(sid, [])) + list(by_v.get(sid, []))
        if not idxs:
            continue
        idx_cp = in_radius(sx, sy)

        supported = []
        if len(idx_cp):
            CP = Pxy[idx_cp]
            for li in idxs:
                row = links.iloc[li]
                if sid == int(row['u']):
                    x0,y0,x1,y1 = row['sx0'],row['sy0'],row['sx1'],row['sy1']
                else:  # sid == v
                    x0,y0,x1,y1 = row['ex0'],row['ey0'],row['ex1'],row['ey1']

                # check all CPs nearby for alignment to this end-segment
                ok = 0
                for (px,py) in CP:
                    dist, t = _point_to_seg_metrics(px, py, x0,y0,x1,y1)
                    # need close and projected roughly forward (t in [0,1] or slightly >0)
                    # compute (dist, t) with unclamped t
                    ok_forward = (0.0 <= t <= 1.2)   # allow a small overshoot
                    if dist <= dist_max and ok_forward:
                        ok += 1
                        break
                if ok:
                    supported.append(row['link_id'])

        if len(set(supported)) >= min_supported_links:
            kept_rows.append(s)
        diags.append(dict(node_id=sid, supported_links=list(set(supported)),
                          n_supported=len(set(supported))))

    kept = pd.DataFrame(kept_rows).reset_index(drop=True) if kept_rows else seeds.iloc[0:0]
    diag_df = pd.DataFrame(diags)
    return kept, diag_df

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# ---------- rectangle helpers ----------
def build_geo_rect_enu(geo_rect: dict, origin: dict):
    from libs.geometry_helpers import geodetic_to_enu
    def _enu(lat, lon):
        x,y,_ = geodetic_to_enu(float(lat), float(lon), 0.0,
                                origin['lat0'], origin['lon0'], origin.get('h0',0.0))
        return float(x), float(y)
    x0,y0 = _enu(geo_rect[0], geo_rect[1])
    x1,y1 = _enu(geo_rect[2], geo_rect[3])
    return (min(x0,x1), min(y0,y1), max(x0,x1), max(y0,y1))

def mask_in_rect_xy(df, xmin, xmax, ymin, ymax, xcol='x', ycol='y'):
    if xcol not in df and xcol=='x' and 'X' in df: xcol='X'
    if ycol not in df and ycol=='y' and 'Y' in df: ycol='Y'
    x = pd.to_numeric(df[xcol], errors='coerce')
    y = pd.to_numeric(df[ycol], errors='coerce')
    return (x>=float(xmin)) & (x<=float(xmax)) & (y>=float(ymin)) & (y<=float(ymax))

# ---------- geometry helpers ----------
def _to_xy(df):
    if {'x','y'}.issubset(df.columns): return df[['x','y']].to_numpy(float)
    if {'X','Y'}.issubset(df.columns): return df[['X','Y']].to_numpy(float)
    raise ValueError("Need X/Y (or x/y).")

import numpy as np, math
import matplotlib.pyplot as plt

def debug_plot_node_cps_by_color(
    ax, 
    nodes_df,            # road nodes ['node_id','x','y'] or ['node_id','X','Y']
    links_df,            # must include u,v and local end segments: sx0,sy0,sx1,sy1, ex0,ey0,ex1,ey1
    conn_pts_df,         # lane CPs with X/Y or x/y (+ lane meta columns if available)
    *,
    geo_rect=None, origin=None,        # rectangle in lat/lon + origin (optional)
    node_ids=None,                     # list of candidate node_ids (optional)
    mode='tube',                       # 'tube' or 'radius'
    rc=18.0,                           # radius if mode='radius'
    L_win=22.0,                        # tube longitudinal window (m)
    dist_max=5.0,                      # tube perpendicular tolerance (m)
    theta_max_deg=25.0,                # cone gate for tube mode
    alpha_front=0.5,                   # keep only the first alpha_front of local segment
    figure_size=(9,5),
    λ_score=0.2                        # weight for perpendicular distance in global tie-break
):
    """Plot CPs found per candidate node (different color per node) within rectangle.
       Post-processing: for each node keep ONLY the closest CP per (group,lane,end) key.
       If two nodes want the same CP, assign to node with minimal front-distance score."""
    import pandas as pd  # local to avoid hard dependency if not needed

    # ---------- helpers ----------
    def _to_xy(df):
        if {'x','y'}.issubset(df.columns): return df[['x','y']].to_numpy(float)
        if {'X','Y'}.issubset(df.columns): return df[['X','Y']].to_numpy(float)
        raise ValueError("Need X/Y (or x/y) in DataFrame.")

    def _tube_metrics(px, py, x0, y0, x1, y1):
        """Return (s, d): s=longitudinal (m) from (x0,y0) toward (x1,y1); d=perp distance (m)."""
        vx, vy = x1-x0, y1-y0
        wx, wy = px-x0, py-y0
        L = math.hypot(vx, vy)
        if L == 0.0:
            return 0.0, math.hypot(px-x0, py-y0)
        s = (wx*vx + wy*vy) / L
        d = abs(wx*vy - wy*vx) / L
        return s, d

    def _seg_geom(x0,y0,x1,y1):
        vx, vy = x1-x0, y1-y0
        L = math.hypot(vx, vy) + 1e-9
        return L, vx/L, vy/L  # length and unit direction

    def _cp_group_key_row(row):
        # Prefer Python-style export
        if {'lane_group_ref','lane_no','end'}.issubset(row.index):
            return (int(row['lane_group_ref']), int(row['lane_no']), str(row['end']))
        # MATLAB-style names
        if {'GroupID','LaneNo','Dir'}.issubset(row.index):
            return (int(row['GroupID']), int(row['LaneNo']), str(row['Dir']))
        return None

    def _cp_angle_key(px, py, nx, ny, nbins=12):
        ang = math.atan2(py-ny, px-nx)  # [-pi,pi]
        b = int(((ang + math.pi) / (2*math.pi)) * nbins) % nbins
        return ('ANG', b)

    # ---------- normalize inputs ----------
    if {'node_id','x','y'}.issubset(nodes_df.columns):
        N = nodes_df[['node_id','x','y']].copy()
    else:
        N = nodes_df.rename(columns={'node_id':'node_id','X':'x','Y':'y'})[['node_id','x','y']].copy()
    N['node_id'] = N['node_id'].astype(int)

    L = links_df.copy()
    L[['u','v']] = L[['u','v']].astype(int)

    # crop to rectangle if given (expects your existing helpers)
    if geo_rect is not None and origin is not None:
        xmin,xmax,ymin,ymax = build_geo_rect_enu(geo_rect, origin)
        N = N[mask_in_rect_xy(N, xmin,xmax,ymin,ymax)].copy()
        CP_full = conn_pts_df[mask_in_rect_xy(
            conn_pts_df, xmin,xmax,ymin,ymax,
            'x' if 'x' in conn_pts_df.columns else 'X',
            'y' if 'y' in conn_pts_df.columns else 'Y'
        )].copy()
    else:
        CP_full = conn_pts_df.copy()

    if node_ids is not None:
        N = N[N['node_id'].isin(list(map(int, node_ids)))].copy()

    if N.empty or CP_full.empty:
        print("No nodes or CPs in the selection.")
        return

    P = _to_xy(CP_full)
    Xcp = CP_full['x'] if 'x' in CP_full.columns else CP_full['X']
    Ycp = CP_full['y'] if 'y' in CP_full.columns else CP_full['Y']

    # Precompute CP keys (group,lane,end) once
    cp_keys = []
    for _, r in CP_full.iterrows():
        cp_keys.append(_cp_group_key_row(r))

    node_pos = {int(r.node_id):(float(r.x), float(r.y)) for _,r in N.iterrows()}

    # coarse grid for nearby lookup
    cell = max(rc if mode=='radius' else L_win, 5.0)
    minx, miny = P.min(0)
    buckets = {}
    for i,(x,y) in enumerate(P):
        ix=int((x-minx)/cell); iy=int((y-miny)/cell)
        buckets.setdefault((ix,iy), []).append(i)
    def nearby_idx(x,y):
        ix=int((x-minx)/cell); iy=int((y-miny)/cell)
        cand=[]
        for dx in(-1,0,1):
            for dy in(-1,0,1):
                cand += buckets.get((ix+dx,iy+dy), [])
        return cand

    # group links by end for drawing tube segments
    by_u = L.groupby('u').indices
    by_v = L.groupby('v').indices

    # ---------- collection (per-node, keep all candidates) ----------
    per_node_candidates = {}  # nid -> list of dicts {k,s,d,key}

    def _add_candidate(nid, k, s, d, nx, ny):
        key = cp_keys[k]
        if key is None:  # fallback one-per-ray if group/meta missing
            key = _cp_angle_key(P[k,0], P[k,1], nx, ny)
        per_node_candidates.setdefault(nid, []).append({'k':k, 's':s, 'd':d, 'key':key})

    theta_max = math.radians(theta_max_deg)

    for _, nrow in N.iterrows():
        nid, nx, ny = int(nrow['node_id']), float(nrow['x']), float(nrow['y'])

        # assemble local outward segments for this node
        segs = []
        for li in by_u.get(nid, []):
            r = L.iloc[li]
            segs.append((r['sx0'],r['sy0'],r['sx1'],r['sy1']))
        for li in by_v.get(nid, []):
            r = L.iloc[li]
            segs.append((r['ex0'],r['ey0'],r['ex1'],r['ey1']))

        cand_idx = nearby_idx(nx, ny)
        if not cand_idx:
            continue

        if mode == 'radius':
            for k in cand_idx:
                dx = P[k,0]-nx; dy = P[k,1]-ny
                d = math.hypot(dx,dy)
                if d <= rc:
                    # emulate "front-distance": use Euclidean as s, d=0 for scoring
                    _add_candidate(nid, k, d, 0.0, nx, ny)
        else:
            # tube mode with front clip + cone
            for k in cand_idx:
                px, py = P[k]
                best_local = None  # keep best (smallest s) segment that accepts this CP
                for (x0,y0,x1,y1) in segs:
                    s, d = _tube_metrics(px, py, x0,y0,x1,y1)
                    Lseg, ux, uy = _seg_geom(x0,y0,x1,y1)
                    if not (0.0 <= s <= min(L_win, alpha_front*Lseg)):
                        continue
                    if d > dist_max:
                        continue
                    dx, dy = px-x0, py-y0
                    nd = math.hypot(dx, dy) + 1e-9
                    cosang = (dx*ux + dy*uy) / nd
                    if cosang < math.cos(theta_max):
                        continue
                    # candidate accepted for this segment
                    if (best_local is None) or (s < best_local[0] or (s==best_local[0] and d < best_local[1])):
                        best_local = (s, d)
                if best_local is not None:
                    s, d = best_local
                    _add_candidate(nid, k, s, d, nx, ny)

    # ---------- per-node pruning: keep closest per (group,lane,end) ----------
    pruned_by_node = {}   # nid -> dict key->cp_index
    sd_lookup = {}        # nid -> cp_index -> (s,d)  (for later scoring)

    for nid, items in per_node_candidates.items():
        # choose winners per key
        best = {}
        for it in items:
            if it['s'] < 0:
                continue
            key = it['key']; k = it['k']; s = it['s']; d = it['d']
            if key not in best or (s < best[key]['s'] or (s == best[key]['s'] and d < best[key]['d'])):
                best[key] = {'k': k, 's': s, 'd': d}

        # winners for this node
        pruned_by_node[nid] = {key: v['k'] for key, v in best.items()}

        # IMPORTANT: make sd_lookup from the WINNERS, not from raw items
        sd_lookup[nid] = {v['k']: (v['s'], v['d']) for v in best.values()}


    # ---------- global conflict resolution (unique owner per CP) ----------
    owners = {}  # cp_index -> (nid, score)
    for nid, m in pruned_by_node.items():
        for _, k in m.items():
            # find s,d for this nid,k
            s, d = sd_lookup[nid].get(k, (1e9, 1e9))
            score = s + λ_score*d
            if (k not in owners) or (score < owners[k][1]):
                owners[k] = (nid, score)

    final_cp_by_node = {}
    for k, (nid, _) in owners.items():
        final_cp_by_node.setdefault(nid, []).append(k)

    # ---------- plotting ----------
    if ax is None:
        fig, ax = plt.subplots(figsize=figure_size)

    # background: all CPs
    ax.plot(Xcp, Ycp, '.', ms=3, color='0.85', alpha=0.7, zorder=1, label='all CPs')

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    legend_entries = []

    for i, (_, row) in enumerate(N.iterrows()):
        nid, nx, ny = int(row['node_id']), float(row['x']), float(row['y'])
        col = colors[i % len(colors)]

        # draw outward link stubs (visual context)
        for li in by_u.get(nid, []):
            r = L.iloc[li]
            ax.plot([r['sx0'],r['sx1']], [r['sy0'],r['sy1']], '--', lw=2, color=col, alpha=0.8, zorder=2)
        for li in by_v.get(nid, []):
            r = L.iloc[li]
            ax.plot([r['ex0'],r['ex1']], [r['ey0'],r['ey1']], '--', lw=2, color=col, alpha=0.8, zorder=2)

        # winners for this node
        idxs = final_cp_by_node.get(nid, [])
        if idxs:
            ax.plot(P[idxs,0], P[idxs,1], 'o', ms=6, color=col, zorder=3)
            legend_entries.append((col, f'node {nid}  (n={len(idxs)})'))

        # draw node on top
        ax.plot([nx],[ny],'o', mfc='white', mec=col, mew=2.0, ms=10, zorder=4)

    if geo_rect is not None and origin is not None:
        xmin,xmax,ymin,ymax = build_geo_rect_enu(geo_rect, origin)
        ax.plot([xmin,xmax,xmax,xmin,xmin],[ymin,ymin,ymax,ymax,ymin], ':', color='k', lw=1)

    for c, txt in legend_entries[:10]:
        ax.plot([], [], 'o', color=c, label=txt)
    ax.legend(loc='best', frameon=True, fontsize=9)

    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Connection points per candidate node ({mode}) — pruned')
    plt.show()

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List

# --- helpers ---
# --- replace the recursive placeholder with this ---
from libs.geometry_helpers import geodetic_to_enu  # your real converter

def _ll_to_enu(lat, lon, origin):
    """
    Convert arrays/lists of lat, lon to ENU Nx2 using geometry.geodetic_to_enu.
    Handles both vectorized and scalar-only implementations of geodetic_to_enu.
    """
    lat = np.asarray(lat, dtype=float).ravel()
    lon = np.asarray(lon, dtype=float).ravel()
    if lat.size != lon.size:
        raise ValueError("lat and lon must have the same length")
    lat0 = float(origin.get("lat0"))
    lon0 = float(origin.get("lon0"))
    h0 = float(origin.get("h0", 0.0))

    # Try vectorized call first (if your geodetic_to_enu supports it)
    try:
        x, y, _ = geodetic_to_enu(lat, lon, np.full_like(lat, h0, dtype=float), origin)
        return np.column_stack((np.asarray(x, float), np.asarray(y, float)))
    except Exception:
        # Fallback: scalar loop
        out = np.empty((lat.size, 2), dtype=float)
        for i, (la, lo) in enumerate(zip(lat, lon)):
            xi, yi, _zi = geodetic_to_enu(float(la), float(lo), h0, lat0, lon0, h0)
            out[i, 0] = float(xi)
            out[i, 1] = float(yi)
        return out

def _geobbox_to_enu_bbox(bbox_latlon, origin):
    lat_min, lon_min, lat_max, lon_max = map(float, bbox_latlon)
    corners_lat = [lat_min, lat_min, lat_max, lat_max]
    corners_lon = [lon_min, lon_max, lon_min, lon_max]
    XY = _ll_to_enu(corners_lat, corners_lon, origin)  # 4x2
    xs, ys = XY[:, 0], XY[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

def _dist_point_to_segment(px, py, x0, y0, x1, y1) -> float:
    """
    Distance from point P=(px,py) to segment [A=(x0,y0), B=(x1,y1)] in ENU.
    """
    ax, ay = x0, y0
    bx, by = x1, y1
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    denom = abx*abx + aby*aby
    if denom <= 1e-18:
        # degenerate segment
        dx, dy = px - ax, py - ay
        return float(np.hypot(dx, dy))
    t = (apx*abx + apy*aby) / denom
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    cx, cy = ax + t*abx, ay + t*aby
    return float(np.hypot(px - cx, py - cy))

def find_deadend_cps(
    *,
    Nodes: pd.DataFrame,        # lane CP nodes
    nodes_df: pd.DataFrame,     # road nodes: ['node_id','x','y']
    links_df: pd.DataFrame,     # road links: ['link_id','u','v',...]
    road: Dict[str, Any],       # must have link_lane_group_refs
    dead_ends: set[int],
    bbox_enu: tuple[float,float,float,float] | None = None,
    r_dead_node: float = 5.0,   # tight radius: dead-end is “clean”
    keep_k: int = 2,            # keep top-k CPs per (rid, link_end)
) -> pd.DataFrame:
    """
    Returns CP matches only for dead-ends.
    Output columns compatible with your CP table:
      road_node_id, road_x, road_y, link_id, link_end, lane_node_id, lane_x, lane_y, d_node, ...
    """

    # ---- road nodes subset ----
    rn = nodes_df.copy()
    rn["node_id"] = rn["node_id"].astype("int64", errors="ignore")
    if bbox_enu is not None:
        xmin, ymin, xmax, ymax = bbox_enu
        rn = rn[(rn["x"] >= xmin) & (rn["x"] <= xmax) & (rn["y"] >= ymin) & (rn["y"] <= ymax)]
    rn = rn[rn["node_id"].isin({int(x) for x in dead_ends})]
    if rn.empty:
        return pd.DataFrame(columns=[
            "road_node_id","road_x","road_y","link_id","link_end",
            "lane_node_id","lane_x","lane_y","d_node",
            "lane_groups","lane_lnos","lane_end_counts","incident_edges"
        ])

    # ---- laned link ids ----
    lane_links = set(road.get("link_lane_group_refs", {}).keys() or [])
    lane_links = {int(str(x).strip().strip("'").strip('"')) for x in lane_links}

    # ---- adjacency: link ends incident to road nodes, but only laned links ----
    # for dead-end definition you expect exactly one laned end, but don’t assume.
    inc = pd.concat([
        links_df[["u","link_id"]].rename(columns={"u":"road_node_id"}).assign(link_end="u"),
        links_df[["v","link_id"]].rename(columns={"v":"road_node_id"}).assign(link_end="v"),
    ], ignore_index=True)
    inc["road_node_id"] = inc["road_node_id"].astype("int64", errors="ignore")
    inc["link_id"] = inc["link_id"].apply(lambda z: int(str(z).strip().strip("'").strip('"')) if pd.notna(z) else None)
    inc = inc.dropna(subset=["link_id"])
    inc = inc[inc["link_id"].isin(lane_links)]
    inc = inc[inc["road_node_id"].isin(rn["node_id"].values)]
    if inc.empty:
        return pd.DataFrame(columns=[
            "road_node_id","road_x","road_y","link_id","link_end",
            "lane_node_id","lane_x","lane_y","d_node",
            "lane_groups","lane_lnos","lane_end_counts","incident_edges"
        ])

    # ---- lane CP arrays ----
    lane_ids = Nodes["node_id"].to_numpy()
    lane_X   = Nodes["X"].to_numpy(dtype=float)
    lane_Y   = Nodes["Y"].to_numpy(dtype=float)

    get_groups = Nodes.get("lane_groups", pd.Series([[]]*len(Nodes))).to_numpy(object)
    get_lnos   = Nodes.get("lane_lnos", pd.Series([[]]*len(Nodes))).to_numpy(object)
    get_counts = Nodes.get("lane_end_counts", pd.Series([{}]*len(Nodes))).to_numpy(object)
    get_incedg = Nodes.get("incident_edges", pd.Series([[]]*len(Nodes))).to_numpy(object)

    # ---- main: per dead-end node, pick nearest CP(s) within tight radius ----
    rows = []
    rn_idx = rn.set_index("node_id")[["x","y"]]

    # pre-group incident ends for speed
    by_rid = inc.groupby("road_node_id")

    for rid in rn_idx.index.to_list():
        rx, ry = float(rn_idx.loc[rid, "x"]), float(rn_idx.loc[rid, "y"])

        # candidates near dead-end road node
        d = np.hypot(lane_X - rx, lane_Y - ry)
        cand = np.nonzero(d <= float(r_dead_node))[0]
        if cand.size == 0:
            continue

        G = by_rid.get_group(int(rid)) if int(rid) in by_rid.groups else None
        if G is None or G.empty:
            continue

        # For each incident laned link-end, keep top-k closest CPs.
        for link_id, link_end in G[["link_id","link_end"]].itertuples(index=False, name=None):
            # sort candidates by distance to road node
            order = cand[np.argsort(d[cand])]
            order = order[:max(1, int(keep_k))]

            for j in order:
                rows.append(dict(
                    road_node_id=int(rid),
                    road_x=rx,
                    road_y=ry,
                    link_id=int(link_id),
                    link_end=str(link_end),  # 'u' or 'v'
                    lane_node_id=int(lane_ids[j]),
                    lane_x=float(lane_X[j]),
                    lane_y=float(lane_Y[j]),
                    d_node=float(d[j]),
                    lane_groups=get_groups[j],
                    lane_lnos=get_lnos[j],
                    lane_end_counts=get_counts[j],
                    incident_edges=get_incedg[j],
                    cp_type="deadend",
                ))

    if not rows:
        return pd.DataFrame(columns=[
            "road_node_id","road_x","road_y","link_id","link_end",
            "lane_node_id","lane_x","lane_y","d_node",
            "lane_groups","lane_lnos","lane_end_counts","incident_edges"
        ])

    C = pd.DataFrame(rows)

    # final dedup safety: per (road_node_id, link_id, link_end) keep nearest by d_node
    C.sort_values(["road_node_id","link_id","link_end","d_node"], inplace=True)
    C = C.groupby(["road_node_id","link_id","link_end"], as_index=False).head(int(keep_k))

    return C

# --- main matcher ---
def find_lane_connection_points_near_road_nodes(
    Nodes: pd.DataFrame,
    nodes_df: pd.DataFrame,            # road nodes: ['node_id','x','y']
    links_df: pd.DataFrame,            # road links with local end segments from road_links_dict_to_df(...)
    junctions: Dict[str, Any],
    road: Dict[str, Any], 
    bbox_enu: Tuple[float,float,float,float],
    origin: Dict[str, Any],
    r_node: float = 25.0,               # max distance from lane node to road node (meters)
    r_seg: float = 5.0,                  # max distance from lane node to local link-end segment (meters)
    L_window = 30.0,   # meters of polyline to consider from the touched end
    endpoint_cap = 0.75,  # meters; optional small cap at the very node
    ax = None, 
    debug = False,
) -> pd.DataFrame:
    """
    Return a tidy DataFrame of matched lane-graph nodes (connection points) to road-topology nodes/links.

    Output columns:
      ['road_node_id','road_x','road_y','link_id','link_end',   # 'u' if at start node, 'v' if at end node
       'lane_node_id','lane_x','lane_y','d_node','d_seg',
       'lane_groups','lane_lnos','lane_end_counts','incident_edges']

    Dedup rule (3): For each (road_node_id, link_id), keep only the candidate with the **smallest d_node**.
    """
    # --- 0) filter road nodes by geo rectangle (convert to ENU box) ---
    xmin, ymin, xmax, ymax = bbox_enu
    # _geobbox_to_enu_bbox(bbox_latlon, origin)
    rn = nodes_df[(nodes_df['x'] >= xmin) & (nodes_df['x'] <= xmax) &
                  (nodes_df['y'] >= ymin) & (nodes_df['y'] <= ymax)].copy()

    def is_drivable_meta(meta):
        """
        A CP is drivable if lane_end_keys contains
        'FORWARD' or 'BACKWARD' anywhere (possibly nested).
        """
        if meta is None:
            return False

        try:
            for ek in meta:
                if ek is None:
                    continue

                # direct string
                if isinstance(ek, str):
                    if ek in ("FORWARD", "BACKWARD"):
                        return True

                # tuple / list: scan contents
                elif isinstance(ek, (tuple, list)):
                    for item in ek:
                        if item in ("FORWARD", "BACKWARD"):
                            return True
        except TypeError:
            # non-iterable garbage → not drivable
            return False

        return False

    from collections import defaultdict

    def compute_valid_lane_groups_for_node(
        rid_i: int,
        here_link_df: pd.DataFrame,
        road: dict,
        *,
        link_id_col="link_id", u_col="u", v_col="v",
        link_to_lane_groups_key="link_to_lane_groups",
        lane_group_to_links_key="lane_group_to_links",
        min_links_per_lg=2,
    ):
        # 1) connected links at this road node
        m = (here_link_df[u_col].astype("int64") == rid_i) | (here_link_df[v_col].astype("int64") == rid_i)
        links = set(here_link_df.loc[m, link_id_col].astype("int64").tolist())
        if not links:
            return set(), set()

        # 2) valid lane groups = touch >=2 of these links
        lg2links_all = road.get(lane_group_to_links_key, {}) or {}
        valid_lgs = set()
        for lg, lids in lg2links_all.items():
            k = len(links.intersection(lids))
            if k >= int(min_links_per_lg):
                valid_lgs.add(str(lg))

        return links, valid_lgs

    import ast

    def normalize_valid_lgs(valid_lgs):
        """
        Converts:
        - set of dict-strings like "{'lane_group_id': '4919...', ...}"
        - list of dicts
        - mixed
        into: set of lane_group_id strings like {"4919...", ...}
        """
        out = set()
        for x in (valid_lgs or []):
            if x is None:
                continue

            # dict already
            if isinstance(x, dict):
                if "lane_group_id" in x:
                    out.add(str(x["lane_group_id"]))
                continue

            # plain id already
            if isinstance(x, (int, str)) and str(x).isdigit():
                out.add(str(x))
                continue

            # stringified dict
            if isinstance(x, str) and "lane_group_id" in x:
                try:
                    d = ast.literal_eval(x)
                    if isinstance(d, dict) and "lane_group_id" in d:
                        out.add(str(d["lane_group_id"]))
                except Exception:
                    pass

        return out

    # boring = compute_boring_road_nodes(
    #     nodes_df=rn,          # only bbox nodes
    #     links_df=links_df,    # full link set in scope
    #     lane_links=lane_links,
    #     min_deg=3,
    # )

    if junctions:
        rn = rn[rn["node_id"].astype("int64", errors="ignore").isin(junctions)].copy()

    if rn.empty:
        return pd.DataFrame(columns=[
            'road_node_id','road_x','road_y','link_id','link_end',
            'lane_node_id','lane_x','lane_y','d_node','d_seg',
            'lane_groups','lane_lnos','lane_end_counts','incident_edges'
        ])

    # Build a per-node list of incident link-ends with full rec attached
    # Each item: (endtag, rec)
    inc_by_node = {}

    # If you want to restrict to a subset of link_ids (e.g., only laned links), do it HERE.
    # For now keep behavior identical: include all link-ends.
    for rec in links_df.itertuples(index=False):
        u = getattr(rec, "u", None)
        v = getattr(rec, "v", None)
        if u is not None and not pd.isna(u):
            inc_by_node.setdefault(int(u), []).append(("u", rec))
        if v is not None and not pd.isna(v):
            inc_by_node.setdefault(int(v), []).append(("v", rec))

    lane_links = None
    if road is not None:
        lane_links = set(road.get("link_lane_group_refs", {}).keys())

    lane_links_str = set(str(x).strip().strip("'").strip('"') for x in lane_links)
    lg2links, link2lgs = build_lg2links_and_link2lgs(road)

    # --- 1) pre-extract lane node coordinates (ENU) & ids ---
    # Nodes: ['node_id','X','Y', ... enrich columns ...]
    lane_ids = Nodes['node_id'].to_numpy()
    lane_X = Nodes['X'].to_numpy(dtype=float)
    lane_Y = Nodes['Y'].to_numpy(dtype=float)

    # convenience accessors (could be empty lists)
    get_groups = Nodes.get('lane_groups', pd.Series([[]]*len(Nodes))).to_numpy(object)
    get_lnos   = Nodes.get('lane_lnos', pd.Series([[]]*len(Nodes))).to_numpy(object)
    get_counts = Nodes.get('lane_end_counts', pd.Series([{}]*len(Nodes))).to_numpy(object)
    get_incedg = Nodes.get('incident_edges', pd.Series([[]]*len(Nodes))).to_numpy(object)
    # --- NEW: cross-check lane_end_keys across candidate CPs ---
    lane_end_keys_arr = Nodes['lane_end_keys'].to_numpy(object)

    # ---- compute once ----
    lane_meta = Nodes.get('incident_edge_meta', pd.Series([None]*len(Nodes)))

    Nodes['is_drivable_cp'] = lane_meta.apply(is_drivable_meta).astype(bool)
    # --- 2) build adjacency: for each road node, find its incident links and the proper local segment ---
    # links_df columns from your road_links_dict_to_df:
    #  ['link_id','u','v','sx0','sy0','sx1','sy1','ex0','ey0','ex1','ey1']
    # Index by 'u' and 'v' for quick lookup
    by_u = links_df.groupby('u')
    by_v = links_df.groupby('v')

    rows = []
    append = rows.append

    def _norm_lgs(v):
        if not isinstance(v, (list, tuple)):
            return []
        out = []
        for x in v:
            if isinstance(x, dict) and "lane_group_id" in x:
                out.append(str(x["lane_group_id"]))
            else:
                out.append(str(x))
        return out

    lane_groups_str = np.array([_norm_lgs(v) for v in get_groups], dtype=object)
    # loop over candidate road nodes in the bbox
    for rid, rx, ry in rn[['node_id','x','y']].itertuples(index=False, name=None):
        # Debug
        if debug is True:
            cps_xy_this_rid = []   # list of (x,y) for CPs that got ACCEPTED for this rid

        cand = []  # list of tuples holding everything needed for selection+final row
        rid_i = int(rid)
        # collect incident links touching this road node
        inc_links = inc_by_node.get(rid_i, [])

        # inside per-node loop, after rid_i is known:
        connected_links, valid_lgs = compute_valid_lane_groups_for_node(
            rid_i,
            here_link_df=links_df,
            road=road,
            min_links_per_lg=2
        )

        if not connected_links:
            continue

        # Keep only incident links that are in your lane_links_str (your existing filter)
        inc_links = [(endtag, rec) for (endtag, rec) in inc_links
                    if str(getattr(rec, "link_id")).strip().strip("'").strip('"') in lane_links_str]
        if not inc_links:
            continue

        # OPTIONAL but recommended: restrict inc_links further to connected_links (robust)
        inc_links = [(endtag, rec) for (endtag, rec) in inc_links
                    if int(getattr(rec, "link_id")) in connected_links]
        if not inc_links:
            continue


        # ---- 1) coarse candidate CPs near this road node ----
        dx = lane_X - float(rx)
        dy = lane_Y - float(ry)
        d_node = np.hypot(dx, dy)
        cand_mask = (d_node <= float(r_node)) & Nodes['is_drivable_cp'].to_numpy()
        if not cand_mask.any():
            continue
        c_idx = np.nonzero(cand_mask)[0]

        # ---- 2) cache end-window polylines for all incident link-ends (per rid) ----
        # end_windows: list of dicts: { 'lid':int, 'endtag':str, 'Xw':np.ndarray, 'Yw':np.ndarray, 'x0':float, 'y0':float }
        end_windows = []
        for endtag, rec in inc_links:
            lid = getattr(rec, "link_id", None)
            if lid is None or pd.isna(lid):
                continue
            lid = int(lid)

            # choose FULL end-oriented polyline, with robust fallbacks
            if endtag == 'u':
                if _is_seq2(getattr(rec, 'u_x', None)) and _is_seq2(getattr(rec, 'u_y', None)):
                    Xfull = _as_float_np(getattr(rec, 'u_x'))
                    Yfull = _as_float_np(getattr(rec, 'u_y'))
                elif all(hasattr(rec, k) for k in ('sx0','sy0','sx1','sy1')):
                    Xfull = np.array([rec.sx0, rec.sx1], dtype=float)
                    Yfull = np.array([rec.sy0, rec.sy1], dtype=float)
                else:
                    continue
            else:  # 'v'
                if _is_seq2(getattr(rec, 'v_x', None)) and _is_seq2(getattr(rec, 'v_y', None)):
                    Xfull = _as_float_np(getattr(rec, 'v_x'))
                    Yfull = _as_float_np(getattr(rec, 'v_y'))
                elif all(hasattr(rec, k) for k in ('ex0','ey0','ex1','ey1')):
                    Xfull = np.array([rec.ex0, rec.ex1], dtype=float)
                    Yfull = np.array([rec.ey0, rec.ey1], dtype=float)
                else:
                    continue

            Xw, Yw = _clip_polyline_by_length(Xfull, Yfull, L_window)
            if len(Xw) < 2:
                continue

            end_windows.append({
                "lid": lid,
                "endtag": endtag,
                "rec": rec,          # keep original rec to reuse link_id etc.
                "Xw": Xw,
                "Yw": Yw,
                "x0": float(Xw[0]),
                "y0": float(Yw[0]),
            })

        if not end_windows:
            continue

        # ---- 3) For each candidate CP, infer candidate links from lane_groups, then pick best end by d_seg ----
        for j in c_idx:
            cp_id = int(lane_ids[j])

            # CP's lane groups, as strings
            LG = lane_groups_str[j]  # list[str] (already)
            if not LG:
                continue

            valid_lg_ids = normalize_valid_lgs(valid_lgs)
            # NEW: intersect with valid lane groups for this junction node
            LGv = [g for g in LG if g in valid_lg_ids]
            if not LGv:
                continue  # CP not relevant for this junction according to topology rule

            # derive candidate links only from valid lane groups
            cand_links = set()
            for g in LGv:
                cand_links |= lg2links.get(g, set())

            # restrict end windows by lane-group-derived candidate links if available
            if cand_links:
                ends = [ew for ew in end_windows if ew["lid"] in cand_links]
                if not ends:
                    continue
            else:
                # If CP has no lane_groups mapping, you can either:
                #  - skip (safer), OR
                #  - fall back to geometry-only matching.
                # I recommend skipping to avoid wrong link_id assignments.
                continue

            best = None  # (d_seg, d_node, lid, endtag, s_on_poly)
            best_dj = None
            best_s = None
            best_lid = None
            best_endtag = None

            xcp = float(lane_X[j])
            ycp = float(lane_Y[j])

            for ew in ends:
                Xw = ew["Xw"]; Yw = ew["Yw"]

                seg_idx, t, qx, qy, dj, s_abs = _point_polyline_metrics(xcp, ycp, Xw, Yw)

                accept = (dj <= float(r_seg))
                if not accept and endpoint_cap and float(endpoint_cap) > 0.0:
                    d0 = math.hypot(xcp - ew["x0"], ycp - ew["y0"])
                    if d0 <= float(endpoint_cap):
                        accept = True
                        dj = d0
                        s_abs = 0.0

                if not accept:
                    continue

                key = (float(dj), float(d_node[j]), int(ew["lid"]), str(ew["endtag"]), float(s_abs))
                if (best is None) or (key < best):
                    best = key
                    best_dj = float(dj)
                    best_s = float(s_abs)
                    best_lid = int(ew["lid"])
                    best_endtag = str(ew["endtag"])                

            if best is None:
                continue

            # choose a representative lane group *id* for selection
            # (simplest: take the first one; better: take one that maps to best_lid)
            LG = lane_groups_str[j]  # list[str]
            if not LG:
                continue

            # if you want "lane group must belong to this link"
            LG_ok = [g for g in LG if best_lid in lg2links.get(g, set())]
            lg_pick = (LG_ok[0] if LG_ok else LG[0])

            cand.append((
                int(best_lid), str(best_endtag), str(lg_pick),            # grouping key pieces
                float(best_s), float(best_dj), float(d_node[j]),          # ranking (s_on_poly first!)
                int(lane_ids[j]), j,                                                       # index back to arrays
            ))

        # 1) For each (link_id, link_end): keep only the *closest lane-group(s)*.
        #    Implementation: find the minimum s_on_poly per lane-group, then keep only the lane-group(s)
        #    whose min_s is within a tolerance of the best lane-group on that link end.
        keep_tol_s = 1.0  # meters; lane-groups essentially co-located at the mouth will both survive

        # per (lid,endtag,lg): best (min) s
        best_s_per_lid_end_lg = {}
        for lid, endtag, lg, s, dj, dn, cp_id, j in cand:
            k = (lid, endtag, lg)
            prev = best_s_per_lid_end_lg.get(k)
            if prev is None or s < prev:
                best_s_per_lid_end_lg[k] = s

        # per (lid,endtag): best lane-group min_s
        best_min_s_per_lid_end = defaultdict(lambda: float("inf"))
        for (lid, endtag, lg), smin in best_s_per_lid_end_lg.items():
            kk = (lid, endtag)
            if smin < best_min_s_per_lid_end[kk]:
                best_min_s_per_lid_end[kk] = smin

        # decide which lane-groups to keep on each (lid,endtag)
        keep_lg = set()
        for (lid, endtag, lg), smin in best_s_per_lid_end_lg.items():
            sbest = best_min_s_per_lid_end[(lid, endtag)]
            if smin <= sbest + keep_tol_s:
                keep_lg.add((lid, endtag, lg))

        # 2) Now append ONLY CPs whose (lid,endtag,lg) survived, but still keep multiple CPs
        #    within that lane-group (parallel lanes).
        accepted_cp_ids = set()
        for lid, endtag, lg, s, dj, dn, cp_id, j in cand:
            if (lid, endtag, lg) not in keep_lg:
                continue
            accepted_cp_ids.add(int(cp_id))

            # append final row
            append(dict(
                road_node_id=int(rid_i),
                road_x=float(rx),
                road_y=float(ry),
                link_id=int(lid),
                link_end=str(endtag),
                lane_node_id=int(lane_ids[j]),
                lane_x=float(lane_X[j]),
                lane_y=float(lane_Y[j]),
                d_node=float(dn),
                d_seg=float(dj),
                s_on_poly=float(s),
                lane_groups=get_groups[j],
                lane_lnos=get_lnos[j],
                lane_end_counts=get_counts[j],
                incident_edges=get_incedg[j],
                cp_type="junction"
            ))
                    
            # Debug
            # recover coordinates here
            if debug is True:
                xcp = float(lane_X[j])
                ycp = float(lane_Y[j])
                cps_xy_this_rid.append((xcp, ycp))

        # Debug plot per rid
        fig = ax = None
        if debug:
            fig, ax = plt.subplots(figsize=(8,8))
            plot_debug_for_rid(
                ax=ax,
                rid_i=rid_i, rx=float(rx), ry=float(ry),
                end_windows=end_windows,
                c_idx=c_idx,
                lane_X=lane_X, lane_Y=lane_Y, lane_ids=lane_ids,
                lane_groups_str=lane_groups_str,
                r_node=r_node, r_seg=r_seg, endpoint_cap=endpoint_cap,
                cand=cand,
                point_polyline_metrics=_point_polyline_metrics,
                accepted_cp_ids=accepted_cp_ids,   # NEW
                accepted_keys=keep_lg,             # NEW (optional)
            )

            if cps_xy_this_rid:
                Xcp = [p[0] for p in cps_xy_this_rid]
                Ycp = [p[1] for p in cps_xy_this_rid]

                # plot CPs
                ax.scatter(Xcp, Ycp, s=30, marker='o', linewidths=1.0,
                        facecolors='none', edgecolors='tab:blue', zorder=8)

                # label them (optional but very useful)
                # If you also want IDs, store (x,y,cp_id) instead of (x,y) above.
                # Example: cps_xy_this_rid.append((xcp, ycp, cp_id))
                # then:
                # for x, y, cid in cps_xy_this_rid:
                #     ax.text(x, y, str(cid), fontsize=7, color='tab:blue', zorder=9)

                # highlight the road node center
                ax.scatter([float(rx)], [float(ry)], s=80, marker='x',
                        color='tab:orange', zorder=9)
                ax.text(float(rx), float(ry), f"rid={int(rid)}", fontsize=8,
                        color='tab:orange', zorder=10)
            else:
                # still mark the road node so you can see "no CP found" cases
                ax.scatter([float(rx)], [float(ry)], s=60, marker='x',
                        color='tab:gray', zorder=6)  
        

    # build C
    if not rows:
        C = pd.DataFrame(columns=[
            'road_node_id','road_x','road_y','link_id','link_end',
            'lane_node_id','lane_x','lane_y','d_node','d_seg',
            'lane_groups','lane_lnos','lane_end_counts','incident_edges'
        ])
    else:
        C = pd.DataFrame(rows)
        C.sort_values(['road_node_id','link_id','link_end','d_seg','d_node'], inplace=True)

    # mu, mv, _ = count_missing_link_ends(C, links_df)
    # print(f"[raw BEFORE fill] rows={len(C)} missing_u={mu} missing_v={mv}")


    # C, lane_node_xy = force_cover_all_link_ends_by_synthetic_cp(
    #     C=C,
    #     links_in_scope=links_df,
    #     nodes_df=nodes_df,
    #     Nodes=Nodes
    # )

    # right after fill
    # eps = 1e-6
    # seed_overlap = (np.abs(C["lane_x"] - C["road_x"]) <= eps) & (np.abs(C["lane_y"] - C["road_y"]) <= eps)

    # print("[AFTER fill] seed-overlap rows:", int(seed_overlap.sum()))
    # print("unique road nodes affected:", C.loc[seed_overlap, "road_node_id"].nunique())
    # print("is_synth counts (if exists):")
    # if "is_synth" in C.columns:
    #     print(C.loc[seed_overlap, "is_synth"].value_counts(dropna=False))

    # # show a few
    # print(C.loc[seed_overlap, ["road_node_id","link_id","link_end","lane_node_id","d_node","d_seg"]].head(20))


    # mu, mv, miss = count_missing_link_ends(C, links_df)
    # print(f"[raw AFTER fill]  rows={len(C)} missing_u={mu} missing_v={mv}")
    # print("Missing ends:", list(miss)[:10])

    # for lid,endtag in list(miss):
    #     rec = links_df[links_df['link_id'].astype(str)==str(lid)].iloc[0]
    #     print("----", lid, endtag)
    #     print(rec[['u','v','u_x','u_y','v_x','v_y','sx0','sy0','sx1','sy1','ex0','ey0','ex1','ey1']])

    C = sector_cull_connection_points(
            C,
            theta_min_deg=0.0,
            theta_max_deg=75.0,
            scope="node",
            group_by_lane=False,
            tol_node=2.0,
    )
    # mu, mv, _ = count_missing_link_ends(C, links_df)
    # print(f"[raw AFTER sectur_cull]  rows={len(C)} missing_u={mu} missing_v={mv}")
    #C = keep_nearest_cp_per_link_end(C)

    # mu, mv, _ = count_missing_link_ends(C, links_df)
    # print(f"[raw AFTER keep_nearest]  rows={len(C)} missing_u={mu} missing_v={mv}")
    # C = drop_if_no_peer_sharing_edge(
    #     C,
    #     cp_id_col="lane_node_id",
    #     protect_link_ends=True,   # default True in my patch
    #     link_id_col="link_id",
    #     link_end_col="link_end",
    #     link_dist_col="d_seg",
    #     node_dist_col="d_node",
    # )

    # mu, mv, _ = count_missing_link_ends(C, links_df)
    # print(f"[raw AFTER drop_if]  rows={len(C)} missing_u={mu} missing_v={mv}")
    # then proceed with your current pruning:
    """
    C_clean = drop_farther_directional_connected_no_edges(
        C,
        by_end=True,
        tol_node=0.05,
        angle_margin_deg=45.0,
        road_nodes_df=None,
        tol_other=0.05
    )
    """


    return C

# --- debug plot for ONE road node rid (call inside the rid-loop, after cand is built) ---
# Requires: matplotlib axes `ax`, numpy as np, math, and `end_windows`, `c_idx`, lane_X/lane_Y/lane_ids,
#          lane_groups_str (list[str] per lane node), r_node, r_seg, endpoint_cap
#          and optionally `cand` list from your code: (lid,endtag,lg,s,dj,dn,cp_id,j)

from collections import defaultdict
import numpy as np
import math

def _hsv_color(k: int, n: int):
    # deterministic distinct-ish colors without extra deps
    import colorsys
    h = (k % max(n, 1)) / max(n, 1)
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
    return (r, g, b)

def plot_debug_for_rid(
    ax,
    rid_i, rx, ry,
    end_windows,
    c_idx,
    lane_X, lane_Y, lane_ids,
    lane_groups_str,
    r_node=25.0, r_seg=5.0, endpoint_cap=0.75,
    cand=None,
    point_polyline_metrics=None,
    highlight_lids=None,
    title_prefix="RID debug",
    accepted_cp_ids=None,      # NEW
    accepted_keys=None,        # NEW (set of (lid,endtag,lg))
):
    """
    Plots raw candidate CPs around one road node rid:
      - RN, r_node circle
      - all end_windows polylines (and their tip points)
      - CPs in c_idx, colored by lane-group
      - if `cand` provided: mark accepted CPs per (lid,endtag,lg) + annotate s,dj,dn
      - optional: projection segments (CP -> closest point on polyline) if point_polyline_metrics is provided
    """

    # --------------------------
    # 0) backdrop: RN and radius
    # --------------------------
    ax.scatter([rx], [ry], s=180, marker="x", linewidths=2, color="k", zorder=50)
    ax.text(rx, ry, f" RN {rid_i}", fontsize=9, color="k", zorder=51)

    # circles: r_node and endpoint_cap
    th = np.linspace(0, 2*np.pi, 200)
    ax.plot(rx + float(r_node)*np.cos(th), ry + float(r_node)*np.sin(th),
            linestyle="--", linewidth=1, color="k", alpha=0.35, zorder=1)
    if endpoint_cap and float(endpoint_cap) > 0:
        ax.plot(rx + float(endpoint_cap)*np.cos(th), ry + float(endpoint_cap)*np.sin(th),
                linestyle=":", linewidth=1, color="k", alpha=0.25, zorder=1)

    # --------------------------
    # 1) plot all end windows
    # --------------------------
    # group by (lid,endtag)
    ew_by = defaultdict(list)
    for ew in (end_windows or []):
        lid = int(ew.get("lid"))
        endtag = str(ew.get("endtag"))
        ew_by[(lid, endtag)].append(ew)

    # draw all, highlight optionally
    for k, ews in ew_by.items():
        lid, endtag = k
        is_hl = (highlight_lids is None) or (lid in set(highlight_lids))
        lw = 2.0 if is_hl else 1.0
        a  = 0.8 if is_hl else 0.25
        col = "tab:red" if is_hl else "0.6"
        for ew in ews:
            Xw = np.asarray(ew.get("Xw", []), float)
            Yw = np.asarray(ew.get("Yw", []), float)
            if Xw.size < 2:
                continue
            ax.plot(Xw, Yw, color=col, linewidth=lw, alpha=a, zorder=5)
            # tip point (the touched road-end)
            x0 = float(ew.get("x0", Xw[0]))
            y0 = float(ew.get("y0", Yw[0]))
            ax.scatter([x0], [y0], s=40, marker="s", color=col, alpha=a, zorder=6)
            ax.text(x0, y0, f"{lid}{endtag}", fontsize=7, color=col, alpha=a, zorder=6)

    # --------------------------
    # 2) CPs in c_idx, colored by lane-group (first group as label)
    # --------------------------
    # map lane-group strings to colors
    # if a CP has multiple lane-groups, use the first one for color (still show all in label)
    lg_keys = []
    for j in c_idx:
        lgs = lane_groups_str[j] if lane_groups_str is not None else []
        lg_keys.append(lgs[0] if (isinstance(lgs, (list, tuple)) and len(lgs) > 0) else "NO_LG")
    uniq_lg = sorted(set(lg_keys))
    lg2c = {g: _hsv_color(i, len(uniq_lg)) for i, g in enumerate(uniq_lg)}

    # all CP points (raw near-node)
    for j in c_idx:
        xcp = float(lane_X[j]); ycp = float(lane_Y[j])
        lgs = lane_groups_str[j] if lane_groups_str is not None else []
        g0 = lgs[0] if (isinstance(lgs, (list, tuple)) and len(lgs) > 0) else "NO_LG"
        col = lg2c.get(g0, (0.2, 0.2, 0.2))
        ax.scatter([xcp], [ycp], s=55, marker="o", facecolors="none", edgecolors=[col], linewidths=1.5, zorder=20)
        ax.text(xcp, ycp, f"{int(lane_ids[j])}", fontsize=7, color=col, zorder=21)

    # --------------------------
    # 3) If `cand` provided: show accepted/rejected per link-end choice
    # --------------------------
    # cand rows are: (lid,endtag,lg,s,dj,dn,cp_id,j)
    if cand is not None and len(cand) > 0:
        # group cand by (lid,endtag,lg)
        by_key = defaultdict(list)
        for (lid, endtag, lg, s, dj, dn, cp_id, j) in cand:
            by_key[(int(lid), str(endtag), str(lg))].append((float(s), float(dj), float(dn), int(cp_id), int(j)))

        # For each key, mark all its CPs; annotate the best one (min s then dj then dn)
        for ki, (key, lst) in enumerate(by_key.items()):
            lid, endtag, lg = key
            # order: smallest s first
            lst_sorted = sorted(lst, key=lambda t: (t[0], t[1], t[2]))
            # choose color by lg
            col = lg2c.get(lg, _hsv_color(ki, len(by_key)))
            # label anchor at best CP
            s0, dj0, dn0, cp0, j0 = lst_sorted[0]
            x0, y0 = float(lane_X[j0]), float(lane_Y[j0])

            # draw a small label for the group-key at its best CP
            ax.text(x0, y0,
                    f"\n(lid={lid}{endtag}, lg={lg})\ns={s0:.2f} dj={dj0:.2f} dn={dn0:.2f}",
                    fontsize=7, color=col, zorder=40)

            # draw all CPs in that key as filled dots
            for (s, dj, dn, cp_id, j) in lst_sorted:
                xcp, ycp = float(lane_X[j]), float(lane_Y[j])

                # TRUE acceptance comes from the caller (post-pruning), not from dj
                acc_cp = (accepted_cp_ids is not None) and (int(cp_id) in set(accepted_cp_ids))
                acc_key = (accepted_keys is not None) and ((lid, endtag, lg) in set(accepted_keys))

                # Coloring:
                # - accepted CP AND accepted bucket => green
                # - accepted bucket but CP not accepted (e.g., removed by later CP-level filter) => orange
                # - not accepted => red
                if acc_cp and (accepted_keys is None or acc_key):
                    mcol = "tab:green"
                    z = 60
                elif acc_key and not acc_cp:
                    mcol = "tab:orange"
                    z = 55
                else:
                    mcol = "tab:red"
                    z = 50

                ax.scatter([xcp], [ycp], s=38, marker="o", color=mcol, alpha=0.9, zorder=z)

                # optional tiny text for debugging
                # ax.text(xcp, ycp, f"{cp_id}", fontsize=6, color=mcol, zorder=z+1)

                # connect CPs sharing same lane-group (optional thin dashed)
                # (only within this (lid,endtag,lg) bucket)
            if len(lst_sorted) >= 2:
                pts = np.array([[float(lane_X[j]), float(lane_Y[j])] for (_,_,_,_,j) in lst_sorted], float)
                ax.plot(pts[:,0], pts[:,1], linestyle="--", linewidth=1, color=col, alpha=0.5, zorder=30)

    # --------------------------
    # 4) Optional: plot projection segments CP -> closest point on each end window
    # --------------------------
    if point_polyline_metrics is not None:
        for j in c_idx:
            xcp = float(lane_X[j]); ycp = float(lane_Y[j])
            # only for highlighted windows if provided
            for ew in (end_windows or []):
                lid = int(ew.get("lid"))
                if (highlight_lids is not None) and (lid not in set(highlight_lids)):
                    continue
                Xw = np.asarray(ew.get("Xw", []), float)
                Yw = np.asarray(ew.get("Yw", []), float)
                if Xw.size < 2:
                    continue
                seg_idx, t, qx, qy, dj, s_abs = point_polyline_metrics(xcp, ycp, Xw, Yw)
                # draw thin line; green if within r_seg else light gray
                ok = (float(dj) <= float(r_seg))
                ax.plot([xcp, float(qx)], [ycp, float(qy)],
                        linewidth=0.8,
                        color=("tab:green" if ok else "0.75"),
                        alpha=(0.7 if ok else 0.35),
                        zorder=10)
                # mark projection point
                ax.scatter([float(qx)], [float(qy)], s=10, marker=".", color=("tab:green" if ok else "0.6"), zorder=11)

    # --------------------------
    # 5) overlay accepted CPs (if provided)
    # --------------------------
    if accepted_cp_ids is not None:
        acc = set(int(x) for x in accepted_cp_ids)
        for j in c_idx:
            cid = int(lane_ids[j])
            if cid not in acc:
                continue
            xcp = float(lane_X[j]); ycp = float(lane_Y[j])
            ax.scatter([xcp], [ycp], s=90, marker="o",
                    facecolors="none", edgecolors="tab:blue",
                    linewidths=2.0, zorder=90)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{title_prefix}: rid={rid_i}  (#c_idx={len(c_idx)}, #end_windows={len(end_windows)})")


from typing import Dict, Set, Any, Tuple

def build_lg2links_and_link2lgs(road: Dict[str, Any]) -> Tuple[Dict[str, Set[int]], Dict[int, Set[str]]]:
    """
    road["link_lane_group_refs"] : { link_id(int): [ {lane_group_ref:{lane_group_id:...}, ...}, ... ] }

    Returns:
      lg2links[g]  = set of link_ids that reference lane group g
      link2lgs[lid]= set of lane_group_ids referenced by link lid
    """
    llgr = road.get("link_lane_group_refs", {}) or {}

    lg2links: Dict[str, Set[int]] = {}
    link2lgs: Dict[int, Set[str]] = {}

    for lid_raw, rows in llgr.items():
        if lid_raw is None:
            continue
        try:
            lid = int(lid_raw)
        except Exception:
            continue

        s: Set[str] = set()
        for r in (rows or []):
            lgref = (r or {}).get("lane_group_ref") or {}
            gid = lgref.get("lane_group_id")
            if gid is None:
                continue
            gid = str(gid)
            s.add(gid)
            lg2links.setdefault(gid, set()).add(lid)

        link2lgs[lid] = s

    return lg2links, link2lgs

from collections import defaultdict
import pandas as pd
from typing import Any, Dict, Tuple, Set

def classify_road_nodes_by_laned_link_ends(
    nodes_df: pd.DataFrame,
    links_df: pd.DataFrame,
    road: Dict[str, Any],
):
    """
    Classify road nodes using *laned link ends* (not links) AND build traversal incidence
    in the SAME format as build_allowed_incidence():

        inc_allowed[node] = [(nbr, edge_idx, link_id, dir), ...]
        dir = +1 means traversing along df (u->v), dir = -1 is reverse (v->u)
        edge_idx = enumeration index stable w.r.t. links_df.iterrows() ordering.

    Returns:
        junction_candidates : set[int]
        chain_nodes         : set[int]
        dead_end_nodes      : set[int]
        stats               : dict[int -> dict]
        inc_allowed          : dict[int -> list[(int,int,int,int)]]
    """

    def _to_int(x):
        try:
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return None
            # strip quotes if it's a string
            if isinstance(x, str):
                x = x.strip().strip("'").strip('"')
            return int(x)
        except Exception:
            return None

    # ---- allowed links: MUST be int set, matches canonical build_allowed_incidence ----
    lane_links = set()
    if road is not None:
        lane_links = set((road.get("link_lane_group_refs", {}) or {}).keys())
    allowed_links = set()
    for k in lane_links:
        ki = _to_int(k)
        if ki is not None:
            allowed_links.add(ki)

    # ----------------------------------------------------------------------
    # (A) Build inc_allowed (canonical semantics embedded)
    # ----------------------------------------------------------------------
    inc_allowed = defaultdict(list)

    for edge_idx, (_, row) in enumerate(links_df.iterrows()):
        lid = _to_int(row.get("link_id"))
        if lid is None:
            continue
        if allowed_links is not None and lid not in allowed_links:
            continue

        u = _to_int(row.get("u"))
        v = _to_int(row.get("v"))
        if u is None or v is None:
            continue

        inc_allowed[u].append((v, edge_idx, lid, +1))
        inc_allowed[v].append((u, edge_idx, lid, -1))

    # ----------------------------------------------------------------------
    # (B) Build link-END incidence table for counting ends (your classification)
    # ----------------------------------------------------------------------
    # NOTE: classification uses "laned link ends", so we count BOTH u-end and v-end,
    # and we keep duplicates if the same link touches the node twice via two ends.
    inc = pd.concat([
        links_df[["u", "link_id"]].rename(columns={"u": "node_id"}).assign(end="u"),
        links_df[["v", "link_id"]].rename(columns={"v": "node_id"}).assign(end="v"),
    ], ignore_index=True)

    inc["node_id"] = inc["node_id"].apply(_to_int)
    inc["link_id_i"] = inc["link_id"].apply(_to_int)

    # keep only laned link-ends
    inc = inc.dropna(subset=["node_id", "link_id_i"])
    inc = inc[inc["link_id_i"].isin(allowed_links)]

    by_node = inc.groupby("node_id")

    junction_candidates: Set[int] = set()
    chain_nodes: Set[int] = set()
    dead_end_nodes: Set[int] = set()
    stats: Dict[int, Dict[str, Any]] = {}

    # only consider nodes that exist in nodes_df (your previous behavior)
    for nid_raw in nodes_df["node_id"].values:
        nid = _to_int(nid_raw)
        if nid is None:
            continue

        if nid in by_node.groups:
            G = by_node.get_group(nid)
            n_ends = int(len(G))                         # laned link-ends count
            link_ids = set(int(x) for x in G["link_id_i"].tolist())
        else:
            n_ends = 0
            link_ids = set()

        stats[nid] = dict(
            n_laned_link_ends=n_ends,
            n_laned_links=len(link_ids),
        )

        if n_ends >= 3:
            junction_candidates.add(nid)
        elif n_ends == 2:
            # chain only if they belong to DIFFERENT links
            if len(link_ids) == 2:
                chain_nodes.add(nid)
        elif n_ends == 1:
            dead_end_nodes.add(nid)

    return junction_candidates, chain_nodes, dead_end_nodes, dict(inc_allowed), stats


def count_missing_link_ends(C: pd.DataFrame, links_df: pd.DataFrame):
    # Each road-link record requires two ends: (link_id,'u') and (link_id,'v')
    required = set()
    for lid, u, v in links_df[['link_id','u','v']].itertuples(index=False, name=None):
        required.add((str(lid), 'u'))
        required.add((str(lid), 'v'))

    covered = set()
    if C is not None and not C.empty:
        covered = set(zip(C['link_id'].astype(str), C['link_end'].astype(str)))

    missing = required - covered
    missing_u = sum(1 for (_, end) in missing if end == 'u')
    missing_v = sum(1 for (_, end) in missing if end == 'v')
    return missing_u, missing_v, missing

def _end_tip_xy(rec, endtag: str):
    """Return (x,y) at the touched end, using best available fields."""
    endtag = str(endtag).lower()
    if endtag == "u":
        if hasattr(rec, "u_x") and hasattr(rec, "u_y") and rec.u_x is not None and rec.u_y is not None and len(rec.u_x) > 0:
            return float(rec.u_x[0]), float(rec.u_y[0])
        # fallback to local segment
        if hasattr(rec, "sx0") and hasattr(rec, "sy0") and pd.notna(rec.sx0) and pd.notna(rec.sy0):
            return float(rec.sx0), float(rec.sy0)
    else:  # "v"
        if hasattr(rec, "v_x") and hasattr(rec, "v_y") and rec.v_x is not None and rec.v_y is not None and len(rec.v_x) > 0:
            return float(rec.v_x[0]), float(rec.v_y[0])
        if hasattr(rec, "ex0") and hasattr(rec, "ey0") and pd.notna(rec.ex0) and pd.notna(rec.ey0):
            return float(rec.ex0), float(rec.ey0)

    return None  # should be rare


def force_cover_all_link_ends_by_synthetic_cp(C: pd.DataFrame,
                                             links_in_scope: pd.DataFrame,
                                             nodes_df: pd.DataFrame,
                                             Nodes = None,
                                             lane_node_xy = None, 
                                             cp_id_col="lane_node_id"):
    if lane_node_xy is None:
        lane_node_xy = {}
    # if Nodes provided, seed from Nodes
    if Nodes is not None and len(lane_node_xy) == 0:
        for nid, x, y in Nodes[['node_id','X','Y']].itertuples(index=False, name=None):
            lane_node_xy[int(nid)] = (float(x), float(y))
    # current CP ids (existing lane nodes + already added synthetic)
    used_ids = set(int(x) for x in pd.unique(C[cp_id_col].dropna()).tolist())
    if lane_node_xy:
        for k in lane_node_xy.keys():
            try: used_ids.add(int(k))
            except: pass

    next_id = (max(used_ids) + 1) if used_ids else 0

    # compute missing ends from C (your existing logic)
    missing_u, missing_v, missing = count_missing_link_ends(C, links_in_scope)  # returns list of (link_id, 'u'/'v')

    new_rows = []

    # index links by link_id for fast lookup
    tmp = links_in_scope.copy()
    tmp["link_id_str"] = tmp["link_id"].astype(str)

    for lid, endtag in missing:
        recs = tmp[tmp["link_id_str"] == str(lid)]
        if recs.empty:
            continue
        rec = recs.iloc[0]  # first match (ok if link_id unique in scope)

        # We need the actual namedtuple-style record to reuse your hasattr(rec,'v_x') etc.
        # easiest: convert row to simple object via itertuples
        rec_obj = next(recs.itertuples(index=False))

        tip = _end_tip_xy(rec_obj, endtag)
        if tip is None:
            # last-resort: use road node coordinate at that end
            node_id = int(rec_obj.v) if str(endtag).lower() == "v" else int(rec_obj.u)
            rn = nodes_df[nodes_df["node_id"] == node_id]
            if rn.empty:
                continue
            tip = (float(rn["x"].iloc[0]), float(rn["y"].iloc[0]))

        xcp, ycp = tip

        # allocate synthetic CP id
        while next_id in used_ids:
            next_id += 1
        cp_id = next_id
        used_ids.add(cp_id)
        next_id += 1

        # add to lane_node_xy so exporter writes it
        lane_node_xy[int(cp_id)] = (float(xcp), float(ycp))

        # create the new C row that covers exactly this (link_id, end)

        new_rows.append(dict(
            road_node_id = int(rec_obj.v) if str(endtag).lower() == "v" else int(rec_obj.u),
            road_x = float(xcp),   # optional; if you prefer, set from nodes_df road node instead
            road_y = float(ycp),
            link_id = str(lid),
            link_end = str(endtag).lower(),
            lane_node_id = int(cp_id),
            lane_x = float(xcp),
            lane_y = float(ycp),
            d_node = 0.0,
            d_seg  = 0.0,
            s_on_poly = 0.0,
            lane_groups = [],
            lane_lnos = [],
            lane_end_counts = {"start":0, "end":0},
            incident_edges = [],
        ))

    if new_rows:
        C = pd.concat([C, pd.DataFrame(new_rows)], ignore_index=True)

    return C, lane_node_xy


import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
import numpy as np

def plot_all_seed_matches(
    ax,
    C,
    bbox_latlon=None,
    bbox_enu=None,
    origin=None,
    color_map='tab10',
    seed_size=7,
    match_size=30,
    linewidth=1.8,
    radius=None,                 # <--- new: circle radius (in ENU meters). If None, no circle.
    circle_linestyle='--',
    circle_alpha=0.8
):
    """
    Plot connection-point matches per road node, optionally with a dashed radius circle
    around each road node (ENU units).

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
    C : pd.DataFrame
        Must contain columns: ['road_node_id','lane_x','lane_y','road_x','road_y'].
    bbox_latlon : (minlon, minlat, maxlon, maxlat) or None
    bbox_enu : (xmin, ymin, xmax, ymax) or None
    origin : dict or tuple, required if bbox_latlon is given
    color_map : str
    seed_size : float
    match_size : float
    linewidth : float
    radius : float or None
        ENU radius to draw around each road node (same radius for all). If None, circles are skipped.
    circle_linestyle : str
    circle_alpha : float
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    import matplotlib.cm as cm
    import numpy as np

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    if C is None or C.empty:
        ax.text(0.5, 0.5, 'No matches', transform=ax.transAxes, ha='center', va='center')
        return

    # colors per intersection
    node_ids = C['road_node_id'].unique()
    cmap = cm.get_cmap(color_map, len(node_ids))
    colors = {rid: cmap(i) for i, rid in enumerate(node_ids)}

    drew_circle = False

    # plot each intersection
    for rid in node_ids:
        sub = C[C['road_node_id'] == rid]
        lx = pd.to_numeric(sub['lane_x'], errors='coerce').to_numpy(float)
        ly = pd.to_numeric(sub['lane_y'], errors='coerce').to_numpy(float)
        good = np.isfinite(lx) & np.isfinite(ly)

        if not good.any():
            rx, ry = float(sub['road_x'].iloc[0]), float(sub['road_y'].iloc[0])
            print(f"[NO_CP_DRAW] rid={rid} rows={len(sub)} road=({rx:.2f},{ry:.2f}) "
                f"lane_x/y all non-finite. Examples lane_x={sub['lane_x'].head(3).tolist()} "
                f"lane_y={sub['lane_y'].head(3).tolist()}")

        col = colors[rid]

        # matches (connection points)
        ax.scatter(sub['lane_x'], sub['lane_y'],
                   s=match_size, color=col, edgecolors='none', zorder=3)

        # seed road node (single hollow marker)
        rx, ry = float(sub['road_x'].iloc[0]), float(sub['road_y'].iloc[0])
        ax.scatter([rx], [ry],
                   facecolors='none', edgecolors=col,
                   s=seed_size**2, linewidths=linewidth, zorder=4)

        # optional dashed circle around the road node
        if radius is not None and np.isfinite(radius) and radius > 0:
            circ = Circle((rx, ry), radius,
                          fill=False, edgecolor=col, linewidth=linewidth,
                          linestyle=circle_linestyle, alpha=circle_alpha, zorder=2.6)
            ax.add_patch(circ)
            drew_circle = True

    # bounding box (fixed)
    if bbox_enu is None and bbox_latlon is not None:
        assert origin is not None, "Provide origin when using bbox_latlon."
        bbox_enu = _bbox_latlon_to_enu_rect(bbox_latlon, origin)

    if bbox_enu is not None:
        xmin, ymin, xmax, ymax = bbox_enu
        rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                         fill=False, edgecolor='k', linewidth=linewidth,
                         linestyle='--', zorder=2)
        ax.add_patch(rect)

    ax.set_aspect('equal')
    ax.grid(True, linestyle=':')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')

    # Legend proxies (avoid duplicates from scatter/patch artists)
    import matplotlib.lines as mlines
    seed_proxy = mlines.Line2D([], [], marker='o', color='k', markerfacecolor='none',
                               markersize=6, linestyle='None', label='Road seed nodes')
    cp_proxy = mlines.Line2D([], [], marker='o', color='w', markerfacecolor='k',
                             markersize=6, linestyle='None', label='Connection points')
    handles = [cp_proxy, seed_proxy]
    labels = ['Connection points', 'Road seed nodes']

    if drew_circle:
        circle_proxy = mlines.Line2D([], [], color='k', linestyle=circle_linestyle,
                                     linewidth=linewidth, label='Search radius')
        handles.append(circle_proxy)
        labels.append('Search radius')

    ax.legend(handles, labels, loc='best', fontsize=8)
    plt.tight_layout()


import pandas as pd
import numpy as np

def normalize_per_lane(C: pd.DataFrame, lane_col: str = 'lane_lnos') -> pd.DataFrame:
    X = C.copy()
    # normalize to list
    def _to_list(v):
        if isinstance(v, (list, tuple, set, np.ndarray)):
            return list(v)
        if pd.isna(v):
            return []
        return [v]
    X[lane_col] = X[lane_col].apply(_to_list)

    # explode to one lane per row (keeps rows with 0 lanes too; drop if you want)
    X = X.explode(lane_col, ignore_index=True)
    X['lane_no'] = X[lane_col].astype('Int64')  # nullable int
    return X

import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
from collections import defaultdict
from math import cos, radians
import numpy as np
import pandas as pd
from math import cos, radians

import numpy as np
import pandas as pd
from math import cos, radians
import numpy as np
import pandas as pd
from math import cos, radians
from collections import Counter

def drop_farther_directional_connected_no_edges(
    C: pd.DataFrame,
    *,
    by_end: bool = True,
    tol_node: float = 0.05,
    angle_margin_deg: float = 90.0,
    road_nodes_df: pd.DataFrame | None = None,   # optional ['node_id','x','y']
    tol_other: float = 0.05                      # nearest-other-node tolerance
) -> pd.DataFrame:
    """
    Prune candidates using:
      (1) Nearest-road-node rule (optional if road_nodes_df provided)
      (2) Directional + adjacency rule within each (road_node_id, link_id[, link_end]):
          Keep nearer P1; drop farther P2 if:
            A) (P1-R) and (P2-R) within angle_margin_deg  (cosine check)
            B) incident_edges(P1) ∩ incident_edges(P2) ≠ ∅
            C) d_node(P2) ≥ d_node(P1) + tol_node
      (3) FINAL FILTER (per group): drop points that share **no edge** with any other point in the group.

    Requires columns in C:
      ['road_node_id','road_x','road_y','link_id','link_end',
       'lane_node_id','lane_x','lane_y','d_node','incident_edges', ...]
    """
    if C is None or C.empty:
        return C

    out = C.copy()

    # ---------- (1) Nearest-road-node filter (optional) ----------
    if road_nodes_df is not None and not road_nodes_df.empty:
        U = out[['lane_node_id','lane_x','lane_y']].drop_duplicates().reset_index(drop=True)
        Rxy = road_nodes_df[['x','y']].to_numpy(dtype=float)
        Rid = road_nodes_df['node_id'].to_numpy()

        nearest_id = np.empty(len(U), dtype=Rid.dtype)
        nearest_dist = np.empty(len(U), dtype=float)
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(Rxy)
            dists, idxs = tree.query(U[['lane_x','lane_y']].to_numpy(dtype=float), k=1)
            nearest_id[:] = Rid[idxs]
            nearest_dist[:] = dists
        except Exception:
            P = U[['lane_x','lane_y']].to_numpy(dtype=float)
            min_d2 = np.full(len(P), np.inf, dtype=float)
            min_j  = np.zeros(len(P), dtype=int)
            B = 4096
            for i0 in range(0, len(P), B):
                Pi = P[i0:i0+B]
                d2 = ((Pi[:,None,:] - Rxy[None,:,:])**2).sum(axis=2)
                jmin = d2.argmin(axis=1)
                vmin = d2[np.arange(len(Pi)), jmin]
                sel = vmin < min_d2[i0:i0+B]
                min_d2[i0:i0+B][sel] = vmin[sel]
                min_j[i0:i0+B][sel]  = jmin[sel]
            nearest_id[:] = Rid[min_j]
            nearest_dist[:] = np.sqrt(min_d2)

        U['nearest_road_node_id'] = nearest_id
        U['nearest_road_dist']    = nearest_dist

        out = out.merge(U[['lane_node_id','nearest_road_node_id','nearest_road_dist']],
                        on='lane_node_id', how='left')

        keep_mask = (
            (out['road_node_id'] == out['nearest_road_node_id']) |
            (out['d_node'] <= out['nearest_road_dist'] + float(tol_other))
        )
        out = out.loc[keep_mask].copy()
        # Optional: drop helper columns
        # out.drop(columns=['nearest_road_node_id','nearest_road_dist'], inplace=True, errors='ignore')

        if out.empty:
            return out

    # ---------- (2) Directional + adjacency pruning ----------
    if 'incident_edges' not in out.columns:
        return out

    ct = cos(radians(max(0.0, min(180.0, angle_margin_deg))))
    gcols = ['road_node_id','link_id'] + (['link_end'] if by_end and 'link_end' in out.columns else [])

    keep_indices = []
    for _, G in out.groupby(gcols, sort=False, dropna=False):
        if len(G) <= 1:
            keep_indices.extend(G.index.tolist())
            continue

        rx = G['road_x'].to_numpy(float)
        ry = G['road_y'].to_numpy(float)
        px = G['lane_x'].to_numpy(float)
        py = G['lane_y'].to_numpy(float)
        d  = G['d_node'].to_numpy(float)

        vx = px - rx
        vy = py - ry
        vn = np.hypot(vx, vy); vn[vn == 0.0] = 1e-12

        inc_sets = []
        for lst in G['incident_edges'].tolist():
            try: inc_sets.append(set(lst))
            except TypeError: inc_sets.append(set())

        order = np.argsort(d)
        dropped = np.zeros(len(G), dtype=bool)

        for a_pos, ia in enumerate(order):
            if dropped[ia]:
                continue
            keep_indices.append(G.index[ia])

            for ib in order[a_pos+1:]:
                if dropped[ib]:
                    continue

                # adjacency via shared edge id
                if not (inc_sets[ia] & inc_sets[ib]):
                    continue

                # angle check
                cosphi = (vx[ia]*vx[ib] + vy[ia]*vy[ib]) / (vn[ia]*vn[ib])
                if cosphi < ct:
                    continue

                # farther by tol?
                if d[ib] >= d[ia] + float(tol_node):
                    dropped[ib] = True

    out2 = out.loc[keep_indices].copy()

    # ---------- (3) FINAL FILTER: must share an edge with someone in the same group ----------
    final_keep = []

    for key, G in out2.groupby(gcols, sort=False, dropna=False):
        if len(G) <= 1:
            # Singletons cannot share an edge -> DROP
            continue

        inc_sets = []
        for lst in G['incident_edges'].tolist():
            try: inc_sets.append(set(lst))
            except TypeError: inc_sets.append(set())

        # Count how many points reference each edge in the group
        edge_counts = Counter(e for s in inc_sets for e in s)

        # Keep a row iff it has at least one incident edge shared by another row
        mask = []
        for s in inc_sets:
            keep = any(edge_counts[e] > 1 for e in s)
            mask.append(keep)

        if any(mask):
            final_keep.extend(G.loc[mask].index.tolist())
        # else: whole group drops (no shared edges)

    if not final_keep:
        return out2.iloc[0:0].copy()

    out3 = out2.loc[final_keep].copy()

    # Stable tidy sort
    sort_cols = gcols + (['d_node'] if 'd_node' in out3.columns else [])
    if 'd_seg' in out3.columns: sort_cols.append('d_seg')
    return out3.sort_values(sort_cols, kind='mergesort').reset_index(drop=True)


# usage:
# C_norm = normalize_per_lane(C)
# C_clean = drop_farther_on_same_link_per_lane(C_norm, by_end=True, tol_node=0.05)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def _bbox_latlon_to_enu_rect(bbox_latlon, origin):
    """[lat_min, lon_min, lat_max, lon_max] -> (xmin, ymin, xmax, ymax) in ENU."""
    lat_min, lon_min, lat_max, lon_max = map(float, bbox_latlon)
    lat_corners = [lat_min, lat_min, lat_max, lat_max]
    lon_corners = [lon_min, lon_max, lon_min, lon_max]
    XY = _ll_to_enu(lat_corners, lon_corners, origin)
    xs, ys = XY[:, 0], XY[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

def plot_ids_in_bbox(
    ax,
    nodes_df,              # columns: ['node_id','x','y'] in ENU
    links_df,              # columns: ['link_id','u','v', ...]
    bbox_latlon=None,      # [lat_min, lon_min, lat_max, lon_max]
    bbox_enu=None,         # (xmin, ymin, xmax, ymax)
    origin=None,
    node_color='k',
    link_color='crimson',
    node_fontsize=8,
    link_fontsize=8,
    draw_bbox=True,
    draw_node_points=True,
    draw_link_midpoints=False,  # set True to also dot the link midpoints
):
    """
    Annotate node_id at node coords, and link_id at midpoint between its end nodes,
    restricted to the bbox. Bbox may be given in lat/lon or ENU.
    """
    assert bbox_enu is not None or (bbox_latlon is not None and origin is not None), \
        "Provide bbox_enu or (bbox_latlon + origin)."
    if bbox_enu is None:
        bbox_enu = _bbox_latlon_to_enu_rect(bbox_latlon, origin)

    xmin, ymin, xmax, ymax = map(float, bbox_enu)
    w, h = (xmax - xmin), (ymax - ymin)
    if w <= 0 or h <= 0:
        return

    # small text offset relative to bbox size (prevents marker/text overlap)
    dx, dy = 0.006 * w, 0.006 * h

    # ---- filter nodes to bbox ----
    N_in = nodes_df[
        (nodes_df['x'] >= xmin) & (nodes_df['x'] <= xmax) &
        (nodes_df['y'] >= ymin) & (nodes_df['y'] <= ymax)
    ].copy()

    # ---- link midpoints from node coords ----
    # build a map for fast lookup
    xmap = dict(zip(nodes_df['node_id'].astype(int), nodes_df['x'].astype(float)))
    ymap = dict(zip(nodes_df['node_id'].astype(int), nodes_df['y'].astype(float)))

    mids = []
    for lid, u, v in links_df[['link_id','u','v']].itertuples(index=False, name=None):
        if (u in xmap) and (v in xmap):
            mx = 0.5 * (xmap[int(u)] + xmap[int(v)])
            my = 0.5 * (ymap[int(u)] + ymap[int(v)])
            if (xmin <= mx <= xmax) and (ymin <= my <= ymax):
                mids.append((str(lid), mx, my))
    L_mid = mids  # list of (link_id, mx, my)

    # ---- draw bbox ----
    if draw_bbox:
        rect = Rectangle((xmin, ymin), w, h, fill=False, edgecolor='k', linestyle='--', linewidth=1.5, zorder=2)
        ax.add_patch(rect)

    # ---- plot nodes + labels ----
    if not N_in.empty:
        if draw_node_points:
            ax.scatter(N_in['x'], N_in['y'], s=16, c=node_color, zorder=3)
        for nid, x, y in N_in[['node_id','x','y']].itertuples(index=False, name=None):
            ax.text(x + dx, y + dy, str(int(nid)), color=node_color, fontsize=node_fontsize,
                    va='bottom', ha='left', zorder=4)

    # ---- plot link midpoints + labels ----
    if L_mid:
        if draw_link_midpoints:
            ax.scatter([mx for _, mx, _ in L_mid], [my for _, _, my in L_mid],
                       s=16, c=link_color, marker='s', zorder=3)
        for lid, mx, my in L_mid:
            ax.text(mx + dx, my + dy, lid, color=link_color, fontsize=link_fontsize,
                    va='bottom', ha='left', zorder=4)

    # cosmetics
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':')

import numpy as np
import pandas as pd
from math import atan2, pi

# ---------- convex hull (monotone chain) ----------
def _convex_hull_indices_xy(xy: np.ndarray) -> np.ndarray:
    """Return indices of convex hull vertices (CCW), using Andrew’s monotone chain."""
    if xy.shape[0] <= 1:
        return np.arange(xy.shape[0], dtype=int)
    pts = xy.astype(float, copy=False)
    order = np.lexsort((pts[:,1], pts[:,0]))
    P = pts[order]
    idx = order

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower, li = [], []
    for i in range(len(P)):
        while len(lower) >= 2 and cross(P[lower[-2]], P[lower[-1]], P[i]) <= 0:
            lower.pop(); li.pop()
        lower.append(i); li.append(idx[i])

    upper, ui = [], []
    for i in range(len(P)-1, -1, -1):
        while len(upper) >= 2 and cross(P[upper[-2]], P[upper[-1]], P[i]) <= 0:
            upper.pop(); ui.pop()
        upper.append(i); ui.append(idx[i])

    # drop the duplicate first/last
    hull_idx = np.array(li[:-1] + ui[:-1], dtype=int)
    return np.unique(hull_idx, return_index=True)[0]  # stable unique

import numpy as np
import pandas as pd
from math import atan2, pi

import numpy as np
import pandas as pd

# --- convex hull that returns POSITIONS (0..m-1) for the given xy order ---
def _convex_hull_positions(xy: np.ndarray) -> np.ndarray:
    n = xy.shape[0]
    if n <= 1:
        return np.arange(n, dtype=int)

    pts = xy.astype(float, copy=False)
    order = np.lexsort((pts[:,1], pts[:,0]))
    P = pts[order]
    pos = order.copy()

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower, lower_pos = [], []
    for i in range(len(P)):
        while len(lower) >= 2 and cross(P[lower[-2]], P[lower[-1]], P[i]) <= 0:
            lower.pop(); lower_pos.pop()
        lower.append(i); lower_pos.append(pos[i])

    upper, upper_pos = [], []
    for i in range(len(P)-1, -1, -1):
        while len(upper) >= 2 and cross(P[upper[-2]], P[upper[-1]], P[i]) <= 0:
            upper.pop(); upper_pos.pop()
        upper.append(i); upper_pos.append(pos[i])

    hull_pos = np.array(lower_pos[:-1] + upper_pos[:-1], dtype=int)
    return np.unique(hull_pos)

# --- per-group filter: drop hull points that have NO shared edge with anyone else in the group ---
def prefer_interior_linked_only_group(G: pd.DataFrame) -> pd.DataFrame:
    """
    Keep all interior points.
    From hull points, drop only those that share NO incident edge with any other point in the group.
    If the group has no interior points (all on hull), keep everything.
    Requires column 'incident_edges' with list-like entries.
    """
    m = len(G)
    if m <= 1:
        return G

    # geometry arrays in row order
    px = G['lane_x'].to_numpy(float)
    py = G['lane_y'].to_numpy(float)
    xy = np.c_[px, py]

    hull_pos = _convex_hull_positions(xy)
    mask_hull = np.zeros(m, dtype=bool); mask_hull[hull_pos] = True
    mask_int  = ~mask_hull

    if not mask_int.any():
        # nothing interior to prefer against; keep all
        return G

    # Build incident-edge sets for all rows once
    sets_all = []
    for lst in G['incident_edges'].tolist():
        try: sets_all.append(set(lst))
        except TypeError: sets_all.append(set())

    keep = np.ones(m, dtype=bool)

    # For each hull row, check if it shares an edge with ANY other row in the group
    for h in np.flatnonzero(mask_hull):
        sh = sets_all[h]
        # union over everyone except self (early-exit if intersection found)
        linked = False
        if sh:
            for k in range(m):
                if k == h: 
                    continue
                if sh & sets_all[k]:
                    linked = True
                    break
        if not linked:
            keep[h] = False

    return G.iloc[keep].copy()

# --- wrapper over all groups ---
def prefer_interior_linked_only(C: pd.DataFrame, by_end: bool = True) -> pd.DataFrame:
    if C is None or C.empty:
        return C
    gcols = ['road_node_id','link_id'] + (['link_end'] if by_end and 'link_end' in C.columns else [])
    parts = []
    for _, G in C.groupby(gcols, sort=False, dropna=False):
        parts.append(prefer_interior_linked_only_group(G))
    out = pd.concat(parts, ignore_index=True) if parts else C.iloc[0:0].copy()

    # tidy, stable sort for readability
    sort_cols = gcols + (['d_node'] if 'd_node' in out.columns else [])
    if 'd_seg' in out.columns: sort_cols.append('d_seg')
    return out.sort_values(sort_cols, kind='mergesort').reset_index(drop=True)


def _angles_from(ox, oy, px, py):
    return np.arctan2(py - oy, px - ox)  # (-pi, pi]

def _ang_diff(a, b):
    d = (a - b + pi) % (2*pi) - pi
    return abs(d)

def prefer_interior_over_hull_group(G: pd.DataFrame,
                                    theta_deg: float = 25.0,
                                    tol_node: float = 0.05) -> pd.DataFrame:
    """
    Prefer interior points (strictly inside convex hull).
    Drop a hull point if:
      (a) it shares NO incident edge with ANY interior point, OR
      (b) it is angle-aligned (Δθ<=theta_deg) to an interior point AND is farther by tol_node.

    Works on one (road_node_id, link_id[, link_end]) group G.
    """
    m = len(G)
    if m <= 2:
        return G

    # geometry arrays in the *row order of G*
    rx = float(G['road_x'].iloc[0]); ry = float(G['road_y'].iloc[0])
    px = G['lane_x'].to_numpy(float)
    py = G['lane_y'].to_numpy(float)
    d  = G['d_node'].to_numpy(float)
    xy = np.c_[px, py]

    # hull positions (0..m-1)
    hull_pos = _convex_hull_positions(xy)
    mask_hull = np.zeros(m, dtype=bool); mask_hull[hull_pos] = True
    mask_int  = ~mask_hull

    if not mask_int.any():
        # everything is on the hull -> nothing to prefer
        return G

    # angles from road node
    ang = _angles_from(rx, ry, px, py)
    theta = np.deg2rad(theta_deg)

    # incident edge union for interior points
    int_sets = []
    for lst in G.loc[mask_int, 'incident_edges'].tolist():
        try: int_sets.append(set(lst))
        except TypeError: int_sets.append(set())
    int_edge_union = set().union(*int_sets) if int_sets else set()

    keep = np.ones(m, dtype=bool)
    int_pos = np.flatnonzero(mask_int)

    for h in np.flatnonzero(mask_hull):
        # (a) drop if no shared edge with any interior
        try: hset = set(G.iloc[h]['incident_edges'])
        except TypeError: hset = set()
        shares = bool(hset & int_edge_union)
        if not shares:
            keep[h] = False
            continue

        # (b) nearest interior by angle; drop hull if aligned and farther
        if int_pos.size:
            diffs = np.array([_ang_diff(ang[h], ang[j]) for j in int_pos])
            j = int_pos[int(diffs.argmin())]
            if diffs.min() <= theta and d[h] >= d[j] + float(tol_node):
                keep[h] = False

    G2 = G.iloc[keep].copy()
    return G2

def prefer_interior_over_hull(C: pd.DataFrame,
                              by_end: bool = True,
                              theta_deg: float = 25.0,
                              tol_node: float = 0.05) -> pd.DataFrame:
    if C is None or C.empty:
        return C
    gcols = ['road_node_id','link_id'] + (['link_end'] if by_end and 'link_end' in C.columns else [])
    parts = []
    for _, G in C.groupby(gcols, sort=False, dropna=False):
        parts.append(prefer_interior_over_hull_group(G, theta_deg=theta_deg, tol_node=tol_node))
    out = pd.concat(parts, ignore_index=True) if parts else C.iloc[0:0].copy()
    sort_cols = gcols + (['d_node'] if 'd_node' in out.columns else [])
    if 'd_seg' in out.columns: sort_cols.append('d_seg')
    return out.sort_values(sort_cols, kind='mergesort').reset_index(drop=True)

import numpy as np
import math

def _is_seq2(x):
    """Return True if x is a sequence with at least 2 finite numbers."""
    if x is None: return False
    # pandas namedtuple will give us lists or np arrays; allow tuples too
    try:
        return len(x) >= 2 and np.isfinite(np.asarray(x, dtype=float)).sum() >= 2
    except Exception:
        return False

def _as_float_np(x):
    return np.asarray(x, dtype=float)

def _poly_arclen(X, Y):
    d = np.hypot(np.diff(X), np.diff(Y))
    return np.concatenate(([0.0], np.cumsum(d)))

def _clip_polyline_by_length(X, Y, L_window):
    """Keep the first L_window meters starting at X[0],Y[0]; guarantee >=2 points."""
    if len(X) < 2:  # nothing to do
        return X, Y
    S = _poly_arclen(X, Y)
    Lw = min(float(L_window), float(S[-1]))
    k = int(np.searchsorted(S, Lw, side='right'))
    k = max(2, k)  # ensure at least 2 points
    return X[:k], Y[:k]

def _point_segment_metrics(px, py, x0, y0, x1, y1):
    vx, vy = (x1-x0), (y1-y0)
    wx, wy = (px-x0), (py-y0)
    seg2 = vx*vx + vy*vy
    if seg2 == 0.0:
        return 0, 0.0, x0, y0, math.hypot(wx, wy), 0.0
    t = (wx*vx + wy*vy) / seg2
    t_clamped = max(0.0, min(1.0, t))
    qx = x0 + t_clamped*vx
    qy = y0 + t_clamped*vy
    dist = math.hypot(px-qx, py-qy)
    return 0, t_clamped, qx, qy, dist, t

def _point_polyline_metrics(px, py, X, Y):
    """Closest point of P to polyline (X,Y). Returns (seg_idx, t, qx, qy, dist, s_abs)."""
    X = _as_float_np(X); Y = _as_float_np(Y)
    n = len(X)
    if n == 0:
        return 0, 0.0, float('nan'), float('nan'), float('inf'), float('nan')
    if n == 1:
        dist = math.hypot(px - X[0], py - Y[0])
        return 0, 0.0, X[0], Y[0], dist, 0.0
    S = _poly_arclen(X, Y)
    best = (0, 0.0, X[0], Y[0], float('inf'), 0.0)
    for i in range(n-1):
        _, t, qx, qy, dseg, _ = _point_segment_metrics(px, py, X[i], Y[i], X[i+1], Y[i+1])
        if dseg < best[4]:
            s_abs = S[i] + t * (S[i+1] - S[i])
            best = (i, t, qx, qy, dseg, s_abs)
    return best

import numpy as np
import pandas as pd
import numpy as np
import pandas as pd

def prefer_near_end_and_drop_farther_per_lane(
    C: pd.DataFrame,
    *,
    by_end: bool = True,
    lane_t_col: str = "t",           # normalized [0,1] along the lane
    lane_no_col: str = "lane_lnos",  # lane identifier (may be list/NaN)
    link_dist_col: str = "d_seg",    # distance-to-link metric
    link_tol: float = 0.50,          # "close to this link" threshold
    use_opposite_side_rule: bool = False,  # set True to require opposite sides to drop
    side_eps: float = 0.0,           # dot < -side_eps means opposite sides
    near_t: float = 0.20,            # keep near the correct end
    tol_node: float = 0.05           # tie tolerance on d_node
) -> pd.DataFrame:
    """
    Pipeline:
      1) Near-end filter (by t and link_end) — optional if 't' missing.
      2) Normalize lane ids (explode list-like; give NaNs row-unique placeholders).
      3) Define CLOSE-TO-LINK mask: C[link_dist_col] <= link_tol. Only these rows
         participate in 'closest d_node' thinning; non-close rows are left untouched.
      4) Per (road_node_id, link_id[, link_end], lane) group:
           - Among CLOSE rows: keep the minimal d_node (within tol_node).
             If use_opposite_side_rule=True, only drop a farther row if there exists
             a closer row on the OPPOSITE side of the road node
             (dot( (pi - r), (pj - r) ) < -side_eps).
           - NON-CLOSE rows are all kept.

    This prevents “orange points” (closer to other links) from being removed just
    because they were assigned to a link where they are not CLOSE.
    """
    if C is None or C.empty:
        return C

    has_t    = lane_t_col in C.columns
    has_end  = by_end and ('link_end' in C.columns)
    has_dseg = link_dist_col in C.columns

    # --- 1) Near-end filter (conservative if t missing) ---
    if has_t and has_end:
        t  = C[lane_t_col].astype(float).to_numpy()
        le = C['link_end'].astype(object).to_numpy()
        near_mask = ((le == 'u') & (t <= float(near_t))) | ((le == 'v') & ((1.0 - t) <= float(near_t)))
    elif has_t:
        t = C[lane_t_col].astype(float).to_numpy()
        near_mask = np.minimum(t, 1.0 - t) <= float(near_t)
    else:
        near_mask = np.ones(len(C), dtype=bool)

    Cn = C.loc[near_mask].copy()
    if Cn.empty:
        gcols = ['road_node_id','link_id'] + (['link_end'] if has_end else [])
        sort_cols = gcols + (['d_node'] if 'd_node' in C.columns else [])
        if 'd_seg' in C.columns: sort_cols.append('d_seg')
        return C.iloc[0:0].copy().sort_values(sort_cols, kind='mergesort').reset_index(drop=True)

    # --- 1.5) Lane id normalization: explode lists; make NaNs unique per row ---
    if lane_no_col not in Cn.columns:
        raise KeyError(f"Missing required column '{lane_no_col}'")

    def _is_listy(v):
        return isinstance(v, (list, tuple, set, np.ndarray))

    if Cn[lane_no_col].apply(_is_listy).any():
        Cn = Cn.assign(
            **{lane_no_col: Cn[lane_no_col].apply(
                lambda v: list(v) if _is_listy(v) else ([v] if pd.notna(v) else [np.nan])
            )}
        ).explode(lane_no_col, ignore_index=True)

    # Create a scalar, hashable lane key; NaNs become unique labels so they don't collapse
    if '_lane_key' in Cn.columns:
        Cn = Cn.drop(columns=['_lane_key'])
    lane_vals = Cn[lane_no_col]
    if lane_vals.isna().any():
        # use index to disambiguate NaNs
        Cn['_lane_key'] = np.where(lane_vals.isna(),
                                   Cn.index.map(lambda i: ('__NA_LANE__', int(i))),
                                   lane_vals)
    else:
        Cn['_lane_key'] = lane_vals

    # --- 2) CLOSE-TO-LINK mask ---
    if not has_dseg:
        # If no per-link distance available, treat all as close (old behavior)
        close_mask = np.ones(len(Cn), dtype=bool)
    else:
        close_mask = Cn[link_dist_col].astype(float).to_numpy() <= float(link_tol)

    # --- 3) Per-lane thinning, but ONLY among CLOSE rows ---
    req = ['road_node_id','link_id','d_node','road_x','road_y','lane_x','lane_y']
    missing = [c for c in req if c not in Cn.columns]
    if use_opposite_side_rule and missing:
        raise KeyError(f"Missing required columns for opposite-side rule: {missing}")

    gcols = ['road_node_id','link_id'] + (['link_end'] if has_end else []) + ['_lane_key']

    parts = []
    for _, G in Cn.groupby(gcols, sort=False, dropna=False):
        if len(G) <= 1:
            parts.append(G)
            continue

        g_idx   = G.index.to_numpy()
        g_close = close_mask[g_idx]

        # If nothing close in this group, keep all
        if not np.any(g_close):
            parts.append(G)
            continue

        # Among CLOSE rows: keep minimal d_node (within tol)
        d  = G['d_node'].to_numpy(float)
        d_close = d[g_close]
        d_min   = np.min(d_close)
        keep_close = d[g_close] <= (d_min + float(tol_node))

        if use_opposite_side_rule:
            # For CLOSE rows that would be dropped, check opposite-side condition
            rcx = float(G['road_x'].iloc[0]); rcy = float(G['road_y'].iloc[0])
            px  = G['lane_x'].to_numpy(float); py = G['lane_y'].to_numpy(float)
            vx, vy = px - rcx, py - rcy

            # indices within the group's CLOSE subset
            idx_close = np.flatnonzero(g_close)
            # precompute dot matrix only for close subset
            vxc, vyc = vx[g_close], vy[g_close]
            dot = np.outer(vxc, vxc) + np.outer(vyc, vyc)  # dot(i,j)

            # mark drops only if there exists a strictly-closer point on opposite side
            drop_close = np.zeros_like(keep_close, dtype=bool)
            for i in range(len(idx_close)):
                if keep_close[i]:
                    continue  # already keeper as min (or within tol)
                # j candidates strictly closer
                closer = d_close < (d_close[i] - float(tol_node))
                if not np.any(closer):
                    continue
                # among closer, is any opposite side?
                opp = dot[i, :] < -float(side_eps)
                if np.any(closer & opp):
                    drop_close[i] = True
            # finalize which CLOSE rows to keep
            keep_close = keep_close | (~drop_close)

        # Build final keep mask for the whole group
        keep_mask = np.ones(len(G), dtype=bool)
        # CLOSE rows: keep as decided; NON-CLOSE rows: keep all
        keep_mask[g_close] = keep_close

        parts.append(G.iloc[keep_mask])

    out = pd.concat(parts, ignore_index=True) if parts else Cn.iloc[0:0].copy()

    # --- tidy sort ---
    sort_cols = gcols + (['d_node'] if 'd_node' in out.columns else [])
    if link_dist_col in out.columns: sort_cols.append(link_dist_col)
    return out.sort_values(sort_cols, kind='mergesort').reset_index(drop=True)

import numpy as np
import pandas as pd

def sector_cull_connection_points(
    C: pd.DataFrame,
    *,
    scope: str = "node",            # "node" | "node_by_end" | "link" | "link_by_end"
    group_by_lane: bool = True,
    lane_no_col: str = "lane_lnos",
    theta_min_deg: float = 25.0,
    theta_max_deg: float = 120.0,
    tol_node: float = 0.05
) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    if C is None or C.empty:
        return C

    # --- grouping per scope ---
    if scope == "node":
        gcols = ['road_node_id']
    elif scope == "node_by_end":
        gcols = ['road_node_id'] + (['link_end'] if 'link_end' in C.columns else [])
    elif scope == "link":
        gcols = ['road_node_id','link_id']
    elif scope == "link_by_end":
        gcols = ['road_node_id','link_id'] + (['link_end'] if 'link_end' in C.columns else [])
    else:
        raise ValueError(f"Unknown scope='{scope}'")

    # --- optional lane grouping ---
    def _is_listy(v): return isinstance(v, (list, tuple, set, np.ndarray))
    if group_by_lane:
        if lane_no_col not in C.columns:
            raise KeyError(f"Missing '{lane_no_col}' for group_by_lane=True")
        C2 = C.copy()
        if C2[lane_no_col].apply(_is_listy).any():
            C2 = (C2.assign(**{
                lane_no_col: C2[lane_no_col].apply(
                    lambda v: list(v) if _is_listy(v) else ([v] if pd.notna(v) else [np.nan])
                )
            }).explode(lane_no_col, ignore_index=True))
        if '_lane_key' in C2.columns:
            C2 = C2.drop(columns=['_lane_key'])
        lv = C2[lane_no_col]
        C2['_lane_key'] = np.where(lv.isna(),
                                   C2.index.map(lambda i: ('__NA_LANE__', int(i))),
                                   lv)
        gcols = gcols + ['_lane_key']
    else:
        C2 = C.copy()

    need = ['road_x','road_y','lane_x','lane_y','d_node']
    miss = [c for c in need if c not in C2.columns]
    if miss: raise KeyError(f"Missing required columns: {miss}")

    th_min = np.deg2rad(float(theta_min_deg))
    th_max = np.deg2rad(float(theta_max_deg))

    def _ang_from(ox, oy, px, py): return np.arctan2(py-oy, px-ox) % (2*np.pi)

    parts = []
    for _, G in C2.groupby(gcols, sort=False, dropna=False):
        n = len(G)
        if n <= 2:
            parts.append(G); continue

        rx = float(G['road_x'].iloc[0]); ry = float(G['road_y'].iloc[0])
        px = G['lane_x'].to_numpy(dtype=float)
        py = G['lane_y'].to_numpy(dtype=float)
        d  = G['d_node'].to_numpy(dtype=float)

        ang = _ang_from(rx, ry, px, py)  # [0, 2π)
        order = np.argsort(ang)
        ang = ang[order]; d = d[order]; G = G.iloc[order]

        drop = np.zeros(n, dtype=bool)

        for i in range(n):
            if drop[i]: continue
            for j in range(i+1, n):
                if drop[j]: continue

                a1, a2 = ang[i], ang[j]
                delta = (a2 - a1) % (2*np.pi)

                # smaller sector [start,end)
                if delta > np.pi:
                    start, end = a2, a1 + 2*np.pi
                else:
                    start, end = a1, a2
                sec_angle = end - start
                if not (th_min <= sec_angle <= th_max):
                    continue

                # NEW: only cull if point is farther than BOTH boundary points
                farther = max(d[i], d[j])

                for k in range(n):
                    if k in (i, j) or drop[k]: 
                        continue
                    ak = ang[k]
                    ak_u = ak if ak >= start else ak + 2*np.pi
                    inside = (start < ak_u) and (ak_u < end)  # strictly inside
                    if inside and (d[k] >= farther + float(tol_node)):
                        drop[k] = True

        parts.append(G.iloc[~drop])

    out = pd.concat(parts, ignore_index=True) if parts else C2.iloc[0:0].copy()
    sort_cols = [c for c in ['road_node_id','link_id','link_end','_lane_key'] if c in out.columns]
    if 'd_node' in out.columns: sort_cols.append('d_node')
    return out.sort_values(sort_cols, kind='mergesort').reset_index(drop=True)



import numpy as np
import pandas as pd
from collections import Counter

def drop_unlinked_candidates(
    C: pd.DataFrame,
    *,
    by_end: bool = True,
    group_by_lane: bool = False,
    lane_no_col: str = "lane_lnos",
    incident_col: str = "incident_edges",
    consider_only_close: bool = False,
    link_dist_col: str = "d_seg",
    link_tol: float = 0.50,
    keep_singletons: bool = False,      # NEW: drop groups with 1 row by default
    mark_flag_col: str | None = None,   # optional: write a boolean flag of kept rows
) -> pd.DataFrame:
    """
    Remove rows whose incident_edges do not overlap with any other row's
    incident_edges in the same group. If keep_singletons=False, groups with
    a single row are dropped.

    Group keys: (road_node_id, link_id[, link_end][, lane]).
    """
    if C is None or C.empty:
        return C

    # build grouping columns
    gcols = ['road_node_id', 'link_id']
    if by_end and ('link_end' in C.columns):
        gcols.append('link_end')

    D = C.copy()
    if group_by_lane:
        if lane_no_col not in D.columns:
            raise KeyError(f"Missing '{lane_no_col}' while group_by_lane=True")
        def _is_listy(v): return isinstance(v, (list, tuple, set, np.ndarray))
        if D[lane_no_col].apply(_is_listy).any():
            D = D.assign(
                **{lane_no_col: D[lane_no_col].apply(
                    lambda v: list(v) if _is_listy(v) else ([v] if pd.notna(v) else [np.nan])
                )}
            ).explode(lane_no_col, ignore_index=True)
        gcols.append(lane_no_col)

    if incident_col not in D.columns:
        raise KeyError(f"Missing '{incident_col}' column")

    if consider_only_close:
        if link_dist_col not in D.columns:
            raise KeyError(f"consider_only_close=True but '{link_dist_col}' not present")
        close_mask_all = D[link_dist_col].to_numpy(float) <= float(link_tol)
    else:
        close_mask_all = None

    keep_flags = np.zeros(len(D), dtype=bool)

    for _, G in D.groupby(gcols, sort=False, dropna=False):
        idx = G.index.to_numpy()
        m = len(G)

        if m == 1:
            # drop singleton unless requested to keep
            keep_flags[idx] = bool(keep_singletons)
            continue

        # choose which rows participate in edge-overlap test
        if close_mask_all is None:
            test_mask = np.ones(m, dtype=bool)
        else:
            test_mask = close_mask_all[idx]

        # convert incident_edges to sets
        sets = []
        for v in G[incident_col].tolist():
            try:
                s = set(v)  # list/tuple/set/ndarray of hashables
            except TypeError:
                s = set()   # None or non-iterable -> empty
            sets.append(s)

        # frequency of edges among rows we test
        freq = Counter()
        for use, s in zip(test_mask, sets):
            if use:
                freq.update(s)

        # keep rule
        kept = np.zeros(m, dtype=bool)
        for i, (use, s) in enumerate(zip(test_mask, sets)):
            if not use:
                # out of scope for test: keep as-is **only if you want**.
                # If you prefer to drop out-of-scope rows too, set to False.
                kept[i] = not consider_only_close
            else:
                # keep if s shares any edge with someone else (freq >=2)
                kept[i] = any(freq[e] >= 2 for e in s)

        keep_flags[idx] = kept

    out = D.loc[keep_flags].copy()

    if mark_flag_col:
        # Optional: annotate original C with flags so you can debug on plots
        C2 = C.copy()
        C2[mark_flag_col] = False
        C2.loc[out.index, mark_flag_col] = True
        out = C2.loc[C2[mark_flag_col]].drop(columns=[mark_flag_col])

    # tidy sort
    sort_cols = [c for c in ['road_node_id','link_id','link_end',lane_no_col] if c in out.columns]
    if 'd_node' in out.columns: sort_cols.append('d_node')
    if link_dist_col in out.columns: sort_cols.append(link_dist_col)
    return out.sort_values(sort_cols, kind='mergesort').reset_index(drop=True)

import numpy as np
import pandas as pd

def drop_if_no_peer_sharing_edge(
    C: pd.DataFrame,
    *,
    road_node_col: str = "road_node_id",
    incident_col: str = "incident_edges",
    cp_id_col: str = "lane_node_id",

    # NEW: coverage protection
    protect_link_ends: bool = True,
    link_id_col: str = "link_id",
    link_end_col: str = "link_end",
    link_dist_col: str = "d_seg",
    node_dist_col: str = "d_node",
) -> pd.DataFrame:
    """
    Peer filter (heuristic) BUT optionally guarantees:
      every (link_id, link_end) that existed in input C remains covered.

    This is critical for tile/boundary links and synthetic CPs.
    """
    if C is None or C.empty:
        return C

    for col in (road_node_col, incident_col, cp_id_col):
        if col not in C.columns:
            raise KeyError(f"Required column '{col}' not found")

    df0 = C.copy()

    # ---------- original peer logic ----------
    def _to_set(x):
        try:
            return set(e for e in x if pd.notna(e))
        except TypeError:
            return set()

    df = df0.copy()
    df["_edge_set"] = df[incident_col].apply(_to_set)

    edge_map = {}
    for rn, cp, s in zip(df[road_node_col].to_numpy(),
                         df[cp_id_col].to_numpy(),
                         df["_edge_set"].to_numpy()):
        for e in s:
            edge_map.setdefault((rn, e), set()).add(cp)

    keep = np.zeros(len(df), dtype=bool)
    for i, (rn, cp, s) in enumerate(zip(df[road_node_col].to_numpy(),
                                        df[cp_id_col].to_numpy(),
                                        df["_edge_set"].to_numpy())):
        if not s:
            keep[i] = False
            continue
        for e in s:
            peers = edge_map.get((rn, e), set())
            if any(p != cp for p in peers):
                keep[i] = True
                break

    out = df.loc[keep].drop(columns=["_edge_set"]).reset_index(drop=True)

    # ---------- NEW: coverage-preserving repair ----------
    if protect_link_ends:
        for col in (link_id_col, link_end_col):
            if col not in df0.columns:
                raise KeyError(f"protect_link_ends=True requires column '{col}' in C")
        for col in (link_dist_col, node_dist_col):
            if col not in df0.columns:
                # you can loosen this, but these exist in your pipeline
                raise KeyError(f"protect_link_ends=True requires column '{col}' in C")

        required = set(zip(df0[link_id_col].astype(str), df0[link_end_col].astype(str)))
        covered  = set(zip(out[link_id_col].astype(str), out[link_end_col].astype(str)))
        missing  = required - covered
        if missing:
            # pick best candidate per missing end from df0
            df0["_lid"] = df0[link_id_col].astype(str)
            df0["_end"] = df0[link_end_col].astype(str)

            add_rows = []
            for lid, end in missing:
                cand = df0[(df0["_lid"] == lid) & (df0["_end"] == end)].copy()
                if cand.empty:
                    continue
                cand.sort_values([link_dist_col, node_dist_col], ascending=[True, True],
                                 kind="mergesort", inplace=True)
                add_rows.append(cand.iloc[0])

            if add_rows:
                add_df = pd.DataFrame(add_rows).drop(columns=["_lid","_end"], errors="ignore")
                out = pd.concat([out, add_df], ignore_index=True)

            df0.drop(columns=["_lid","_end"], inplace=True, errors="ignore")

    return out.reset_index(drop=True)

def keep_nearest_cp_per_link_end(
    C: pd.DataFrame,
    *,
    link_id_col: str = "link_id",
    link_end_col: str = "link_end",
    road_node_col: str = "road_node_id",
    cp_id_col: str = "lane_node_id",
    link_dist_col: str = "d_seg",
    node_dist_col: str = "d_node",
) -> pd.DataFrame:
    """
    Keep exactly one best CP candidate for each (road_node_id, link_id, link_end).

    This preserves full link-end coverage and allows the same CP to be reused
    across multiple link-ends if geometry dictates it.
    """
    if C is None or C.empty:
        return C

    need = [road_node_col, link_id_col, link_end_col, cp_id_col, link_dist_col]
    miss = [c for c in need if c not in C.columns]
    if miss:
        raise KeyError(f"Missing required columns in C: {miss}")

    df = C.copy()

    # stable ordering so ties are deterministic
    df["_ord"] = np.arange(len(df), dtype=np.int64)

    sort_cols = [road_node_col, link_id_col, link_end_col, link_dist_col]
    if node_dist_col in df.columns:
        sort_cols.append(node_dist_col)
    sort_cols.append("_ord")

    df = df.sort_values(sort_cols, ascending=True, kind="mergesort")

    # one winner per (road node, link end)
    df = df.drop_duplicates(subset=[road_node_col, link_id_col, link_end_col], keep="first")

    return df.drop(columns=["_ord"]).reset_index(drop=True)


import numpy as np
import pandas as pd

def keep_nearest_link_per_connection(
    C: pd.DataFrame,
    *,
    cp_id_col: str = "lane_node_id",   # the connection-point identifier
    link_dist_col: str = "d_seg",      # "distance to link" metric
    node_dist_col: str = "d_node",     # secondary tie-break
    tol_link: float = 1e-9,            # tie tolerance on link distance
) -> pd.DataFrame:
    """
    If the same connection point appears for different road nodes,
    keep only the row(s) with the smallest link distance.
    Ties within tol_link are broken by smallest d_node.

    Requirements:
      - Columns: cp_id_col, link_dist_col
      - node_dist_col is optional (only used for tie-break if present)
    """
    if C is None or C.empty:
        return C
    for col in (cp_id_col, link_dist_col):
        if col not in C.columns:
            raise KeyError(f"Required column '{col}' not found")

    df = C.copy()

    # 1) min link distance per connection-point id
    dmin = (df.groupby(cp_id_col, as_index=False, dropna=False)[link_dist_col]
              .min().rename(columns={link_dist_col: "_link_min"}))
    out = df.merge(dmin, on=cp_id_col, how="left")

    # 2) keep rows within tolerance of the minimum link distance
    close = out[link_dist_col].to_numpy(float) <= (out["_link_min"].to_numpy(float) + float(tol_link))
    out = out.loc[close].copy()
    out.drop(columns=["_link_min"], inplace=True)

    # 3) if multiple rows still remain for the same cp_id, use smallest d_node as tie-break
    if node_dist_col in out.columns:
        out["_ord"] = np.arange(len(out))
        out = (out.sort_values([cp_id_col, node_dist_col, "_ord"],
                               ascending=[True, True, True],
                               kind="mergesort")
                  .drop_duplicates(subset=[cp_id_col], keep="first")
                  .drop(columns=["_ord"])
                  .reset_index(drop=True))
    else:
        # no secondary tie-break available → keep all within tol
        out = out.reset_index(drop=True)

    return out

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any

# --- convex hull via monotone chain (dependency-free) ---
def _convex_hull_xy(xy: np.ndarray) -> np.ndarray:
    if xy.shape[0] <= 1:
        return xy
    pts = xy[np.lexsort((xy[:,1], xy[:,0]))]  # sort by x then y

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))

    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))

    hull = np.array(lower[:-1] + upper[:-1], dtype=float)
    return hull

def _cp_xy_map_from_lanes(lanes: List[Dict[str, Any]],
                          id_key: str = "lane_node_id",
                          x_key: str = "lane_x",
                          y_key: str = "lane_y") -> Dict[int, Tuple[float,float]]:
    m: Dict[int, Tuple[float,float]] = {}
    for rec in lanes:
        if id_key in rec and x_key in rec and y_key in rec:
            m[int(rec[id_key])] = (float(rec[x_key]), float(rec[y_key]))
    return m

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from typing import Any, Dict, List, Tuple

# --- convex hull (Andrew’s monotone chain) ---
def _convex_hull_xy(xy: np.ndarray) -> np.ndarray:
    if xy.shape[0] <= 1:
        return xy
    pts = xy[np.lexsort((xy[:,1], xy[:,0]))]
    def cross(o, a, b): return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))
    return np.array(lower[:-1] + upper[:-1], dtype=float)

def _cp_xy_map_from_lanes(lanes: List[Dict[str, Any]],
                          id_key="lane_node_id", x_key="lane_x", y_key="lane_y") -> Dict[int, Tuple[float,float]]:
    m = {}
    for rec in lanes:
        if id_key in rec and x_key in rec and y_key in rec:
            m[int(rec[id_key])] = (float(rec[x_key]), float(rec[y_key]))
    return m

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

def _arrow_mutation_scale(ax, x0, y0, x1, y1, frac=0.10, min_px=8, max_px=28):
    p0 = ax.transData.transform((x0, y0))
    p1 = ax.transData.transform((x1, y1))
    seg_px = float(np.hypot(*(p1 - p0)))
    return max(min_px, min(max_px, seg_px * frac))

def _draw_center_arrow(ax, p0, p1, color, lw,
                       center_len_frac=0.25,  # portion of segment used for the arrow shaft
                       offset_frac=0.0,       # perpendicular offset as a fraction of segment length
                       head_frac=0.10,        # head size ~ this frac of segment (in px, via mutation_scale)
                       alpha=0.95,
                       z=4.0):
    x0, y0 = p0; x1, y1 = p1
    dx, dy = (x1 - x0), (y1 - y0)
    L = float(np.hypot(dx, dy))
    if L <= 0:
        return
    ux, uy = dx / L, dy / L

    # center of the segment
    mx, my = (x0 + x1) * 0.5, (y0 + y1) * 0.5

    # small shaft centered at mid
    half = 0.5 * center_len_frac * L
    sx0, sy0 = mx - half * ux, my - half * uy
    sx1, sy1 = mx + half * ux, my + half * uy

    # optional perpendicular offset (useful when drawing both directions)
    if offset_frac != 0.0:
        px, py = -uy, ux  # unit perpendicular
        off = offset_frac * L
        sx0 += px * off; sy0 += py * off
        sx1 += px * off; sy1 += py * off

    ms = _arrow_mutation_scale(ax, sx0, sy0, sx1, sy1, frac=head_frac)

    # draw a small line first (for visibility on long edges)
    ax.plot([sx0, sx1], [sy0, sy1], color=color, linewidth=lw, alpha=alpha, zorder=z-0.1)

    # centered arrow
    patch = FancyArrowPatch((sx0, sy0), (sx1, sy1),
                            arrowstyle='-|>', mutation_scale=ms,
                            linewidth=lw, color=color, alpha=alpha,
                            shrinkA=0, shrinkB=0, zorder=z)
    ax.add_patch(patch)


def plot_junctions_from_adj(
    ax,
    junctions: List[Any],                 # JunctionNX with .node_ids, .cp_adj, .lanes
    *,
    cp_id_key: str = "lane_node_id",
    cp_x_key: str  = "lane_x",
    cp_y_key: str  = "lane_y",
    color_map: str = "tab10",
    hull_linestyle: str = ":",
    hull_linewidth: float = 1.6,
    edge_linewidth: float = 2.0,
    edge_alpha: float = 0.95,
    cp_size: float = 18,
    draw_cp_labels: bool = False,
    directed_matrix: bool = True,         # True: draw arrow for every A[i,j] != -1
):
    import matplotlib.cm as cm

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    cmap = cm.get_cmap(color_map, max(1, len(junctions)))

    for j, jnx in enumerate(junctions):
        col = cmap(j)

        # --- CP coordinates (by id) ---
        id2xy = _cp_xy_map_from_lanes(jnx.lanes, cp_id_key, cp_x_key, cp_y_key)
        cp_ids = list(jnx.node_ids)
        xy = np.array([id2xy[i] for i in cp_ids if i in id2xy], dtype=float)

        # --- convex hull ---
        if xy.shape[0] >= 3:
            hull = _convex_hull_xy(xy)
            if hull.shape[0] >= 3:
                ax.plot(np.r_[hull[:,0], hull[0,0]], np.r_[hull[:,1], hull[0,1]],
                        linestyle=hull_linestyle, linewidth=hull_linewidth,
                        color=col, zorder=2.5)

        # --- CP scatter & labels ---
        if xy.size:
            ax.scatter(xy[:,0], xy[:,1], s=cp_size, color=col, edgecolors='white',
                       linewidths=0.5, zorder=3.2)
        if draw_cp_labels:
            for nid in cp_ids:
                if nid in id2xy:
                    x, y = id2xy[nid]
                    ax.text(x, y, str(nid), color=col, fontsize=8, ha='left', va='bottom', zorder=3.3)

        # --- edges with arrows from adjacency ---
        # draw base line for every connected pair, then a centered arrow for direction
        def _is_conn(v):
            # valid if not -1/None/NaN (handles object dtype with strings/ints)
            if v is None: 
                return False
            if isinstance(v, float) and np.isnan(v):
                return False
            return v != -1

        def _arrow_mutation_scale(ax, x0, y0, x1, y1, frac=0.10, min_px=8, max_px=28):
            p0 = ax.transData.transform((x0, y0))
            p1 = ax.transData.transform((x1, y1))
            seg_px = float(np.hypot(*(p1 - p0)))
            return max(min_px, min(max_px, seg_px * frac))

        def _draw_center_arrow(ax, p_from, p_to, color, lw,
                            center_len_frac=0.25,   # arrow shaft length as fraction of segment
                            offset_frac=0.0,        # perpendicular offset (for bidirectional)
                            head_frac=0.12,         # head size relative to segment in px
                            alpha=0.95, z=4.0):
            x0, y0 = p_from; x1, y1 = p_to
            dx, dy = (x1 - x0), (y1 - y0)
            L = float(np.hypot(dx, dy))
            if L <= 0:
                return
            ux, uy = dx / L, dy / L
            mx, my = (x0 + x1) * 0.5, (y0 + y1) * 0.5  # segment midpoint

            half = 0.5 * center_len_frac * L
            sx0, sy0 = mx - half * ux, my - half * uy
            sx1, sy1 = mx + half * ux, my + half * uy

            if offset_frac:
                px, py = -uy, ux  # unit perpendicular
                off = offset_frac * L
                sx0 += px * off; sy0 += py * off
                sx1 += px * off; sy1 += py * off

            ms = _arrow_mutation_scale(ax, sx0, sy0, sx1, sy1, frac=head_frac)

            # optional small shaft for visibility
            ax.plot([sx0, sx1], [sy0, sy1], color=color, linewidth=lw, alpha=alpha, zorder=z-0.1)

            # centered arrow from FROM -> TO
            patch = FancyArrowPatch((sx0, sy0), (sx1, sy1),
                                    arrowstyle='-|>',
                                    mutation_scale=ms,
                                    linewidth=lw,
                                    color=color,
                                    alpha=alpha,
                                    shrinkA=0, shrinkB=0,
                                    zorder=z)
            ax.add_patch(patch)
        A = np.asarray(jnx.cp_adj, dtype=object)
        n = A.shape[0]
        idx2xy = [id2xy.get(cp_ids[i], None) for i in range(n)]

        for i in range(n):
            p_i = idx2xy[i]
            if p_i is None:
                continue
            for j2 in range(i+1, n):
                p_j = idx2xy[j2]
                if p_j is None:
                    continue

                ij = _is_conn(A[i, j2])   # i -> j
                ji = _is_conn(A[j2, i])   # j -> i

                if not (ij or ji):
                    continue

                # draw a base line for context (undirected backbone)
                ax.plot([p_i[0], p_j[0]], [p_i[1], p_j[1]],
                        color=col, linewidth=edge_linewidth, alpha=edge_alpha, zorder=3.0)

                if ij and ji:
                    # two directions: draw two centered arrows with tiny opposite offsets
                    _draw_center_arrow(ax, p_i, p_j, col, edge_linewidth,
                                    center_len_frac=0.30, offset_frac=+0.02,
                                    head_frac=0.12, alpha=edge_alpha, z=4.0)
                    _draw_center_arrow(ax, p_j, p_i, col, edge_linewidth,
                                    center_len_frac=0.30, offset_frac=-0.02,
                                    head_frac=0.12, alpha=edge_alpha, z=4.0)
                elif ij:
                    _draw_center_arrow(ax, p_i, p_j, col, edge_linewidth,
                                    center_len_frac=0.30, offset_frac=0.0,
                                    head_frac=0.12, alpha=edge_alpha, z=4.0)
                else:  # ji
                    _draw_center_arrow(ax, p_j, p_i, col, edge_linewidth,
                                    center_len_frac=0.30, offset_frac=0.0,
                                    head_frac=0.12, alpha=edge_alpha, z=4.0)


    ax.set_aspect('equal')
    ax.grid(True, linestyle=':')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')


import math
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree
import networkx as nx

# ----------------------------
# Helpers you may need to adapt
# ----------------------------
import numpy as np
import pandas as pd

def _to_array(val):
    """Coerce list/tuple/ndarray/Series to 1D float np.array; return None if unusable."""
    if isinstance(val, (list, tuple, np.ndarray, pd.Series)):
        arr = np.asarray(val, dtype=float).ravel()
        return arr if arr.size >= 2 else None
    return None

def _dedupe_consecutive(x, y):
    """Remove consecutive duplicate points."""
    if x is None or y is None: return None, None
    if x.size != y.size or x.size < 2: return None, None
    keep = np.ones(x.size, dtype=bool)
    keep[1:] = (np.diff(x) != 0) | (np.diff(y) != 0)
    return x[keep], y[keep]

def get_polyline(row) -> np.ndarray:
    """
    Return Nx2 array (float) representing the link polyline oriented u->v.

    Priority:
      1) (u_x, u_y) as the full polyline from u to v (v_x/v_y is its reverse).
      2) (x_all, y_all) if available.
      3) sx*/sy* + ex*/ey* samples if present.
      4) Fallback to straight segment using (ux,uy)->(vx,vy).
    """
    # --- 1) u_x / u_y ---
    ux_arr = _to_array(row.get('u_x'))
    uy_arr = _to_array(row.get('u_y'))
    if ux_arr is not None and uy_arr is not None:
        ux_arr, uy_arr = _dedupe_consecutive(ux_arr, uy_arr)
        if ux_arr is not None:
            return np.column_stack([ux_arr, uy_arr])

    # --- 2) x_all / y_all ---
    xa = _to_array(row.get('x_all'))
    ya = _to_array(row.get('y_all'))
    if xa is not None and ya is not None:
        xa, ya = _dedupe_consecutive(xa, ya)
        if xa is not None:
            # ensure orientation u->v if we can
            ux, uy, vx, vy = row.get('ux'), row.get('uy'), row.get('vx'), row.get('vy')
            if pd.notna(ux) and pd.notna(uy) and pd.notna(vx) and pd.notna(vy):
                # choose direction whose endpoints are closer to (ux,uy)->(vx,vy)
                d_forward  = np.hypot(xa[0]-ux, ya[0]-uy) + np.hypot(xa[-1]-vx, ya[-1]-vy)
                d_backward = np.hypot(xa[-1]-ux, ya[-1]-uy) + np.hypot(xa[0]-vx, ya[0]-vy)
                if d_backward < d_forward:
                    xa = xa[::-1]; ya = ya[::-1]
            return np.column_stack([xa, ya])

    # --- 3) sx*/sy* + ex*/ey* (sparse samples) ---
    pts = []
    for k in range(20):  # generous cap
        xk = row.get(f'sx{k}', None); yk = row.get(f'sy{k}', None)
        if pd.notna(xk) and pd.notna(yk): pts.append((float(xk), float(yk)))
        else: break
    for k in range(20):
        xk = row.get(f'ex{k}', None); yk = row.get(f'ey{k}', None)
        if pd.notna(xk) and pd.notna(yk): pts.append((float(xk), float(yk)))
        else: break
    if len(pts) >= 2:
        P = np.array(pts, dtype=float)
        # orient u->v if endpoints exist
        ux, uy, vx, vy = row.get('ux'), row.get('uy'), row.get('vx'), row.get('vy')
        if pd.notna(ux) and pd.notna(uy) and pd.notna(vx) and pd.notna(vy):
            d_forward  = np.hypot(P[0,0]-ux, P[0,1]-uy) + np.hypot(P[-1,0]-vx, P[-1,1]-vy)
            d_backward = np.hypot(P[-1,0]-ux, P[-1,1]-uy) + np.hypot(P[0,0]-vx, P[0,1]-vy)
            if d_backward < d_forward:
                P = P[::-1]
        # drop consecutive duplicates
        keep = np.ones(len(P), dtype=bool)
        keep[1:] = (np.diff(P[:,0]) != 0) | (np.diff(P[:,1]) != 0)
        return P[keep]

    # --- 4) Fallback: straight from endpoints ---
    ux, uy = float(row['ux']), float(row['uy'])
    vx, vy = float(row['vx']), float(row['vy'])
    return np.array([[ux, uy], [vx, vy]], dtype=float)


def headings(poly):
    d = np.diff(poly, axis=0)
    ang = np.arctan2(d[:,1], d[:,0])
    return ang

def unwrap(ang):
    return np.unwrap(ang)

def total_turn_deg(ang):
    a = unwrap(ang)
    return math.degrees(a[-1] - a[0])

def min_radius(poly):
    # discrete curvature κ ≈ |Δθ|/Δs -> radius ≈ 1/κ
    d = np.diff(poly, axis=0)
    seglen = np.hypot(d[:,0], d[:,1])
    ang = headings(poly)
    dang = np.abs(np.diff(unwrap(ang)))
    ds = (seglen[:-1] + seglen[1:]) * 0.5
    with np.errstate(divide='ignore', invalid='ignore'):
        kappa = np.where(ds>0, dang/ds, 0.0)
        R = np.where(kappa>0, 1.0/kappa, np.inf)
    return np.nanmin(R) if R.size else np.inf

def straightness(poly):
    # R^2 of linear fit as a straightness proxy
    x, y = poly[:,0], poly[:,1]
    X = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - y.mean())**2) + 1e-9
    return 1.0 - ss_res/ss_tot

def line_geom(poly): return LineString(poly)

import networkx as nx
import numpy as np

def build_junction_graph_from_links(
    junctions,
    links_df,
    *,
    directed: bool = True,
) -> nx.MultiDiGraph:
    """
    Build a junction-level graph:
      - nodes: junctions (one per JunctionNX, keyed by jid)
      - edges: HERE links whose (u,v) are both junction node ids

    Parameters
    ----------
    junctions : list[JunctionNX]
        Each object must have at least:
          .jid (int), .centroid -> (x, y), .n_legs, .radius_m, .itype, .lanes, .cp_adj
    links_df : pandas.DataFrame
        From road_links_dict_to_df(...), must contain:
          'link_id', 'u', 'v', 'x_all', 'y_all',
          'u_x', 'u_y', 'v_x', 'v_y', 'u_s', 'v_s', 'length_m'
    directed : bool, optional
        If True, return MultiDiGraph, else MultiGraph.

    Returns
    -------
    G : nx.MultiGraph or nx.MultiDiGraph
        Junction graph.
    """
    GraphCls = nx.MultiDiGraph if directed else nx.MultiGraph
    G = GraphCls()

    # --- node layer: one node per junction ---
    jid2j = {}
    for jnx in junctions:
        jid = int(jnx.jid)
        jid2j[jid] = jnx
        cx, cy = jnx.centroid

        G.add_node(
            jid,
            junction=jnx,               # keep the full object
            x = float(cx), y = float(cy),
            centroid_x=float(cx),
            centroid_y=float(cy),
            itype=getattr(jnx, "itype", None),
            n_legs=getattr(jnx, "n_legs", None),
            radius_m=getattr(jnx, "radius_m", None),
        )

    junction_ids = set(jid2j.keys())

    # --- edge layer: links between junction nodes ---
    for _, row in links_df.iterrows():
        u = int(row["u"])
        v = int(row["v"])

        # keep only links that connect two junction nodes
        if (u not in junction_ids) or (v not in junction_ids):
            continue

        # pull all geometry-related attributes you might need
        attrs = {
            "link_id": row["link_id"],
            "length_m": float(row["length_m"]),
            "x_all": row["x_all"],
            "y_all": row["y_all"],
            "u_x": row["u_x"],
            "u_y": row["u_y"],
            "v_x": row["v_x"],
            "v_y": row["v_y"],
            "u_s": row["u_s"],
            "v_s": row["v_s"],
        }

        # edge orientation is u -> v as in HERE
        G.add_edge(u, v, **attrs)

        # for undirected, one edge is enough; MultiGraph will treat it as bidirectional
        if not directed:
            continue

    return G


from typing import Optional, List, Any
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

def _center_arrow_on_polyline(
    ax,
    xs,
    ys,
    *,
    color="k",
    lw=1.5,
    alpha=0.9,
    frac: float = 0.25,
    head_frac: float = 0.12,
    zorder: float = 4.0,
):
    """
    Draw a centered arrow along a polyline (xs, ys) in data coords.
    'frac' controls the arrow-shaft length as fraction of total length.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size < 2:
        return

    # cumulative arclength to find the midpoint
    dx = np.diff(xs)
    dy = np.diff(ys)
    seg_len = np.hypot(dx, dy)
    L = float(seg_len.sum())
    if L <= 0:
        return

    # target center and shaft length
    center_s = 0.5 * L
    shaft_L = frac * L
    s0 = center_s - 0.5 * shaft_L
    s1 = center_s + 0.5 * shaft_L

    # helper: interpolate point at arclength s along polyline
    cum = np.concatenate(([0.0], np.cumsum(seg_len)))

    def interp_at(s):
        s = np.clip(s, 0.0, L)
        idx = np.searchsorted(cum, s, side="right") - 1
        idx = max(0, min(idx, len(dx) - 1))
        ds = s - cum[idx]
        if seg_len[idx] > 0:
            t = ds / seg_len[idx]
        else:
            t = 0.0
        x = xs[idx] + t * dx[idx]
        y = ys[idx] + t * dy[idx]
        return x, y

    x0, y0 = interp_at(s0)
    x1, y1 = interp_at(s1)

    # small shaft for visibility
    ax.plot([x0, x1], [y0, y1],
            linewidth=lw, color=color, alpha=alpha, zorder=zorder - 0.2)

    # scale arrow head in screen space
    p0 = ax.transData.transform((x0, y0))
    p1 = ax.transData.transform((x1, y1))
    seg_px = float(np.hypot(*(p1 - p0)))
    ms = max(8.0, min(28.0, seg_px * head_frac))

    patch = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle='-|>',
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        alpha=alpha,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(patch)

from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt
from typing import Optional, List, Any
import networkx as nx

def _center_arrow_on_polyline(
    ax,
    xs,
    ys,
    *,
    color="k",
    lw=1.5,
    alpha=0.9,
    frac: float = 0.25,
    head_frac: float = 0.12,
    zorder: float = 4.0,
):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size < 2:
        return

    dx = np.diff(xs)
    dy = np.diff(ys)
    seg_len = np.hypot(dx, dy)
    L = float(seg_len.sum())
    if L <= 0:
        return

    center_s = 0.5 * L
    shaft_L = frac * L
    s0 = center_s - 0.5 * shaft_L
    s1 = center_s + 0.5 * shaft_L

    cum = np.concatenate(([0.0], np.cumsum(seg_len)))

    def interp_at(s):
        s = np.clip(s, 0.0, L)
        idx = np.searchsorted(cum, s, side="right") - 1
        idx = max(0, min(idx, len(dx) - 1))
        ds = s - cum[idx]
        t = ds / seg_len[idx] if seg_len[idx] > 0 else 0.0
        x = xs[idx] + t * dx[idx]
        y = ys[idx] + t * dy[idx]
        return x, y

    x0, y0 = interp_at(s0)
    x1, y1 = interp_at(s1)

    ax.plot([x0, x1], [y0, y1],
            linewidth=lw, color=color, alpha=alpha, zorder=zorder - 0.2)

    p0 = ax.transData.transform((x0, y0))
    p1 = ax.transData.transform((x1, y1))
    seg_px = float(np.hypot(*(p1 - p0)))
    ms = max(8.0, min(28.0, seg_px * head_frac))

    patch = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle='-|>',
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        alpha=alpha,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(patch)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

def _point_dir_at_fraction(poly_xy, frac=0.6):
    """
    Given a polyline P (N,2) and a fraction in [0,1],
    return (point, direction_vector) at that arclength fraction.
    """
    P = np.asarray(poly_xy, float)
    if P.shape[0] < 2:
        return (P[0], np.array([1.0, 0.0], float))

    dif = np.diff(P, axis=0)
    seg_len = np.hypot(dif[:, 0], dif[:, 1])
    total = float(seg_len.sum())
    if total <= 0:
        return (P[0], np.array([1.0, 0.0], float))

    target = frac * total
    acc = 0.0
    for i, L in enumerate(seg_len):
        if acc + L >= target:
            t = (target - acc) / max(L, 1e-9)
            p = P[i] + t * dif[i]
            v = dif[i] / max(L, 1e-9)
            return (p, v)
        acc += L
    # fallback: last segment end
    v = dif[-1] / max(seg_len[-1], 1e-9)
    return (P[-1], v)


def _plot_arrow_on_poly(ax, poly_xy, color, alpha=0.95, z=6,
                        frac=0.6, head_frac=0.12):
    """
    Draw one arrow on the given polyline, at arclength fraction `frac`.
    """
    P = np.asarray(poly_xy, float)
    if P.shape[0] < 2:
        return

    (pt, vdir) = _point_dir_at_fraction(P, frac=frac)
    x0, y0 = float(pt[0]), float(pt[1])
    vx, vy = float(vdir[0]), float(vdir[1])
    L = np.hypot(vx, vy)
    if L <= 0:
        return

    # short shaft for visibility
    shaft_len = 0.4  # meters, purely visual
    sx0, sy0 = x0 - 0.5 * shaft_len * vx, y0 - 0.5 * shaft_len * vy
    sx1, sy1 = x0 + 0.5 * shaft_len * vx, y0 + 0.5 * shaft_len * vy

    ax.plot([sx0, sx1], [sy0, sy1],
            color=color, linewidth=1.2, alpha=alpha, zorder=z-0.1)

    # pick arrow size relative to pixel length
    p0 = ax.transData.transform((sx0, sy0))
    p1 = ax.transData.transform((sx1, sy1))
    seg_px = float(np.hypot(*(p1 - p0)))
    mutation_scale = max(8, min(26, seg_px * head_frac))

    patch = FancyArrowPatch((sx0, sy0), (sx1, sy1),
                            arrowstyle='-|>',
                            mutation_scale=mutation_scale,
                            linewidth=1.2,
                            color=color,
                            alpha=alpha,
                            shrinkA=0, shrinkB=0,
                            zorder=z)
    ax.add_patch(patch)

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from matplotlib.patches import FancyArrowPatch


def _plot_arrow_on_poly(ax, poly_xy, frac=0.6, alpha=0.95, z=7, **kwargs):
    """
    Draw a small arrow along a polyline, aligned with its local tangent.
    poly_xy: (N,2) array.
    frac: fraction of arclength where arrow is centered.
    """
    P = np.asarray(poly_xy, float)
    if P.ndim != 2 or P.shape[0] < 2:
        return

    # cumulative arclength
    d = np.diff(P, axis=0)
    seg_len = np.hypot(d[:, 0], d[:, 1])
    L = float(seg_len.sum())
    if L <= 0:
        return
    target = frac * L

    # find segment containing target
    cum = np.cumsum(seg_len)
    idx = int(np.searchsorted(cum, target))
    idx = max(0, min(idx, len(seg_len) - 1))

    # local segment
    p0 = P[idx]
    p1 = P[idx + 1]
    dx, dy = (p1[0] - p0[0]), (p1[1] - p0[1])
    segL = float(np.hypot(dx, dy))
    if segL <= 0:
        return
    ux, uy = dx / segL, dy / segL

    # small arrow segment
    arrow_len = 0.25 * segL
    start = p0 + 0.4 * (p1 - p0) - 0.5 * arrow_len * np.array([ux, uy])
    end   = start + arrow_len * np.array([ux, uy])

    ms = 10.0
    patch = FancyArrowPatch(
        (start[0], start[1]),
        (end[0],   end[1]),
        arrowstyle='-|>',
        mutation_scale=ms,
        linewidth=kwargs.get("lw", 1.5),
        color=kwargs.get("color", "k"),
        alpha=alpha,
        shrinkA=0,
        shrinkB=0,
        zorder=z,
    )
    ax.add_patch(patch)


def plot_junction_graph_xy(
    J_graph,
    *,
    ax=None,
    lane_node_xy = None,
    show_junction_nodes=True,
    show_junction_lanes=True,
    show_link_lanes=True,
    show_lane_centerlines=True,
    show_lane_boundaries=True,
    junction_node_size=50,
    junction_node_color="tab:red",
    junc_lane_color="tab:green",
    link_lane_palette=None,
    superlink_edge_color: str = "tab:purple",
    superlink_edge_lw: float = 2.0,

    # --- NEW: debug overlay of boring nodes ---
    here_node_df=None,          # DataFrame with ['node_id','x','y'] in same ENU frame
    boring_nodes=None,          # iterable of HERE node_ids
    show_boring_nodes=False,
    boring_node_size=18,
    boring_node_color="tab:blue",
    label_boring=False,
    label_boring_limit=200,     # avoid insane text spam
):
    import numpy as np
    import matplotlib.pyplot as plt
    from itertools import cycle

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure

    # -------------------------------
    # 1) Junction nodes
    # -------------------------------
    def _node_xy(ndata):
        if "x" in ndata and "y" in ndata:
            return float(ndata["x"]), float(ndata["y"])
        if "X" in ndata and "Y" in ndata:
            return float(ndata["X"]), float(ndata["Y"])
        if "pos" in ndata and len(ndata["pos"]) >= 2:
            return float(ndata["pos"][0]), float(ndata["pos"][1])
        return None

    if show_junction_nodes:
        xs, ys = [], []
        for nid, ndata in J_graph.nodes(data=True):
            p = _node_xy(ndata)
            if p is None:
                continue
            xs.append(p[0]); ys.append(p[1])
        if xs:
            ax.scatter(
                xs, ys,
                s=junction_node_size,
                c=junction_node_color,
                edgecolors="black",
                linewidths=0.7,
                zorder=4,
                label="junction nodes",
            )
    # -------------------------------
    # 2) Junction connection points
    # -------------------------------
    if lane_node_xy is not None:

        in_xs, in_ys = [], []
        out_xs, out_ys = [], []

        for nid, ndata in J_graph.nodes(data=True):

            # IN points
            in_pts = ndata.get("in_points", {}) or {}
            for cp_raw in in_pts.keys():
                try:
                    cp = int(cp_raw)
                except Exception:
                    continue
                if cp not in lane_node_xy:
                    continue
                x, y = lane_node_xy[cp]
                in_xs.append(float(x))
                in_ys.append(float(y))

            # OUT points
            out_pts = ndata.get("out_points", {}) or {}
            for cp_raw in out_pts.keys():
                try:
                    cp = int(cp_raw)
                except Exception:
                    continue
                if cp not in lane_node_xy:
                    continue
                x, y = lane_node_xy[cp]
                out_xs.append(float(x))
                out_ys.append(float(y))

        # plot IN CPs (filled green)
        if in_xs:
            ax.scatter(
                in_xs, in_ys,
                s=40,
                c="green",
                edgecolors="black",
                linewidths=0.5,
                zorder=5,
                label="IN CPs",
            )

    # plot OUT CPs (filled red)
    if out_xs:
        ax.scatter(
            out_xs, out_ys,
            s=40,
            c="red",
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
            label="OUT CPs",
        )

    # -------------------------------
    # 2) Junction-local lanes (from nodes)
    # -------------------------------
    if show_junction_lanes:
        for nid, ndata in J_graph.nodes(data=True):
            j_lanes_raw = ndata.get("junction_lanes", []) or []
            for item in j_lanes_raw:
                if isinstance(item, dict) and "lanes" in item:
                    lane_list = item["lanes"] or []
                else:
                    lane_list = [item]

                for lane in lane_list:
                    cl = lane.get("centerline", {})
                    cx = np.asarray(cl.get("x", []), float)
                    cy = np.asarray(cl.get("y", []), float)
                    if cx.size < 2:
                        continue

                    P = np.column_stack([cx, cy])

                    if show_lane_centerlines:
                        ax.plot(cx, cy, "-", color=junc_lane_color, lw=2.0, alpha=0.95, zorder=6)
                        _plot_arrow_on_poly(ax, P, color=junc_lane_color, lw=2.0, alpha=0.95, z=7)

                    if show_lane_boundaries:
                        lb = lane.get("left_boundary")
                        if lb and "x" in lb and "y" in lb:
                            lx = np.asarray(lb["x"], float)
                            ly = np.asarray(lb["y"], float)
                            if lx.size >= 2:
                                ax.plot(lx, ly, "--", color=junc_lane_color, lw=1.2, alpha=0.9, zorder=5)

                        rb = lane.get("right_boundary")
                        if rb and "x" in rb and "y" in rb:
                            rx = np.asarray(rb["x"], float)
                            ry = np.asarray(rb["y"], float)
                            if rx.size >= 2:
                                ax.plot(rx, ry, "--", color=junc_lane_color, lw=1.2, alpha=0.9, zorder=5)

    # -------------------------------
    # 3) Link lanes (on J_graph edges)
    # -------------------------------
    if show_link_lanes:
        if link_lane_palette is None:
            link_lane_palette = [
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                "#bcbd22", "#17becf"
            ]
        color_cycle = cycle(link_lane_palette)
        edge_color_map = {}

        for u, v, k, edata in J_graph.edges(keys=True, data=True):
            lanes = edata.get("lanes", []) or []
            if not lanes:
                continue

            is_super = bool(edata.get("is_superlink", False))
            if not is_super:
                link_seq = edata.get("link_seq")
                if link_seq is not None and len(link_seq) > 1:
                    is_super = True

            ekey = (u, v, k)
            if is_super:
                color = superlink_edge_color
                lw = superlink_edge_lw
            else:
                color = edge_color_map.get(ekey)
                lw = 2.0

            if color is None:
                color = next(color_cycle)
                edge_color_map[ekey] = color

            for lane in lanes:
                cl = lane.get("centerline", {})
                cx = np.asarray(cl.get("x", []), float)
                cy = np.asarray(cl.get("y", []), float)
                if cx.size < 2:
                    continue

                P = np.column_stack([cx, cy])
                if show_lane_centerlines:
                    ax.plot(cx, cy, "-", color=color, lw=lw, alpha=0.9, zorder=5)
                    _plot_arrow_on_poly(ax, P, color=color, lw=lw, alpha=0.9, z=6)

                if show_lane_boundaries:
                    for bkey in ("left_boundary", "right_boundary"):
                        b = lane.get(bkey)
                        if not b or "x" not in b or "y" not in b:
                            continue
                        bx = np.asarray(b["x"], float)
                        by = np.asarray(b["y"], float)
                        if bx.size < 2:
                            continue
                        ax.plot(bx, by, "--", color=color, lw=1.2, alpha=0.8, zorder=4)

    # -------------------------------
    # 0) Debug: boring nodes overlay
    # -------------------------------
    if show_boring_nodes and here_node_df is not None and boring_nodes is not None:
        bset = set(int(n) for n in boring_nodes)
        if len(bset) > 0 and ("node_id" in here_node_df.columns):
            dfb = here_node_df[here_node_df["node_id"].astype(int).isin(bset)]
            if not dfb.empty and ("x" in dfb.columns) and ("y" in dfb.columns):
                bx = dfb["x"].to_numpy(float)
                by = dfb["y"].to_numpy(float)
                ax.scatter(
                    bx, by,
                    s=boring_node_size,
                    c=boring_node_color,
                    alpha=0.8,
                    linewidths=0.0,
                    zorder=2,
                    label=f"boring nodes ({len(dfb)})",
                )
                if label_boring:
                    # label only a limited number
                    nlab = min(len(dfb), int(label_boring_limit))
                    for nid, x, y in dfb[["node_id","x","y"]].head(nlab).itertuples(index=False, name=None):
                        ax.text(float(x), float(y), str(int(nid)),
                                fontsize=7, alpha=0.8, zorder=3)
                        
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc="best")
    fig.tight_layout()
    return ax



import numpy as np
from typing import Dict, Any
# assumes: prepare_lane_groups_for_link, reorient_lane_groups_result_to_graph are available

def _poly_from_df_row_links(row) -> list[tuple[float, float]]:
    """Build polyline from links_df row (x_all / y_all)."""
    xs = row["x_all"]
    ys = row["y_all"]
    return list(zip(xs, ys))


def _is_same_orientation_simple(poly_graph, poly_df) -> bool:
    """
    Decide if poly_df has the same orientation as poly_graph by comparing
    proximity of graph start to df start vs df end.
    """
    if len(poly_graph) < 2 or len(poly_df) < 2:
        return True

    gx0, gy0 = poly_graph[0]
    df_start = poly_df[0]
    df_end   = poly_df[-1]

    d_start = (gx0 - df_start[0])**2 + (gy0 - df_start[1])**2
    d_end   = (gx0 - df_end[0])**2   + (gy0 - df_end[1])**2
    return d_start <= d_end

def attach_junction_lanes_to_graph_nodes(J_graph, J_list):
    """
    Attach JunctionNX + junction_lanes to J_graph nodes.

    NEW:
      - Attach direction-relevant CP sets:
          in_cps, out_cps, in_mask, out_mask
        (and optionally in_by_link/out_by_link if present)
      - Keep legacy in_points/out_points only if they exist, but do not rely on them for direction.
    """

    # jid -> JunctionNX
    jmap = {int(jnx.jid): jnx for jnx in (J_list or [])}

    def _as_int_list(x):
        if x is None:
            return []
        if isinstance(x, (list, tuple, set)):
            out = []
            for t in x:
                try:
                    out.append(int(t))
                except Exception:
                    pass
            return out
        try:
            return [int(x)]
        except Exception:
            return []

    for nid in J_graph.nodes:
        jid = int(nid)
        jnx = jmap.get(jid)
        if jnx is None:
            continue

        # Attach raw junction object
        J_graph.nodes[nid]["junction"] = jnx

        # edge_lanes was created elsewhere (junction internal lanes)
        edge_lanes = getattr(jnx, "edge_lanes", []) or []
        J_graph.nodes[nid]["junction_lanes"] = edge_lanes

        # --- NEW: attach computed CP direction info (CP ids, not lane-group IDs) ---
        J_graph.nodes[nid]["in_cps"]  = _as_int_list(getattr(jnx, "in_cps", []))
        J_graph.nodes[nid]["out_cps"] = _as_int_list(getattr(jnx, "out_cps", []))

        # masks aligned with node_ids (CCW order) if present
        if hasattr(jnx, "in_mask"):
            J_graph.nodes[nid]["in_mask"] = list(getattr(jnx, "in_mask") or [])
        if hasattr(jnx, "out_mask"):
            J_graph.nodes[nid]["out_mask"] = list(getattr(jnx, "out_mask") or [])

        # link-specific maps if you attached them earlier
        if hasattr(jnx, "in_by_link"):
            J_graph.nodes[nid]["in_by_link"] = getattr(jnx, "in_by_link") or {}
        if hasattr(jnx, "out_by_link"):
            J_graph.nodes[nid]["out_by_link"] = getattr(jnx, "out_by_link") or {}

        # Fields for lane connections
        if hasattr(jnx, "in_points"):
            J_graph.nodes[nid]["in_points"] = getattr(jnx, "in_points") or {}
        if hasattr(jnx, "out_points"):
            J_graph.nodes[nid]["out_points"] = getattr(jnx, "out_points") or {}
        
        # Junction type
        if hasattr(jnx, "junction_type"):
            J_graph.nodes[nid]["junction_type"] = getattr(jnx, "junction_type")

    return J_graph


def attach_link_attrs_to_jgraph_edges(J_graph, here_link_df):
    """
    Add 'road_data' dict to each edge with routing attrs, if link_id exists.
    Assumes here_link_df has columns from attach_routing_to_links_df().
    """
    if "link_id" not in here_link_df.columns:
        return J_graph

    # build quick lookup from df
    cols = ["functional_class","accessible_by","is_ramp","is_within_interchange","is_urban",
            "built_up_area_road","admin_area_id","admin_area_partition_id"]
    df = here_link_df.set_index("link_id")

    def _to_int_or_none(x):
        if x is None:
            return None
        if isinstance(x, numbers.Integral):  # handles int + numpy.int64 + etc.
            return int(x)
        try:
            return int(str(x).strip().strip("'").strip('"'))
        except Exception:
            return None

    def _is_nan(x):
        try:
            return isinstance(x, float) and math.isnan(x)
        except Exception:
            return False

    def _fill_road_data_from_row(road_data: dict, row, cols):
        for c in cols:
            val = row.get(c, None)
            if val is None or _is_nan(val) or str(val) == "nan":
                continue
            road_data[c] = val
        return road_data

    def _normalize_rd(rd):
        # for equality check: stable key order + JSON-like primitives
        if not isinstance(rd, dict):
            return {}
        return {k: rd[k] for k in sorted(rd.keys())}

    for u, v, key, data in J_graph.edges(keys=True, data=True):

        # ----------------------------
        # SUPERLINK: use link_seq
        # ----------------------------
        if bool(data.get("is_superlink", False)):

            seq = data.get("link_seq", []) or []
            seq_ids = []
            for lid in seq:
                lid_i = _to_int_or_none(lid)
                if lid_i is not None:
                    seq_ids.append(lid_i)

            rd = {}  # the single road_data we will attach

            # pick first available road data along the sequence
            picked = None
            for lid_i in seq_ids:
                if lid_i in df.index:
                    row = df.loc[lid_i]
                    rd = _fill_road_data_from_row({}, row, cols)
                    if rd:
                        picked = lid_i
                        break

            # optional: verify consistency (assumes "should be the same")
            # If you don't want warnings, delete this block.
            if picked is not None:
                ref = _normalize_rd(rd)
                for lid_i in seq_ids:
                    if lid_i == picked or lid_i not in df.index:
                        continue
                    other = _fill_road_data_from_row({}, df.loc[lid_i], cols)
                    if other and _normalize_rd(other) != ref:
                        print(f"[WARN] superlink road_data differs within link_seq. picked={picked}, differs_at={lid_i}")
                        break

            data["road_data"] = rd  # {} if none found
            continue

        # ----------------------------
        # NORMAL link: use link_id
        # ----------------------------
        lid_i = _to_int_or_none(data.get("link_id", None))
        if lid_i is None or lid_i not in df.index:
            data["road_data"] = {}
            continue

        row = df.loc[lid_i]

        road_data = data.get("road_data", None)
        if not isinstance(road_data, dict) or not road_data:
            road_data = {}

        road_data = _fill_road_data_from_row(road_data, row, cols)
        data["road_data"] = road_data

    return J_graph

def attach_lane_groups_to_junction_graph(
    J_graph,
    links_df,
    road,
    lg_data,
    *,
    strict=False,
):
    """
    For each edge (u,v) in J_graph with 'link_id', attach oriented lane groups
    based on HERE link_lane_group_refs and lg_data.

    After this, each edge has:
        data["lane_result"]   : the full payload (including graph_polyline_xy)
        data["link_lanes"]    : flattened list of lane dicts
    """
    # Build lookup: global_link_id (int) -> row in links_df
    links_df = links_df.copy()
    links_df["link_id_int"] = links_df["link_id"].astype(int)
    links_by_id = links_df.set_index("link_id_int")

    results_by_link = {}

    for u, v, key, data in J_graph.edges(keys=True, data=True):
        lid_raw = data.get("link_id", None)
        if lid_raw is None:
            if strict:
                raise ValueError(f"edge ({u},{v},{key}) has no 'link_id'")
            continue

        try:
            glid = int(lid_raw)
        except Exception:
            if strict:
                raise
            continue

        if glid not in links_by_id.index:
            if strict:
                raise KeyError(f"global link_id {glid} not found in links_df")
            continue

        row = links_by_id.loc[glid]

        # Polyline in graph direction (u->v): we use u_x/u_y from links_df
        x = np.asarray(row["u_x"], float)
        y = np.asarray(row["u_y"], float)
        poly_graph = list(zip(x, y))
        if len(poly_graph) < 2:
            if strict:
                raise ValueError(f"link {glid} has <2 points in u_x/u_y")
            continue

        # HERE lane_group references for this link
        refs_for_link = road.get("link_lane_group_refs", {}).get(glid, [])
        if not refs_for_link:
            continue

        llgr_payload = [{
            "link_local_ref": glid,
            "lane_group_references": refs_for_link,
        }]

        # Your existing helper for per-link lane groups
        res = prepare_lane_groups_for_link(
            link_id=glid,
            link_poly_xy=poly_graph,
            link_lane_group_refs=llgr_payload,
            lg_data=lg_data,
        )

        # Orient lane groups (centerline+boundaries) to match graph_polyline
        reorient_lane_groups_result_to_graph(res)

        res["u"] = int(u)
        res["v"] = int(v)
        res["graph_polyline_xy"] = poly_graph

        results_by_link[glid] = res

        # Flatten lanes for direct plotting / export
        flattened = []
        for g in res.get("ordered_groups", []):
            gid = str(g.get("lane_group_ref", g.get("group_id", "")))
            for lane in g.get("lanes", []):
                lane_copy = dict(lane)
                lane_copy["group_id"] = gid
                flattened.append(lane_copy)

        data["lane_result"] = res
        data["link_lanes"] = flattened

    return results_by_link

def build_junction_cp_sets_from_Jlist(J_list):
    """
    Build a compact {jid: {"in": {cp_id:[group_ids]}, "out": {cp_id:[group_ids]}}}
    from the existing JunctionNX objects, assuming we’ve already populated:
        jnx.in_points  : {cp_id -> [group_ids]}
        jnx.out_points : {cp_id -> [group_ids]}
    """
    junction_cp_sets = {}
    for jnx in (J_list or []):
        jid = int(getattr(jnx, "jid"))
        in_pts  = getattr(jnx, "in_points",  {}) or {}
        out_pts = getattr(jnx, "out_points", {}) or {}
        junction_cp_sets[jid] = {
            "in":  {int(cp): list(gids) for cp, gids in in_pts.items()},
            "out": {int(cp): list(gids) for cp, gids in out_pts.items()},
        }
    return junction_cp_sets

import numpy as np
from typing import Dict, Tuple, Any, Optional, List

def _euclid(p, q):
    return float(np.hypot(p[0] - q[0], p[1] - q[1]))


def trim_poly_between_points(poly_xy: np.ndarray,
                             start_xy: Tuple[float, float],
                             end_xy: Tuple[float, float]) -> np.ndarray:
    """
    Given a polyline (N,2) and two points in the same region,
    trim the polyline so that:
        new_poly[0]   == start_xy (projected onto the polyline),
        new_poly[-1]  == end_xy   (projected onto the polyline),
    by snapping to the nearest vertices and replacing endpoints.
    """
    if poly_xy.shape[0] < 2:
        return poly_xy.copy()

    # distances to start / end
    d_start = np.sum((poly_xy - np.asarray(start_xy))**2, axis=1)
    d_end   = np.sum((poly_xy - np.asarray(end_xy))**2,   axis=1)
    i0 = int(np.argmin(d_start))
    i1 = int(np.argmin(d_end))

    if i0 == i1:
        # degenerate: just keep that point twice
        return np.array([start_xy, end_xy], dtype=float)

    if i0 > i1:
        poly_xy = poly_xy[::-1, :]
        i0 = poly_xy.shape[0] - 1 - i0
        i1 = poly_xy.shape[0] - 1 - i1

    out = poly_xy[i0:i1+1, :].copy()
    out[0, :]  = start_xy
    out[-1, :] = end_xy
    return out


def choose_best_lane_for_group(
    lg_group: Dict[str, Any],
    cp_start_xy: Tuple[float, float],
    cp_end_xy: Tuple[float, float],
    tol: float = 50.0
) -> Optional[Tuple[Dict[str, Any], np.ndarray]]:
    """
    From a lane_group dict (as in lg_data["lane_groups"][gid]),
    choose the single lane whose endpoints best match the two CPs.
    Returns (lane_obj, oriented_poly_xy) or None.
    """
    best_lane = None
    best_poly = None
    best_cost = float("inf")

    for lane in lg_group.get("lanes", []) or []:
        cx = np.asarray(lane["centerline"]["x"], float)
        cy = np.asarray(lane["centerline"]["y"], float)
        if cx.size < 2:
            continue
        poly = np.column_stack([cx, cy])

        p0 = poly[0, :]
        p1 = poly[-1, :]

        # cost if used as-is
        cost_fwd = _euclid(cp_start_xy, p0) + _euclid(cp_end_xy, p1)
        # cost if reversed
        cost_rev = _euclid(cp_start_xy, p1) + _euclid(cp_end_xy, p0)

        if cost_fwd < cost_rev:
            cost = cost_fwd
            poly_oriented = poly
        else:
            cost = cost_rev
            poly_oriented = poly[::-1, :]

        if cost < best_cost:
            best_cost = cost
            best_lane = lane
            best_poly = poly_oriented

    if best_lane is None:
        return None

    # Optional sanity threshold
    if best_cost > tol:
        # too far; likely mismatch
        return None

    return best_lane, best_poly


import numpy as np
from typing import Dict, Tuple

import numpy as np

# ------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------

def _norm_int(x):
    """Robust normalize to Python int, or None."""
    if x is None:
        return None
    s = str(x).strip().strip("'").strip('"')
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None

def _euclid(p, q):
    return float(np.hypot(p[0] - q[0], p[1] - q[1]))

def orient_polyline_by_nodes(poly_xy, start_xy, end_xy,
                             tol_start=5.0, tol_end=5.0):
    """
    Given a polyline poly_xy (Nx2) and desired endpoints start_xy, end_xy,
    orient the polyline such that poly[0]~start, poly[-1]~end. Reverse if needed.
    Returns (poly_oriented, matched:bool, reversed_flag:bool)
    """
    poly_xy = np.asarray(poly_xy, float)
    if poly_xy.shape[0] < 2:
        return poly_xy, False, False

    d0_start = _euclid(poly_xy[0], start_xy)
    dN_start = _euclid(poly_xy[-1], start_xy)
    d0_end   = _euclid(poly_xy[0], end_xy)
    dN_end   = _euclid(poly_xy[-1], end_xy)

    # Two possibilities:
    # 1) poly[0] ~ start, poly[-1] ~ end
    err_fwd  = d0_start + dN_end
    # 2) reversed: poly[0] ~ end, poly[-1] ~ start
    err_rev  = d0_end + dN_start

    if err_fwd <= err_rev:
        # keep orientation
        ok = (d0_start <= tol_start) and (dN_end <= tol_end)
        return poly_xy, ok, False
    else:
        # reverse
        poly_rev = poly_xy[::-1].copy()
        ok = (dN_start <= tol_start) and (d0_end <= tol_end)
        return poly_rev, ok, True

def trim_poly_between_points(poly_xy, start_xy, end_xy):
    """
    Trim polyline between the point closest to start_xy and point closest to end_xy
    along the already oriented polyline. Assumes poly_xy is oriented start→end.
    """
    poly_xy = np.asarray(poly_xy, float)
    if poly_xy.shape[0] < 2:
        return poly_xy

    ds = np.hypot(poly_xy[:, 0] - start_xy[0],
                  poly_xy[:, 1] - start_xy[1])
    de = np.hypot(poly_xy[:, 0] - end_xy[0],
                  poly_xy[:, 1] - end_xy[1])

    i0 = int(np.argmin(ds))
    i1 = int(np.argmin(de))

    if i0 <= i1:
        return poly_xy[i0:i1+1, :]
    else:
        # worst case: order is swapped; still return continuous segment
        return poly_xy[i1:i0+1, :]

def choose_best_lane_for_group(lg_group, p_start, p_end,
                               tol_start=5.0, tol_end=5.0):
    """
    Among all lanes in a lane_group, choose the lane whose centerline
    best matches p_start→p_end (using orient_polyline_by_nodes).
    Returns (lane_obj, oriented_poly_xy) or None.
    """
    lanes = lg_group.get("lanes", []) or []
    best = None
    best_err = float("inf")

    for lane in lanes:
        cx = np.asarray(lane["centerline"]["x"], float)
        cy = np.asarray(lane["centerline"]["y"], float)
        if cx.size < 2:
            continue

        poly = np.column_stack([cx, cy])
        poly_oriented, matched, _rev = orient_polyline_by_nodes(
            poly, p_start, p_end,
            tol_start=tol_start, tol_end=tol_end
        )
        if not matched:
            continue

        err = _euclid(poly_oriented[0], p_start) + _euclid(poly_oriented[-1], p_end)
        if err < best_err:
            best_err = err
            best = (lane, poly_oriented)

    return best


# ------------------------------------------------------------
# Main rewritten function
# ------------------------------------------------------------
def split_and_attach_link_lanes(
    J_graph,
    links_df,
    road,
    lg_data,
    lane_node_xy,
    tol_cp=25.0,
    tol_lane_end=5.0,
):
    """
    For every original HERE edge u-v in J_graph:
      1. Determine whether a direction u->v exists.
      2. Determine whether a direction v->u exists.
      3. Remove all original edges.
      4. Insert new directed edges (per existing direction),
         with correctly oriented lanes.

    This implements:
        G_keep = G_link - (G_intersect_u ∪ G_intersect_v)
    and orients lanes OUT->IN according to junction out_points / in_points.
    """

    import networkx as nx
    from itertools import cycle

    new_edges = []   # (src, dst, lanes, lid_int, base_attrs)

    # ----- lane-node coordinates -----
    node_xy = {}
    for nid, xy in (lane_node_xy or {}).items():
        try:
            node_xy[int(nid)] = (float(xy[0]), float(xy[1]))
        except Exception:
            continue

    # ----- normalize link dataframe -----
    df_links = links_df.copy()
    df_links["link_id_int"] = (
        df_links["link_id"]
        .astype(str)
        .str.strip().str.strip("'").str.strip('"')
        .astype("int64")
    )
    links_by_id = df_links.set_index("link_id_int")

    # ----- lane_group map -----
    lg_map = {}
    raw_lgmap = lg_data.get("lane_groups", {}) or {}
    for gid_raw, gval in raw_lgmap.items():
        try:
            gid = int(str(gid_raw).strip().strip("'").strip('"'))
            lg_map[gid] = gval
        except Exception:
            continue

    # ----- link -> lane-group IDs -----
    link2gids = {}
    raw_l2g = road.get("link_lane_group_refs", {}) or {}
    for lid_raw, refs in raw_l2g.items():
        try:
            lid = int(str(lid_raw).strip().strip("'").strip('"'))
        except Exception:
            continue
        gids = set()
        for entry in (refs or []):
            if isinstance(entry, dict):
                r = entry.get("lane_group_ref") or {}
                gid_raw = r.get("lane_group_id") or r.get("id")
            else:
                gid_raw = entry
            try:
                g = int(str(gid_raw).strip().strip("'").strip('"'))
                gids.add(g)
            except Exception:
                continue
        if gids:
            link2gids[lid] = gids

    # --------------------------------------------
    # STEP 1: iterate over existing edges, collect new ones
    # --------------------------------------------
    edges_original = list(J_graph.edges(keys=True, data=True))

    for u, v, k, edata in edges_original:

        lid_raw = edata.get("link_id")
        try:
            lid = int(str(lid_raw).strip().strip("'").strip('"'))
        except Exception:
            continue

        if lid not in links_by_id.index:
            continue
        if lid not in link2gids:
            continue

        G_link = link2gids[lid]
        if not G_link:
            continue

        # read junction objects
        jd_u = J_graph.nodes[u].get("junction")
        jd_v = J_graph.nodes[v].get("junction")
        if jd_u is None or jd_v is None:
            continue

        u_outpts = J_graph.nodes[u].get("out_points", {}) or {}
        v_inpts  = J_graph.nodes[v].get("in_points",  {}) or {}
        v_outpts = J_graph.nodes[v].get("out_points", {}) or {}
        u_inpts  = J_graph.nodes[u].get("in_points",  {}) or {}

        # lane-group sets
        LG_u_out = {
            int(g)
            for _, gids in u_outpts.items()
            for g in gids
            if int(g) in G_link
        }
        LG_v_in = {
            int(g)
            for _, gids in v_inpts.items()
            for g in gids
            if int(g) in G_link
        }
        LG_v_out = {
            int(g)
            for _, gids in v_outpts.items()
            for g in gids
            if int(g) in G_link
        }
        LG_u_in = {
            int(g)
            for _, gids in u_inpts.items()
            for g in gids
            if int(g) in G_link
        }

        dir_u_v_exists = bool(LG_u_out and LG_v_in)
        dir_v_u_exists = bool(LG_v_out and LG_u_in)

        # -------------- helper: build lanes for one direction --------------
        def build_direction(src, dst, LG_src_out, LG_dst_in):

            if not LG_src_out or not LG_dst_in:
                return []

            G_intersect_src = set(LG_src_out)
            G_intersect_dst = set(LG_dst_in)

            # lane groups to keep 
            G_keep = G_link # G_link.difference(G_intersect_src.union(G_intersect_dst))
            if not G_keep:
                return []

            # extract CPs from src.out_points
            src_out_cps = []
            OP_src = J_graph.nodes[src].get("out_points", {}) or {}
            for cp_raw, gids in OP_src.items():
                cp = int(cp_raw)
                if cp not in node_xy:
                    continue
                if any(int(g) in G_intersect_src for g in gids):
                    src_out_cps.append(cp)
            src_out_cps = sorted(set(src_out_cps))

            # extract CPs from dst.in_points
            dst_in_cps = []
            IP_dst = J_graph.nodes[dst].get("in_points", {}) or {}
            for cp_raw, gids in IP_dst.items():
                cp = int(cp_raw)
                if cp not in node_xy:
                    continue
                if any(int(g) in G_intersect_dst for g in gids):
                    dst_in_cps.append(cp)
            dst_in_cps = sorted(set(dst_in_cps))

            if not src_out_cps or not dst_in_cps:
                return []

            lane_list = []

            # For each lane-group in G_keep
            for gid in sorted(G_keep):
                lg = lg_map.get(gid)
                if not lg:
                    continue

                for lane in (lg.get("lanes", []) or []):
                    cx = np.asarray(lane["centerline"]["x"], float)
                    cy = np.asarray(lane["centerline"]["y"], float)
                    if cx.size < 2:
                        continue
                    poly = np.column_stack([cx, cy])

                    # best OUT CP
                    best_out = None
                    best_out_err = float("inf")
                    for cp in src_out_cps:
                        p = node_xy[cp]
                        d = min(
                            np.hypot(poly[0, 0] - p[0], poly[0, 1] - p[1]),
                            np.hypot(poly[-1, 0] - p[0], poly[-1, 1] - p[1]),
                        )
                        if d < best_out_err and d <= tol_cp:
                            best_out_err = d
                            best_out = cp

                    # best IN CP
                    best_in = None
                    best_in_err = float("inf")
                    for cp in dst_in_cps:
                        p = node_xy[cp]
                        d = min(
                            np.hypot(poly[0, 0] - p[0], poly[0, 1] - p[1]),
                            np.hypot(poly[-1, 0] - p[0], poly[-1, 1] - p[1]),
                        )
                        if d < best_in_err and d <= tol_cp:
                            best_in_err = d
                            best_in = cp

                    if best_out is None or best_in is None:
                        continue

                    p_out = node_xy[best_out]
                    p_in  = node_xy[best_in]

                    # Orientation for centerline
                    poly_oriented, matched, reversed_flag = orient_polyline_by_nodes(
                        poly,
                        p_out,
                        p_in,
                        tol_start=tol_lane_end,
                        tol_end=tol_lane_end,
                    )
                    if not matched:
                        continue

                    lane_entry = {
                        "group_id": gid,
                        "lane_no": int(lane.get("lane_number", 0)),
                        "start_cp": int(best_out),
                        "end_cp":   int(best_in),
                        "direction": f"{src}->{dst}",
                        "centerline": {
                            "x": poly_oriented[:, 0].tolist(),
                            "y": poly_oriented[:, 1].tolist(),
                        },
                    }

                    # boundaries: only flip if centerline reversed
                    for bkey in ("left_boundary", "right_boundary"):
                        b = lane.get(bkey)
                        if not b:
                            continue
                        bx = np.asarray(b.get("x", []), float)
                        by = np.asarray(b.get("y", []), float)
                        if bx.size < 2:
                            continue
                        if reversed_flag:
                            bx = bx[::-1]
                            by = by[::-1]
                        lane_entry[bkey] = {
                            "x": bx.tolist(),
                            "y": by.tolist(),
                            "markings": b.get("markings", []),   # NEW
                        }

                    lane_list.append(lane_entry)

            return lane_list

        # capture the base attributes of the original edge (for geometry etc.)
        base_attrs = dict(edata)

        # build both directions
        if dir_u_v_exists:
            lanes_uv = build_direction(u, v, LG_u_out, LG_v_in)
            if lanes_uv:
                new_edges.append((u, v, lanes_uv, lid, base_attrs))

        if dir_v_u_exists:
            lanes_vu = build_direction(v, u, LG_v_out, LG_u_in)
            if lanes_vu:
                new_edges.append((v, u, lanes_vu, lid, base_attrs))

    # --------------------------------------------
    # STEP 2: remove original edges
    # --------------------------------------------
    J_graph.remove_edges_from(list(J_graph.edges(keys=True)))

    # --------------------------------------------
    # STEP 3: add new directed edges with lanes
    # --------------------------------------------
    next_key = 0
    for (src, dst, lanes, lid, base_attrs) in new_edges:
        # start from original attributes but overwrite some keys
        attrs = dict(base_attrs)
        attrs["link_id"]      = lid
        attrs["lanes"]        = lanes
        attrs["is_superlink"] = False
        attrs["link_seq"]     = [int(lid)]

        J_graph.add_edge(src, dst, key=next_key, **attrs)
        next_key += 1

    return J_graph


import numpy as np

# ---------------------------------------------------------------------
# Distance helper
# ---------------------------------------------------------------------
def _min_dist_to_cloud(p, cloud_arr):
    """
    p         : (2,) array-like
    cloud_arr : (N,2) ndarray of points

    Returns min Euclidean distance between p and any point in cloud_arr.
    If cloud_arr is empty, returns +inf.
    """
    if cloud_arr is None:
        return float("inf")
    cloud_arr = np.asarray(cloud_arr, float)
    if cloud_arr.size == 0:
        return float("inf")
    dx = cloud_arr[:, 0] - p[0]
    dy = cloud_arr[:, 1] - p[1]
    return float(np.hypot(dx, dy).min())
    

# ---------------------------------------------------------------------
# Chain orientation helper
# ---------------------------------------------------------------------
def orient_lane_chain(chain_segments, src_cloud_arr, dst_cloud_arr,
                      tol_src=np.inf, tol_dst=np.inf):
    """
    Orient a *sequence* of lane segments to form a continuous chain
    from the source junction towards the destination junction.

    Parameters
    ----------
    chain_segments : list of dict
        Each dict must contain:
            {
              "center_xy": (M_i, 2) ndarray,
              "lane":      <original lane dict>,
              "group_id":  int,
            }

    src_cloud_arr : (N_s,2) ndarray
        All OUT CP positions of the source junction for this direction.

    dst_cloud_arr : (N_d,2) ndarray
        All IN CP positions of the destination junction for this direction.

    tol_src, tol_dst : float
        Optional sanity thresholds; currently not used to reject segments,
        but kept for future checks.

    Returns
    -------
    oriented_segments : list of dict
        Same length as chain_segments (minus any degenerate ones), each:
            {
              "center_xy": (M_i,2) ndarray oriented along chain,
              "reversed":  bool,   # True if we reversed the original polyline
              "lane":      <original lane dict>,
              "group_id":  int,
            }
    """
    if not chain_segments:
        return []

    # ---- 0) Normalize clouds ----
    src_cloud_arr = np.asarray(src_cloud_arr, float)
    dst_cloud_arr = np.asarray(dst_cloud_arr, float)

    # ---- 1) Sort segments by how "close" they are to the source junction ----
    # We use the minimum distance between *either* endpoint and src_cloud.
    scored_segments = []
    for seg in chain_segments:
        P = np.asarray(seg["center_xy"], float)
        if P.shape[0] < 2:
            continue
        p0 = P[0]
        p1 = P[-1]
        d0_src = _min_dist_to_cloud(p0, src_cloud_arr)
        d1_src = _min_dist_to_cloud(p1, src_cloud_arr)
        d_src  = min(d0_src, d1_src)
        scored_segments.append((d_src, seg))

    if not scored_segments:
        return []

    scored_segments.sort(key=lambda t: t[0])
    ordered_segments = [seg for _, seg in scored_segments]

    oriented = []

    # ---- 2) First segment: orient to face away from src_cloud ----
    first = ordered_segments[0]
    P = np.asarray(first["center_xy"], float)
    p0 = P[0]
    p1 = P[-1]

    d0_src = _min_dist_to_cloud(p0, src_cloud_arr)
    d1_src = _min_dist_to_cloud(p1, src_cloud_arr)

    if d1_src < d0_src:
        P = P[::-1, :]
        reversed_flag = True
    else:
        reversed_flag = False

    oriented.append({
        "center_xy": P,
        "reversed":  reversed_flag,
        "lane":      first["lane"],
        "group_id":  first["group_id"],
    })

    current_end = P[-1, :]

    # ---- 3) Subsequent segments: orient to connect to current_end ----
    for seg in ordered_segments[1:]:
        Q = np.asarray(seg["center_xy"], float)
        if Q.shape[0] < 2:
            continue

        q0 = Q[0]
        q1 = Q[-1]

        d0 = np.hypot(q0[0] - current_end[0], q0[1] - current_end[1])
        d1 = np.hypot(q1[0] - current_end[0], q1[1] - current_end[1])

        if d1 < d0:
            Q = Q[::-1, :]
            rev = True
        else:
            rev = False

        oriented.append({
            "center_xy": Q,
            "reversed":  rev,
            "lane":      seg["lane"],
            "group_id":  seg["group_id"],
        })
        current_end = Q[-1, :]

    # ---- 4) Optional consistency check with dst_cloud ----
    # (for now we just compute, you can add assertions/logging if you want)
    if dst_cloud_arr.size:
        _ = _min_dist_to_cloud(current_end, dst_cloud_arr)

    return oriented

def build_lane_chain_segments_for_direction(
    G_keep,
    lg_map,
    node_xy,
    src_out_cps,
    dst_in_cps,
    tol_cp,
):
    if not G_keep:
        return []

    # Build CP clouds
    src_cloud = np.array([node_xy[cp] for cp in src_out_cps if cp in node_xy], dtype=float)
    dst_cloud = np.array([node_xy[cp] for cp in dst_in_cps if cp in node_xy], dtype=float)
    if src_cloud.size == 0 or dst_cloud.size == 0:
        return []

    def _iter_lg_variants(gid):
        lg = lg_map.get(gid)
        if lg is None:
            return []
        # support either dict or list[dict]
        return lg if isinstance(lg, list) else [lg]

    chain_segments = []

    for gid in sorted(G_keep):
        for lg in _iter_lg_variants(gid):
            lanes = (lg.get("lanes") or [])
            for lane in lanes:
                cl = lane.get("centerline") or {}
                cx = np.asarray(cl.get("x", []), float)
                cy = np.asarray(cl.get("y", []), float)
                if cx.size < 2:
                    continue

                poly = np.column_stack([cx, cy])
                p0 = poly[0]
                p1 = poly[-1]

                # distances of each endpoint to each cloud
                d0_src = _min_dist_to_cloud(p0, src_cloud)
                d1_src = _min_dist_to_cloud(p1, src_cloud)
                d0_dst = _min_dist_to_cloud(p0, dst_cloud)
                d1_dst = _min_dist_to_cloud(p1, dst_cloud)

                # enforce *bridge* constraint:
                # either (p0 near src AND p1 near dst) OR (p1 near src AND p0 near dst)
                forward_ok  = (d0_src <= tol_cp) and (d1_dst <= tol_cp)
                reverse_ok  = (d1_src <= tol_cp) and (d0_dst <= tol_cp)
                if not (forward_ok or reverse_ok):
                    continue

                # If both ok (clouds overlap), choose the more consistent one
                # based on total endpoint-to-cloud distance.
                if forward_ok and reverse_ok:
                    cost_f = d0_src + d1_dst
                    cost_r = d1_src + d0_dst
                    prefer_reversed = (cost_r < cost_f)
                else:
                    prefer_reversed = reverse_ok  # if only reverse_ok, we prefer reversing

                chain_segments.append({
                    "center_xy": poly,
                    "lane": lane,
                    "group_id": gid,
                    # optional hint for later (you can ignore if orient_lane_chain already handles it)
                    "prefer_reversed": bool(prefer_reversed),
                    "d_endpoints": (float(d0_src), float(d1_src), float(d0_dst), float(d1_dst)),
                })

    return chain_segments

import numpy as np
import pandas as pd

def split_and_attach_superlink_lanes(
    J_graph,
    superlinks_df,
    road,
    lg_data,
    lane_node_xy,
    dead_ends=None,
    tol_cp=25.0,
    tol_lane_end=5.0,   # kept for signature compatibility (not used below)
):
    """
    Attach lane geometry for *superlinks* (junction -> ... -> junction)
    to J_graph as additional directed edges.

    Each superlink row/dict must contain:
        'u'        : upstream junction id (global node id)
        'v'        : downstream junction id
        'node_seq' : [u, b1, ..., bk, v]
        'link_seq' : [link_id_0, ...]   (HERE link ids)
    Optionally:
        'dir_seq'  : list of traversal directions for each link in link_seq
                    (+1 = as stored, -1 = reversed)  [ONLY used for ordering]

    Key fix in this version:
      - For each candidate lane polyline, enforce direction using CP clouds:
            start near src_out_cps, end near dst_in_cps
        If not, reverse the polyline (and boundaries) deterministically.

      - If corridor backbone can be built from road link polylines,
        order lane entries by projected arclength along the backbone.
    """

    # ----------------------------
    # Normalize superlinks to DataFrame
    # ----------------------------
    if isinstance(superlinks_df, list):
        if not superlinks_df:
            return J_graph
        s_df = pd.DataFrame(superlinks_df)
    else:
        s_df = superlinks_df
        if s_df is None or getattr(s_df, "empty", False):
            return J_graph

    # ----------------------------
    # lane-node coordinates: int -> (x,y)
    # ----------------------------
    node_xy = {}
    for nid, xy in (lane_node_xy or {}).items():
        try:
            node_xy[int(nid)] = (float(xy[0]), float(xy[1]))
        except Exception:
            continue

    # ----------------------------
    # lane-group map: gid(int) -> lane_group_dict
    # ----------------------------
    lg_map = {}
    raw_lgmap = lg_data.get("lane_groups", {}) or {}
    for gid_raw, gval in raw_lgmap.items():
        try:
            gid = int(str(gid_raw).strip().strip("'").strip('"'))
        except Exception:
            continue
        lg_map[gid] = gval

    # ----------------------------
    # link -> lane-group IDs
    # ----------------------------
    link2gids = {}
    raw_l2g = road.get("link_lane_group_refs", {}) or {}
    for lid_raw, refs in raw_l2g.items():
        try:
            lid = int(str(lid_raw).strip().strip("'").strip('"'))
        except Exception:
            continue
        gids = set()
        for entry in (refs or []):
            if isinstance(entry, dict):
                r = entry.get("lane_group_ref") or {}
                gid_raw2 = r.get("lane_group_id") or r.get("id")
            else:
                gid_raw2 = entry
            try:
                g = int(str(gid_raw2).strip().strip("'").strip('"'))
                gids.add(g)
            except Exception:
                continue
        if gids:
            link2gids[lid] = gids

    def _road_node_xy(road, nid):
        """Try to fetch ENU x,y from road['nodes'] dict; fallback None."""
        if road is None:
            return None
        nd = (road.get("nodes") or {}).get(int(nid), None)
        if nd is None:
            nd = (road.get("nodes") or {}).get(str(nid), None)
        if not isinstance(nd, dict):
            return None
        x = nd.get("x", None)
        y = nd.get("y", None)
        if x is None or y is None:
            return None
        try:
            return float(x), float(y)
        except Exception:
            return None

    def _ensure_deadend_node(J_graph, nid, road=None):
        """Create a J_graph node compatible with junction nodes (minimal fields)."""
        nid = int(nid)
        if nid in J_graph.nodes:
            # ensure required keys exist even if node existed
            n = J_graph.nodes[nid]
            n.setdefault("in_points", {})
            n.setdefault("out_points", {})
            n.setdefault("in_cps", [])
            n.setdefault("out_cps", [])
            n.setdefault("in_mask", [])
            n.setdefault("out_mask", [])
            n.setdefault("itype", n.get("itype", "DeadEnd"))
            n.setdefault("n_legs", n.get("n_legs", 1))
            n.setdefault("radius_m", n.get("radius_m", 0.0))
            # centroid duplicates
            if "x" in n and "centroid_x" not in n: n["centroid_x"] = n["x"]
            if "y" in n and "centroid_y" not in n: n["centroid_y"] = n["y"]
            return

        xy = _road_node_xy(road, nid)
        if xy is None:
            # if no road xy, still create node with placeholders
            x = y = 0.0
        else:
            x, y = xy

        # If JunctionNX class is available in your scope, instantiate it.
        # Otherwise store None and keep scalar fields.
        try:
            j = JunctionNX(jid=nid, itype="DeadEnd", n_legs=1, centroid=(x, y), radius_m=0.0)
        except Exception:
            j = None

        J_graph.add_node(
            nid,
            junction=j,
            x=float(x), y=float(y),
            centroid_x=float(x), centroid_y=float(y),
            itype="DeadEnd",
            n_legs=1,
            radius_m=0.0,
            # CP containers (empty by default)
            in_cps=[],
            out_cps=[],
            in_mask=[],
            out_mask=[],
            in_points={},   # cp -> gids
            out_points={},  # cp -> gids
            # optional bookkeeping
            is_dead_end=True,
        )

    def lane_groups_for_link_seq(link_seq):
        gset = set()
        for lid_raw in (link_seq or []):
            try:
                lid = int(str(lid_raw).strip().strip("'").strip('"'))
            except Exception:
                continue
            gset.update(link2gids.get(lid, set()))
        return gset

    # ----------------------------
    # Helpers: distances & direction enforcement
    # ----------------------------
    def _mindist_point_to_cloud(p_xy, cloud_xy):
        if cloud_xy is None or cloud_xy.size == 0:
            return float("inf")
        d = np.hypot(cloud_xy[:, 0] - float(p_xy[0]), cloud_xy[:, 1] - float(p_xy[1]))
        return float(d.min()) if d.size else float("inf")

    def _enforce_src_to_dst_direction(P, src_cloud, dst_cloud):
        """
        Force polyline P to go from src_cloud to dst_cloud.
        Returns (P_oriented, flipped_bool)
        """
        if P is None or len(P) < 2:
            return P, False

        p0 = P[0]
        p1 = P[-1]

        d0s = _mindist_point_to_cloud(p0, src_cloud)
        d1s = _mindist_point_to_cloud(p1, src_cloud)
        d0d = _mindist_point_to_cloud(p0, dst_cloud)
        d1d = _mindist_point_to_cloud(p1, dst_cloud)

        # Compare the two assignments:
        #  forward cost: start~src + end~dst
        #  reverse cost: start~dst + end~src  (equivalently end~src + start~dst)
        forward = d0s + d1d
        reverse = d1s + d0d

        if reverse < forward:
            return P[::-1].copy(), True
        return P, False

    # ----------------------------
    # Optional backbone for ordering along the superlink
    # ----------------------------
    def _try_get_link_poly_xy(road, lid):
        """Return Nx2 ENU polyline for a road link if available, else None."""
        links = road.get("links") or {}
        L = links.get(lid, None)
        if L is None:
            L = links.get(str(lid), None)
        if not isinstance(L, dict):
            return None
        x = np.asarray(L.get("x", []), float)
        y = np.asarray(L.get("y", []), float)
        if x.size >= 2 and y.size == x.size:
            return np.column_stack([x, y])
        return None

    def _superlink_backbone_xy(road, link_seq, dir_seq):
        """
        Build an ENU backbone polyline from concatenated road link polylines.
        Requires road["links"][lid]["x","y"] (already ENU).
        If missing, returns None.
        """
        if not link_seq:
            return None
        if not dir_seq or len(dir_seq) != len(link_seq):
            return None

        pts = []
        last = None
        for lid_raw, d in zip(link_seq, dir_seq):
            try:
                lid = int(str(lid_raw).strip().strip("'").strip('"'))
            except Exception:
                continue
            P = _try_get_link_poly_xy(road, lid)
            if P is None or len(P) < 2:
                return None  # if any segment missing, skip ordering
            if int(d) < 0:
                P = P[::-1].copy()

            if last is not None and np.hypot(*(last - P[0])) < 1e-6:
                pts.extend(P[1:])
            else:
                pts.extend(P)
            last = P[-1]

        if len(pts) < 2:
            return None
        return np.asarray(pts, float)

    def _project_to_backbone_s(p_xy, Xb, Yb):
        # uses your existing function; expected signature:
        # _point_polyline_metrics(px,py, X, Y) -> (..., s_abs)
        _, _, _, _, _, s_abs = _point_polyline_metrics(float(p_xy[0]), float(p_xy[1]), Xb, Yb)
        return float(s_abs)

    # ----------------------------
    # Build lanes for one direction on one superlink
    # ----------------------------
    def build_direction_for_superlink(u, v, G_link, LG_src_out, LG_dst_in, backbone_xy=None):
        if not LG_src_out or not LG_dst_in:
            return []

        # Lane-groups to keep: those on the corridor links but not "inside" the endpoints
        G_intersect_src = set(LG_src_out)
        G_intersect_dst = set(LG_dst_in)
        G_keep = set(G_link).difference(G_intersect_src.union(G_intersect_dst))
        if not G_keep:
            return []

        # relevant CPs at src OUT and dst IN
        src_out_cps = []
        OP_src = J_graph.nodes[u].get("out_points", {}) or {}
        for cp_raw, gids in OP_src.items():
            try:
                cp = int(cp_raw)
            except Exception:
                continue
            if cp not in node_xy:
                continue
            try:
                if any(int(g) in G_intersect_src for g in gids):
                    src_out_cps.append(cp)
            except Exception:
                continue
        src_out_cps = sorted(set(src_out_cps))

        dst_in_cps = []
        IP_dst = J_graph.nodes[v].get("in_points", {}) or {}
        for cp_raw, gids in IP_dst.items():
            try:
                cp = int(cp_raw)
            except Exception:
                continue
            if cp not in node_xy:
                continue
            try:
                if any(int(g) in G_intersect_dst for g in gids):
                    dst_in_cps.append(cp)
            except Exception:
                continue
        dst_in_cps = sorted(set(dst_in_cps))

        if not src_out_cps or not dst_in_cps:
            return []

        # Build chain segments from kept groups (your existing helper)
        chain_segments = build_lane_chain_segments_for_direction(
            G_keep,
            lg_map,
            node_xy,
            src_out_cps,
            dst_in_cps,
            tol_cp,
        )
        if not chain_segments:
            return []

        # Orient the chain roughly (your existing helper)
        src_cloud = np.array([node_xy[cp] for cp in src_out_cps], float)
        dst_cloud = np.array([node_xy[cp] for cp in dst_in_cps], float)

        oriented_segments = orient_lane_chain(
            chain_segments,
            src_cloud,
            dst_cloud,
            tol_src=tol_cp,
            tol_dst=tol_cp,
        )
        if not oriented_segments:
            return []

        # For stable lane_entry start/end CP selection
        def _nearest_cp(point_xy, cp_list):
            px, py = float(point_xy[0]), float(point_xy[1])
            best = None
            bestd = float("inf")
            for cp in cp_list:
                cx, cy = node_xy[int(cp)]
                d = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                if d < bestd:
                    bestd = d
                    best = int(cp)
            return best, bestd

        # Optional corridor-ordering setup
        if backbone_xy is not None and len(backbone_xy) >= 2:
            Xb = backbone_xy[:, 0]
            Yb = backbone_xy[:, 1]
        else:
            Xb = Yb = None

        lane_entries = []
        for seg in oriented_segments:
            P = seg.get("center_xy", None)
            if P is None or len(P) < 2:
                continue
            rev = bool(seg.get("reversed", False))
            lane = seg.get("lane", {}) or {}
            gid = int(seg.get("group_id", 0))

            # === HARD ENFORCEMENT: src->dst using CP clouds ===
            P2, flipped = _enforce_src_to_dst_direction(P, src_cloud, dst_cloud)
            if flipped:
                rev = not rev  # keep boundary reversal consistent

            # choose start/end CPs
            start_cp, d0 = _nearest_cp(P2[0],  src_out_cps)
            end_cp,   d1 = _nearest_cp(P2[-1], dst_in_cps)
            if start_cp is None or end_cp is None:
                continue
            if d0 > tol_cp or d1 > tol_cp:
                continue

            lane_entry = {
                "group_id": int(gid),
                "lane_no": int(lane.get("lane_number", lane.get("lane_no", 0)) or 0),
                "start_cp": int(start_cp),
                "end_cp":   int(end_cp),
                "direction": f"{int(u)}->{int(v)}",
                "centerline": {
                    "x": P2[:, 0].tolist(),
                    "y": P2[:, 1].tolist(),
                },
            }

            # boundaries (reverse if needed)
            for bkey in ("left_boundary", "right_boundary"):
                b = lane.get(bkey)
                if not b:
                    continue
                bx = np.asarray(b.get("x", []), float)
                by = np.asarray(b.get("y", []), float)
                if bx.size < 2:
                    continue
                if rev:
                    bx = bx[::-1]
                    by = by[::-1]
                lane_entry[bkey] = {"x": bx.tolist(), "y": by.tolist()}

            # optional ordering key along backbone
            if Xb is not None:
                try:
                    sm = 0.5 * (
                        _project_to_backbone_s(P2[0],  Xb, Yb) +
                        _project_to_backbone_s(P2[-1], Xb, Yb)
                    )
                    lane_entry["_s_mid"] = float(sm)
                except Exception:
                    pass

            lane_entries.append(lane_entry)

        # corridor ordering if available
        if lane_entries and any("_s_mid" in d for d in lane_entries):
            lane_entries.sort(key=lambda d: d.get("_s_mid", 0.0))
            for d in lane_entries:
                d.pop("_s_mid", None)

        return lane_entries

    # ----------------------------
    # MAIN LOOP
    # ----------------------------
    try:
        next_key = (max((k for *_, k in J_graph.edges(keys=True)), default=-1) + 1)
    except Exception:
        next_key = 0

    dead_ends_set = set(dead_ends or [])
    # ensure endpoints exist for any superlink
    for _, row in s_df.iterrows():
        u = int(row["u"]); v = int(row["v"])
        if u not in J_graph.nodes:
            # if u is a dead end, create dead-end node; else create minimal placeholder
            if u in dead_ends_set:
                _ensure_deadend_node(J_graph, u, road=road)
            else:
                _ensure_deadend_node(J_graph, u, road=road)  # safe: creates "DeadEnd" but ok if you want a different tag
                J_graph.nodes[u]["itype"] = "Unknown"
                J_graph.nodes[u]["is_dead_end"] = False

        if v not in J_graph.nodes:
            if v in dead_ends_set:
                _ensure_deadend_node(J_graph, v, road=road)
            else:
                _ensure_deadend_node(J_graph, v, road=road)
                J_graph.nodes[v]["itype"] = "Unknown"
                J_graph.nodes[v]["is_dead_end"] = False


    for _, row in s_df.iterrows():
        ju = int(row["u"])
        jv = int(row["v"])

        if ju not in J_graph.nodes or jv not in J_graph.nodes:
            continue

        link_seq = row.get("link_seq", []) or []
        if not link_seq:
            continue

        # Optional: direction of traversal per link for backbone ordering
        dir_seq = row.get("dir_seq", None)

        # All lane-groups appearing on any link in this superlink
        G_link = lane_groups_for_link_seq(link_seq)
        if not G_link:
            continue

        # Junction lane-group memberships (OUT / IN)
        u_outpts = J_graph.nodes[ju].get("out_points", {}) or {}
        v_inpts  = J_graph.nodes[jv].get("in_points",  {}) or {}
        v_outpts = J_graph.nodes[jv].get("out_points", {}) or {}
        u_inpts  = J_graph.nodes[ju].get("in_points",  {}) or {}

        LG_u_out = {int(g) for _, gids in u_outpts.items() for g in gids if int(g) in G_link}
        LG_v_in  = {int(g) for _, gids in v_inpts.items()  for g in gids if int(g) in G_link}
        LG_v_out = {int(g) for _, gids in v_outpts.items() for g in gids if int(g) in G_link}
        LG_u_in  = {int(g) for _, gids in u_inpts.items()  for g in gids if int(g) in G_link}

        # Backbone for ordering (optional)
        backbone = None
        if dir_seq is not None:
            try:
                backbone = _superlink_backbone_xy(road, link_seq, dir_seq)
            except Exception:
                backbone = None

        # Existence of direction(s)
        dir_u_v_exists = bool(LG_u_out and LG_v_in)
        dir_v_u_exists = bool(LG_v_out and LG_u_in)

        # u -> v
        if dir_u_v_exists:
            lanes_uv = build_direction_for_superlink(
                ju, jv, G_link, LG_u_out, LG_v_in, backbone_xy=backbone
            )
            if lanes_uv:
                J_graph.add_edge(
                    ju,
                    jv,
                    key=next_key,
                    link_id=tuple(int(l) for l in link_seq),
                    lanes=lanes_uv,
                    is_superlink=True,
                    link_seq=[int(l) for l in link_seq],
                    dir_seq=(list(dir_seq) if dir_seq is not None else None),
                    length_m=0.0,
                    x_all=None,
                    y_all=None,
                    u_x=None,
                    u_y=None,
                    v_x=None,
                    v_y=None,
                    u_s=None,
                    v_s=None,
                )
                next_key += 1

        # v -> u  (note: backbone direction must flip for correct ordering)
        if dir_v_u_exists:
            backbone2 = None
            if backbone is not None:
                backbone2 = backbone[::-1].copy()

            lanes_vu = build_direction_for_superlink(
                jv, ju, G_link, LG_v_out, LG_u_in, backbone_xy=backbone2
            )
            if lanes_vu:
                J_graph.add_edge(
                    jv,
                    ju,
                    key=next_key,
                    link_id=tuple(int(l) for l in link_seq),
                    lanes=lanes_vu,
                    is_superlink=True,
                    link_seq=[int(l) for l in link_seq],
                    dir_seq=(list(dir_seq) if dir_seq is not None else None),
                    length_m=0.0,
                    x_all=None,
                    y_all=None,
                    u_x=None,
                    u_y=None,
                    v_x=None,
                    v_y=None,
                    u_s=None,
                    v_s=None,
                )
                next_key += 1

    return J_graph


def _closest_index_on_poly(P, target_xy):
    """Return index of point on P closest to target_xy."""
    P = np.asarray(P, float)
    t = np.asarray(target_xy, float)
    d2 = (P[:, 0] - t[0])**2 + (P[:, 1] - t[1])**2
    return int(np.argmin(d2)), float(np.sqrt(d2.min()))


def _nearest_cp_on_poly(P, cp_ids, node_xy):
    """
    Find the CP in cp_ids whose coordinate is closest to polyline P.
    Returns (best_cp_id, best_idx_on_P, best_dist).
    If cp_ids is empty, returns (None, None, +inf).
    """
    best_cp = None
    best_idx = None
    best_d = float("inf")
    for cp in cp_ids:
        xy = node_xy.get(int(cp))
        if xy is None:
            continue
        idx, d = _closest_index_on_poly(P, xy)
        if d < best_d:
            best_d = d
            best_cp = int(cp)
            best_idx = idx
    return best_cp, best_idx, best_d

def orient_and_trim_link_lane(
    center_xy,
    j_u, j_v,                     # upstream and downstream junction ids
    junction_cp_sets,             # from compute_junction_in_out_cps
    lane_node_xy,                 # from build_lane_node_xy_from_Nodes
    tol_cp=25.0,                  # max allowed dist polyline <-> cp
):
    """
    center_xy: list/array of (x,y) along the lane (full Sxy polyline).
    j_u, j_v : junction IDs at the ends of the link (J_graph edge u -> v).
    Returns:
       {
         "direction": "u_to_v" or "v_to_u" or None,
         "poly_xy": trimmed & oriented Nx2 array,
         "start_cp": cp_id at upstream, or None,
         "end_cp": cp_id at downstream, or None,
       }
    or None if we cannot reliably associate this lane with both junctions.
    """

    P = np.asarray(center_xy, float)
    if P.ndim != 2 or P.shape[0] < 2:
        return None

    # CP sets
    cp_u_out = junction_cp_sets.get(j_u, {}).get("out", set())
    cp_u_in  = junction_cp_sets.get(j_u, {}).get("in", set())
    cp_v_out = junction_cp_sets.get(j_v, {}).get("out", set())
    cp_v_in  = junction_cp_sets.get(j_v, {}).get("in", set())

    # --- option A: lane oriented u -> v ---
    # upstream out at j_u, downstream in at j_v
    cp_u_out_A, idx_u_A, d_u_A = _nearest_cp_on_poly(P, cp_u_out, lane_node_xy)
    cp_v_in_A, idx_v_A, d_v_A = _nearest_cp_on_poly(P, cp_v_in, lane_node_xy)

    # create trimmed polyline for A if indices are valid
    poly_A = None
    cost_A = float("inf")
    if cp_u_out_A is not None and cp_v_in_A is not None:
        # ensure correct order
        if idx_u_A <= idx_v_A:
            poly_A = P[idx_u_A:idx_v_A+1, :]
        else:
            # indices reversed -> no good in this orientation
            poly_A = None
        if poly_A is not None:
            cost_A = d_u_A + d_v_A

    # --- option B: lane oriented v -> u (i.e., reverse P) ---
    P_rev = P[::-1, :]
    cp_v_out_B, idx_v_B, d_v_B = _nearest_cp_on_poly(P_rev, cp_v_out, lane_node_xy)
    cp_u_in_B, idx_u_B, d_u_B = _nearest_cp_on_poly(P_rev, cp_u_in, lane_node_xy)

    poly_B = None
    cost_B = float("inf")
    if cp_v_out_B is not None and cp_u_in_B is not None:
        if idx_v_B <= idx_u_B:
            poly_B = P_rev[idx_v_B:idx_u_B+1, :]
        else:
            poly_B = None
        if poly_B is not None:
            cost_B = d_v_B + d_u_B

    # --- choose best direction ---
    # require distances below tolerance on *both* ends
    if poly_A is not None and d_u_A <= tol_cp and d_v_A <= tol_cp:
        candidate_A = True
    else:
        candidate_A = False

    if poly_B is not None and d_v_B <= tol_cp and d_u_B <= tol_cp:
        candidate_B = True
    else:
        candidate_B = False

    if candidate_A and (not candidate_B or cost_A <= cost_B):
        return {
            "direction": "u_to_v",
            "poly_xy": poly_A,
            "start_cp": cp_u_out_A,
            "end_cp": cp_v_in_A,
        }

    if candidate_B:
        return {
            "direction": "v_to_u",
            "poly_xy": poly_B,
            "start_cp": cp_v_out_B,
            "end_cp": cp_u_in_B,
        }

    # no reliable association
    return None
