# routing.py
from __future__ import annotations
from typing import Dict, Any, Tuple, List, Optional
import math
import numpy as np
import pandas as pd
import networkx as nx

from libs.geometry_helpers import latlon_to_xy
from libs.graph import build_simple_lane_graph

# Optional rust/retworkx backend (package was renamed from retworkx -> rustworkx)
_HAS_RX = False
try:
    import rustworkx as rx  # new name
    _HAS_RX = True
except Exception:
    try:
        import retworkx as rx  # old name, still widely used
        _HAS_RX = True
    except Exception:
        rx = None


# ============================ Public API ============================

def route_between_points(
    Sxy: Dict[str, Any],
    SA: Dict[str, Any],
    origin: Dict[str, float],
    start: Tuple[float, float],
    goal: Tuple[float, float],
    start_is_gnss: bool = False,
    goal_is_gnss: bool = False,
    node_merge_eps: float = 0.10,
    k_end_candidates: int = 5,
    weight_attr: str = "weight",
    backend: str = "networkx",  # "networkx" (default) or "retworkx"
    _preselect_lanes: int = 64,  # how many bbox-near lanes to evaluate exactly
) -> Dict[str, Any]:
    """
    Route over the strict lane-level directed graph built from attributes.

    Policy:
      - Use the closest GOAL lane first. For that lane, try all admissible end-nodes
        (ordered by proximity to the goal projection) against all admissible START
        nodes (ordered by proximity to the start projection). Select the shortest path
        among those. If none exist, ONLY THEN move to the next-closest goal lane.
    """
    # 0) Normalize inputs to ENU
    sx, sy = _to_enu(start, origin) if start_is_gnss else start
    gx, gy = _to_enu(goal,  origin) if goal_is_gnss  else goal

    # 1) Build lane graph once (choose backend)
    G, Nodes, Edges, lane_endpoint_map = build_simple_lane_graph(
        Sxy, SA, node_merge_eps=node_merge_eps, backend=backend
    )

    # Fast coordinate maps for sorting
    NX = Nodes.set_index("node_id")["X"].to_dict()
    NY = Nodes.set_index("node_id")["Y"].to_dict()

    # 2) Spatial index for snapping (reused for start + goal) with robust fallback
    lsi = LaneSpatialIndex(Sxy)

    # 2a) Snap START to closest lane; collect admissible start nodes (edge tails)
    s_snap = lsi.snap_to_closest((sx, sy), preselect=_preselect_lanes)
    if s_snap is None:  # robust fallback: full exact scan
        s_snap = snap_to_closest_lane_full(Sxy, (sx, sy))
    if s_snap is None:
        raise RuntimeError("Could not snap START to any lane.")

    s_nodes = _possible_start_nodes_for_lane(Edges, s_snap["GroupID"], s_snap["LaneNo"])
    if not s_nodes:
        s_nodes = [_nearest_node(Nodes, (sx, sy))]
    s_nodes = _sort_nodes_by_xy_fast(NX, NY, s_nodes, (sx, sy))

    # 3) Prepare K closest GOAL lane candidates (sorted by distance)
    goal_candidates = lsi.k_closest((gx, gy), k=max(1, int(k_end_candidates)), preselect=_preselect_lanes)
    if not goal_candidates:  # robust fallback
        goal_candidates = k_closest_lanes_full(Sxy, (gx, gy), k=max(1, int(k_end_candidates)))
    if not goal_candidates:
        raise RuntimeError("Could not snap GOAL to any lane.")

    last_exc: Optional[Exception] = None

    # Try candidates in *distance order*, but do NOT accept a farther lane if the closest is routable.
    for _, cand in enumerate(goal_candidates):
        # admissible end nodes for this (gid, ln)
        g_nodes = _possible_end_nodes_for_lane(Edges, cand["GroupID"], cand["LaneNo"])
        if not g_nodes:
            g_nodes = [_nearest_node(Nodes, (gx, gy))]
        g_nodes = _sort_nodes_by_xy_fast(NX, NY, g_nodes, cand["proj_xy"])

        # ---- Multi-source shortest paths once per candidate lane ----
        try:
            if _is_retworkx_graph(G):
                best = _rx_multi_source_to_multi_targets(G, s_nodes, set(g_nodes))
                if best is None:
                    # pairwise fallback (rare)
                    best = _rx_pairwise_fallback(G, s_nodes, g_nodes)
            else:
                best = _nx_multi_source_to_multi_targets(G, s_nodes, set(g_nodes), weight_attr)
                if best is None:
                    best = _nx_pairwise_fallback(G, s_nodes, g_nodes, weight_attr)
        except Exception as ex:
            last_exc = ex
            best = None

        if best is not None:
            length, path_nodes, end_node = best

            # Build path_edges:
            if _is_retworkx_graph(G):
                # Create a fast lookup {(FromN,ToN) -> E}; last one wins (matches non-multigraph collapse)
                _edge_key_to_e = {}
                for r in Edges[["FromN","ToN","E"]].itertuples(index=False):
                    _edge_key_to_e[(int(r.FromN), int(r.ToN))] = int(r.E)
                path_edges = []
                for u, v in zip(path_nodes[:-1], path_nodes[1:]):
                    eidx = _edge_key_to_e.get((int(u), int(v)))
                    if eidx is not None:
                        path_edges.append(int(eidx))
            else:
                path_edges = _path_edges_from_nodes_nx(G, path_nodes)

            # --- inline lane attribute enrichment (unchanged) ---
            def _idx_edges_by_id(Edges):
                idx = {}
                if hasattr(Edges, "to_dict"):  # pandas.DataFrame
                    for _, row in Edges.iterrows():
                        d = row.to_dict(); e = d.get("E", len(idx)); idx[int(e)] = d
                elif isinstance(Edges, list):
                    for d in Edges:
                        if isinstance(d, dict):
                            e = d.get("E", len(idx)); idx[int(e)] = d
                return idx

            by_group = (SA or {}).get("by_group", {})
            edges_map = _idx_edges_by_id(Edges)

            def _as_list_set(x):
                return sorted(list(x)) if isinstance(x, set) else x

            steps = []
            for e in path_edges:
                erow = edges_map.get(int(e), {})
                gid  = str(erow.get("GroupID", "")).strip()
                ln   = erow.get("LaneNo", None)
                try: ln = int(ln) if ln is not None else None
                except: ln = None
                L = ((by_group.get(gid) or {}).get("lanes") or {}).get(ln) if gid and (ln is not None) else None
                step = {
                    "E": int(e),
                    "GroupID": gid,
                    "LaneNo": ln,
                    "Dir": erow.get("Dir"),
                    "weight": erow.get("Weight", erow.get("weight")),
                    "attrs": None
                }
                if L:
                    step["attrs"] = {
                        "directions": _as_list_set(L.get("directions", set())),
                        "ranges": list(L.get("ranges", [])),
                        "types": _as_list_set(L.get("types", set())),
                        "width_profiles": list(L.get("width_profiles", [])),
                        "flags": dict(L.get("flags", {})),
                    }
                steps.append(step)

            uniq = {}
            for s in steps:
                k = (s["GroupID"], s["LaneNo"])
                if not all(k) or k in uniq: continue
                L = ((by_group.get(s["GroupID"]) or {}).get("lanes") or {}).get(s["LaneNo"])
                if not L: continue
                uniq[k] = {
                    "GroupID": s["GroupID"],
                    "LaneNo": s["LaneNo"],
                    "attrs": {
                        "directions": _as_list_set(L.get("directions", set())),
                        "ranges": list(L.get("ranges", [])),
                        "types": _as_list_set(L.get("types", set())),
                        "width_profiles": list(L.get("width_profiles", [])),
                        "flags": dict(L.get("flags", {})),
                    }
                }
            # --- end enrichment ---

            return {
                "path_nodes": path_nodes,
                "path_edges": path_edges,
                "length": float(length),
                "start_snap": s_snap,
                "end_snap":   cand,
                "G": G, "Nodes": Nodes, "Edges": Edges, "lane_endpoint_map": lane_endpoint_map,
                "steps": steps,
                "lane_attrs_summary": list(uniq.values())
            }

    raise RuntimeError(f"No route found to any of the {len(goal_candidates)} goal candidates. Last error: {last_exc}")


# ====================== Snapping / candidates (fast + robust) ======================

class LaneSpatialIndex:
    """
    Lightweight lane spatial index:
      - Stores per-lane (gid,ln) with (x,y) and a bounding box.
      - For snapping: preselect top-N by bbox distance, then compute exact polyline distance.
      - Robust: if preselect misses, caller falls back to full scan.
    """
    __slots__ = ("records",)

    def __init__(self, Sxy: Dict[str, Any]):
        recs = []
        for rec in (Sxy.get("lanes") or []):
            gid = str(rec.get("lane_group_ref"))
            ln  = int(rec.get("lane_number") if rec.get("lane_number") is not None
                      else rec.get("lane_index_within_group"))
            x = np.asarray(rec.get("x", []), float).reshape(-1)
            y = np.asarray(rec.get("y", []), float).reshape(-1)
            if x.size < 2:  # ignore degenerate
                continue
            xmin, xmax = float(x.min()), float(x.max())
            ymin, ymax = float(y.min()), float(y.max())
            recs.append((gid, ln, x, y, xmin, xmax, ymin, ymax))
        self.records = recs

    @staticmethod
    def _bbox_dist(px: float, py: float, xmin: float, xmax: float, ymin: float, ymax: float) -> float:
        dx = 0.0 if xmin <= px <= xmax else (xmin - px if px < xmin else px - xmax)
        dy = 0.0 if ymin <= py <= ymax else (ymin - py if py < ymin else py - ymax)
        return math.hypot(dx, dy)

    def snap_to_closest(self, pt_xy: Tuple[float, float], preselect: int = 64) -> Optional[Dict[str, Any]]:
        x0, y0 = pt_xy
        if not self.records:
            return None
        coarse = []
        for gid, ln, x, y, xmin, xmax, ymin, ymax in self.records:
            coarse.append((self._bbox_dist(x0, y0, xmin, xmax, ymin, ymax), gid, ln, x, y))
        coarse.sort(key=lambda t: t[0])
        best = None
        for _, gid, ln, x, y in coarse[:max(1, int(preselect))]:
            d, s, qx, qy = _point_polyline_dist_frac(x, y, x0, y0)
            if (best is None) or (d < best["dist"]):
                best = {"GroupID": gid, "LaneNo": ln, "dist": float(d), "proj_xy": (float(qx), float(qy)), "s": float(s)}
        return best

    def k_closest(self, pt_xy: Tuple[float, float], k: int = 5, preselect: int = 64) -> List[Dict[str, Any]]:
        x0, y0 = pt_xy
        if not self.records:
            return []
        coarse = []
        for gid, ln, x, y, xmin, xmax, ymin, ymax in self.records:
            coarse.append((self._bbox_dist(x0, y0, xmin, xmax, ymin, ymax), gid, ln, x, y))
        coarse.sort(key=lambda t: t[0])
        bag = []
        for _, gid, ln, x, y in coarse[:max(k*4, int(preselect))]:  # compute exact on a bit more than k
            d, s, qx, qy = _point_polyline_dist_frac(x, y, x0, y0)
            bag.append({"GroupID": gid, "LaneNo": ln, "dist": float(d), "proj_xy": (float(qx), float(qy)), "s": float(s)})
        bag.sort(key=lambda z: z["dist"])
        return bag[:max(1, int(k))]


# --- robust full-scan fallbacks (used if index misses) ----------------

def snap_to_closest_lane_full(Sxy: Dict[str, Any], pt_xy: Tuple[float, float]) -> Optional[Dict[str, Any]]:
    x0, y0 = pt_xy
    best = None
    for rec in (Sxy.get("lanes") or []):
        gid = str(rec.get("lane_group_ref"))
        ln  = int(rec.get("lane_number") if rec.get("lane_number") is not None
                  else rec.get("lane_index_within_group"))
        x = np.asarray(rec.get("x", []), float).reshape(-1)
        y = np.asarray(rec.get("y", []), float).reshape(-1)
        if x.size < 2:
            continue
        d, s, qx, qy = _point_polyline_dist_frac(x, y, x0, y0)
        if (best is None) or (d < best["dist"]):
            best = {"GroupID": gid, "LaneNo": ln, "dist": float(d), "proj_xy": (float(qx), float(qy)), "s": float(s)}
    return best

def k_closest_lanes_full(Sxy: Dict[str, Any], pt_xy: Tuple[float, float], k: int = 5) -> List[Dict[str, Any]]:
    x0, y0 = pt_xy
    bag = []
    for rec in (Sxy.get("lanes") or []):
        gid = str(rec.get("lane_group_ref"))
        ln  = int(rec.get("lane_number") if rec.get("lane_number") is not None
                  else rec.get("lane_index_within_group"))
        x = np.asarray(rec.get("x", []), float).reshape(-1)
        y = np.asarray(rec.get("y", []), float).reshape(-1)
        if x.size < 2:
            continue
        d, s, qx, qy = _point_polyline_dist_frac(x, y, x0, y0)
        bag.append({"GroupID": gid, "LaneNo": ln, "dist": float(d), "proj_xy": (float(qx), float(qy)), "s": float(s)})
    bag.sort(key=lambda z: z["dist"])
    return bag[:max(1, int(k))]


# ===================== Endpoint selection helpers =====================

def _possible_start_nodes_for_lane(Edges: pd.DataFrame, gid: str, ln: int) -> List[int]:
    sub = Edges[(Edges["GroupID"]==gid) & (Edges["LaneNo"]==int(ln))]
    return sorted(set(int(v) for v in sub["FromN"].tolist()))

def _possible_end_nodes_for_lane(Edges: pd.DataFrame, gid: str, ln: int) -> List[int]:
    sub = Edges[(Edges["GroupID"]==gid) & (Edges["LaneNo"]==int(ln))]
    return sorted(set(int(v) for v in sub["ToN"].tolist()))

def _sort_nodes_by_xy(Nodes: pd.DataFrame, node_ids: List[int], ref_xy: Tuple[float,float]) -> List[int]:
    X = Nodes.set_index("node_id")["X"].to_dict()
    Y = Nodes.set_index("node_id")["Y"].to_dict()
    rx, ry = float(ref_xy[0]), float(ref_xy[1])
    return sorted(node_ids, key=lambda n: math.hypot(float(X[int(n)])-rx, float(Y[int(n)])-ry))

def _sort_nodes_by_xy_fast(NX: Dict[int,float], NY: Dict[int,float], node_ids: List[int], ref_xy: Tuple[float,float]) -> List[int]:
    rx, ry = float(ref_xy[0]), float(ref_xy[1])
    return sorted(node_ids, key=lambda n: math.hypot(float(NX[int(n)])-rx, float(NY[int(n)])-ry))


# ========================= Graph/path utilities =========================

def _is_retworkx_graph(G) -> bool:
    if not _HAS_RX:
        return False
    return isinstance(G, (rx.PyGraph, rx.PyDiGraph))

def _nx_multi_source_to_multi_targets(G: nx.DiGraph, sources: List[int], targets: set, weight_attr: str):
    """
    Run multi-source Dijkstra once, then pick the best among targets.
    Returns (length, path_nodes, best_target) or None if unreachable.
    """
    lengths, paths = nx.multi_source_dijkstra(G, sources=sources, target=None, weight=weight_attr)
    best_len = float("inf"); best_t = None
    for t in targets:
        if t in lengths and lengths[t] < best_len:
            best_len = lengths[t]; best_t = t
    if best_t is None or not math.isfinite(best_len):
        return None
    return float(best_len), [int(n) for n in paths[best_t]], int(best_t)

def _nx_pairwise_fallback(G: nx.DiGraph, sources: List[int], targets: List[int], weight_attr: str):
    """
    Robust fallback: try pairwise shortest_path (weighted); if that fails, try unweighted.
    """
    best = None
    for s in sources:
        for t in targets:
            try:
                path = nx.shortest_path(G, source=int(s), target=int(t), weight=weight_attr)
                total = _path_length_nx(G, path, weight_attr)
                if (best is None) or (total < best[0] - 1e-9):
                    best = (float(total), [int(n) for n in path], int(t))
            except Exception:
                try:
                    path = nx.shortest_path(G, source=int(s), target=int(t))
                    total = _path_length_nx(G, path, weight_attr)
                    if (best is None) or (total < best[0] - 1e-9):
                        best = (float(total), [int(n) for n in path], int(t))
                except Exception:
                    continue
    return best

# -------- retworkx / rustworkx helpers (version-agnostic) --------

def _rx_dijkstra_paths_from(G, source: int):
    """
    Robust wrapper for retworkx/rustworkx Dijkstra.
    Returns mapping-like object: target_node -> list[path nodes].
    Tries multiple signatures (new/old), then falls back to unweighted if needed.
    Edge weight is stored as the edge payload (float).
    """
    def _w(edge_payload):
        return float(edge_payload)

    # Prefer newer name first
    fn = getattr(rx, "dijkstra_shortest_paths", None)
    if fn is not None:
        # Try common signatures without passing target (signature differs by version)
        for args in [
            (G, int(source), _w),          # (graph, source, weight_fn)
            (G, int(source), None, _w),    # (graph, source, target=None, weight_fn)
            (G, int(source),),             # unweighted
        ]:
            try:
                return fn(*args)
            except TypeError:
                continue

    # Older directed name
    fn = getattr(rx, "digraph_dijkstra_shortest_paths", None)
    if fn is not None:
        for args in [
            (G, int(source), _w),
            (G, int(source), None, _w),
            (G, int(source),),
        ]:
            try:
                return fn(*args)
            except TypeError:
                continue

    raise RuntimeError("retworkx/rustworkx Dijkstra function not found. Please upgrade rustworkx/retworkx.")

def _rx_multi_source_to_multi_targets(G, sources: List[int], targets: set):
    """
    Run one Dijkstra per source (fast in Rust), pick the best among 'targets'.
    Returns (length, path_nodes, best_target) or None if unreachable.
    """
    targets_int = set(map(int, targets))

    def _path_len(path: List[int]) -> float:
        total = 0.0
        for u, v in zip(path[:-1], path[1:]):
            total += float(G.get_edge_data(u, v))  # payload is the weight
        return total

    best_len = float("inf")
    best_path = None
    best_t = None

    for s in map(int, sources):
        spaths = _rx_dijkstra_paths_from(G, s)  # mapping-like (PathMapping in rustworkx)
        for t in targets_int:
            # PathMapping doesn't have .get(); use subscripting with KeyError handling
            try:
                path = spaths[t]
            except Exception:
                path = None
            if not path or len(path) < 2:
                continue
            total = _path_len(path)
            if total < best_len:
                best_len = total
                best_path = path
                best_t = t

    if best_path is None:
        return None
    return float(best_len), [int(n) for n in best_path], int(best_t)

def _rx_pairwise_fallback(G, sources: List[int], targets: List[int]):
    """
    Simpler (unweighted) fallback for retworkx if the weighted calls failed due to version quirks.
    """
    best_len = float("inf"); best_path = None; best_t = None

    def _path_len(path: List[int]) -> float:
        total = 0.0
        for u, v in zip(path[:-1], path[1:]):
            total += float(G.get_edge_data(u, v))
        return total

    # choose a callable regardless of version
    fn = getattr(rx, "dijkstra_shortest_paths", None)
    if fn is None:
        fn = getattr(rx, "digraph_dijkstra_shortest_paths", None)
    if fn is None:
        return None

    for s in map(int, sources):
        # Unweighted (let retworkx choose any shortest-hops path)
        try:
            spaths = fn(G, int(s))  # PathMapping-like
        except TypeError:
            # Try signature (G, source, None) if needed
            try:
                spaths = fn(G, int(s), None)
            except Exception:
                continue

        for t in map(int, targets):
            try:
                path = spaths[t]
            except Exception:
                path = None
            if not path or len(path) < 2:
                continue
            total = _path_len(path)
            if total < best_len:
                best_len = total; best_path = path; best_t = t

    if best_path is None:
        return None
    return float(best_len), [int(n) for n in best_path], int(best_t)

def _path_edges_from_nodes_nx(G: nx.DiGraph, nodes: List[int]) -> List[int]:
    """
    Extract 'E' indices for NetworkX graph edges along node path.
    Handles the case where DiGraph has single edge per (u,v).
    """
    eidxs = []
    for u, v in zip(nodes[:-1], nodes[1:]):
        data = G.get_edge_data(u, v)
        if isinstance(data, dict) and "eidx" in data:
            eidxs.append(int(data["eidx"]))
        else:
            # Possible MultiDiGraph-like dict form (still try to find 'eidx')
            if isinstance(data, dict):
                for _, d in data.items():
                    if isinstance(d, dict) and ("eidx" in d):
                        eidxs.append(int(d["eidx"]))
                        break
    return eidxs

def _path_length_nx(G: nx.DiGraph, nodes: List[int], weight_attr: str = "weight") -> float:
    total = 0.0
    for u, v in zip(nodes[:-1], nodes[1:]):
        total += float(G[u][v].get(weight_attr, 0.0))
    return total


# ========================= Geometry helpers =========================

def _to_enu(pt: Tuple[float,float], origin: Dict[str, float]) -> Tuple[float,float]:
    lat0, lon0, h0 = origin["lat0"], origin["lon0"], origin["h0"]
    x, y = latlon_to_xy(np.asarray([pt[0]]), np.asarray([pt[1]]), lat0, lon0, h0)
    return float(x[0]), float(y[0])

def _point_polyline_dist_frac(x: np.ndarray, y: np.ndarray, px: float, py: float):
    """
    Return (dist, s, qx, qy) where:
      dist = minimal distance from P to the piecewise linear path
      s    = normalized arc-length fraction in [0,1] of closest point
      (qx,qy) = coordinates of the closest point
    """
    seg_dx = np.diff(x); seg_dy = np.diff(y)
    seg_L = np.hypot(seg_dx, seg_dy)
    cum = np.concatenate([[0.0], np.cumsum(seg_L)])
    totalL = float(cum[-1]) if cum[-1] > 0 else 1.0

    best_d = float("inf"); best_q = (x[0], y[0]); best_cum = 0.0
    for i in range(len(seg_L)):
        L = seg_L[i]
        if L <= 0: continue
        vx, vy = seg_dx[i], seg_dy[i]
        wx, wy = px - x[i], py - y[i]
        t = (vx*wx + vy*wy) / (L*L)
        t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
        qx, qy = x[i] + t*vx, y[i] + t*vy
        d = math.hypot(px - qx, py - qy)
        if d < best_d:
            best_d = d; best_q = (qx, qy); best_cum = float(cum[i] + t*L)
    s = best_cum / totalL
    return best_d, s, float(best_q[0]), float(best_q[1])

def _nearest_node(Nodes: pd.DataFrame, pt_xy: Tuple[float,float]) -> int:
    dx = Nodes["X"].to_numpy(dtype=float) - float(pt_xy[0])
    dy = Nodes["Y"].to_numpy(dtype=float) - float(pt_xy[1])
    idx = int(np.argmin(np.hypot(dx, dy)))
    return int(Nodes.at[idx, "node_id"])


# --- lane polyline index ------------------------------------------------------

def build_lane_polyline_index(Sxy: Dict[str, Any]) -> Dict[tuple, tuple]:
    """
    Returns {(GroupID, LaneNo): (x_np, y_np)} using Sxy['lanes'] arrays.
    """
    idx = {}
    for rec in (Sxy.get("lanes") or []):
        gid = str(rec.get("lane_group_ref"))
        ln  = int(rec.get("lane_number") if rec.get("lane_number") is not None
                  else rec.get("lane_index_within_group"))
        x = np.asarray(rec.get("x", []), float).reshape(-1)
        y = np.asarray(rec.get("y", []), float).reshape(-1)
        if x.size >= 2:
            idx[(gid, ln)] = (x, y)
    return idx


# --- fraction trimming on a polyline -----------------------------------------

def _cumlen(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    seg = np.hypot(np.diff(x), np.diff(y))
    return np.concatenate([[0.0], np.cumsum(seg)])

def _cut_polyline_by_fraction(x: np.ndarray, y: np.ndarray, s0: float, s1: float) -> tuple:
    """
    Keep segment between arc-length fractions s0..s1 (0..1, inclusive),
    interpolating endpoints as needed.
    """
    s0 = max(0.0, min(1.0, float(s0)))
    s1 = max(0.0, min(1.0, float(s1)))
    if s1 < s0: s0, s1 = s1, s0

    L = _cumlen(x, y)
    if L[-1] <= 0:
        return x[:1].copy(), y[:1].copy()
    a = s0 * L[-1]; b = s1 * L[-1]

    def _interp_at(arclen: float) -> tuple:
        i = int(np.searchsorted(L, arclen, side="right") - 1)
        i = max(0, min(i, len(x)-2))
        segL = L[i+1] - L[i]
        if segL <= 0: return float(x[i]), float(y[i])
        t = (arclen - L[i]) / segL
        return float(x[i] + t*(x[i+1]-x[i])), float(y[i] + t*(y[i+1]-y[i]))

    xa, ya = _interp_at(a)
    xb, yb = _interp_at(b)

    mid_mask = (L > a + 1e-12) & (L < b - 1e-12)
    xm = x[mid_mask]; ym = y[mid_mask]

    X = np.concatenate([[xa], xm, [xb]])
    Y = np.concatenate([[ya], ym, [yb]])
    return X, Y


# --- main extractor -----------------------------------------------------------
def route_polyline_xy(route: dict, Sxy: dict, prefer_smoothing: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a single piecewise (X,Y) following lane polylines in the route.
    Always returns a tuple of numpy arrays (possibly empty).
    """
    try:
        G, Nodes, Edges = route["G"], route["Nodes"], route["Edges"]
        start_snap, end_snap = route["start_snap"], route["end_snap"]
        path_nodes = route.get("path_nodes", [])
        path_edges = route.get("path_edges", [])
    except Exception:
        return np.array([]), np.array([])

    lane_idx = build_lane_polyline_index(Sxy)

    if not path_edges or len(Edges) == 0:
        return np.array([]), np.array([])

    Emap = Edges.set_index("E")[["GroupID","LaneNo","Dir"]]
    first_e = int(path_edges[0]); last_e  = int(path_edges[-1])

    def _safe_edge_row(eidx: int):
        if eidx not in Emap.index:
            return None
        r = Emap.loc[eidx]
        return str(r["GroupID"]), int(r["LaneNo"]), str(r["Dir"]).upper()

    first_info = _safe_edge_row(first_e)
    last_info  = _safe_edge_row(last_e)
    if first_info is None or last_info is None:
        return np.array([]), np.array([])

    first_gid, first_ln, first_dir = first_info
    last_gid,  last_ln,  last_dir  = last_info

    X_all: List[np.ndarray] = []
    Y_all: List[np.ndarray] = []

    # ---------- 0) Preface ----------
    try:
        if not (str(start_snap["GroupID"]) == first_gid and int(start_snap["LaneNo"]) == first_ln):
            gid, ln = str(start_snap["GroupID"]), int(start_snap["LaneNo"])
            if (gid, ln) in lane_idx:
                x, y = lane_idx[(gid, ln)]
                f_start, f_end = _forward_endpoints_for_lane(Edges, gid, ln)
                if f_start is not None and f_end is not None and len(path_nodes) > 0:
                    start_node = int(path_nodes[0])
                    if start_node == f_start:
                        Xc, Yc = _cut_polyline_by_fraction(x, y, 0.0, float(start_snap["s"]))
                        Xc, Yc = Xc[::-1], Yc[::-1]
                        if Xc.size: X_all.append(Xc); Y_all.append(Yc)
                    elif start_node == f_end:
                        Xc, Yc = _cut_polyline_by_fraction(x, y, float(start_snap["s"]), 1.0)
                        if Xc.size: X_all.append(Xc); Y_all.append(Yc)
    except Exception:
        pass

    # ---------- 1) Main body ----------
    for eidx in path_edges:
        info = _safe_edge_row(int(eidx))
        if info is None:
            continue
        gid, ln, direction = info
        if (gid, ln) not in lane_idx:
            continue
        x, y = lane_idx[(gid, ln)]
        if direction == "FORWARD":
            def s_flip(s): return float(s)
        else:
            x = x[::-1].copy(); y = y[::-1].copy()
            def s_flip(s): return 1.0 - float(s)

        if int(eidx) == first_e:
            s0 = float(start_snap["s"]) if (str(start_snap["GroupID"])==gid and int(start_snap["LaneNo"])==ln) else 0.0
            s0 = s_flip(s0)
            Xc, Yc = _cut_polyline_by_fraction(x, y, s0, 1.0)
        elif int(eidx) == last_e:
            s1 = float(end_snap["s"]) if (str(end_snap["GroupID"])==gid and int(end_snap["LaneNo"])==ln) else 1.0
            s1 = s_flip(s1) if direction != "FORWARD" else s1
            Xc, Yc = _cut_polyline_by_fraction(x, y, 0.0, s1)
        else:
            Xc, Yc = x, y

        if Xc is None or Yc is None or Xc.size == 0:
            continue
        if X_all and X_all[-1].size and Xc.size:
            if (abs(X_all[-1][-1]-Xc[0]) < 1e-8) and (abs(Y_all[-1][-1]-Yc[0]) < 1e-8):
                Xc, Yc = Xc[1:], Yc[1:]
        X_all.append(np.asarray(Xc)); Y_all.append(np.asarray(Yc))

    # ---------- 2) Postface ----------
    try:
        if not (str(end_snap["GroupID"]) == last_gid and int(end_snap["LaneNo"]) == last_ln):
            gid, ln = str(end_snap["GroupID"]), int(end_snap["LaneNo"])
            if (gid, ln) in lane_idx and len(path_nodes) > 0:
                x, y = lane_idx[(gid, ln)]
                f_start, f_end = _forward_endpoints_for_lane(Edges, gid, ln)
                last_node = int(path_nodes[-1])
                Xc = Yc = None
                if f_start is not None and f_end is not None:
                    if last_node == f_start:
                        Xc, Yc = _cut_polyline_by_fraction(x, y, 0.0, float(end_snap["s"]))
                    elif last_node == f_end:
                        Xc, Yc = _cut_polyline_by_fraction(x, y, float(end_snap["s"]), 1.0)
                        Xc, Yc = Xc[::-1], Yc[::-1]
                if Xc is not None and Xc.size:
                    if X_all and X_all[-1].size:
                        if (abs(X_all[-1][-1]-Xc[0]) < 1e-8) and (abs(Y_all[-1][-1]-Yc[0]) < 1e-8):
                            Xc, Yc = Xc[1:], Yc[1:]
                    X_all.append(np.asarray(Xc)); Y_all.append(np.asarray(Yc))
    except Exception:
        pass

    if not X_all:
        return np.array([]), np.array([])

    X = np.concatenate(X_all); Y = np.concatenate(Y_all)

    if prefer_smoothing and X.size >= 3:
        keep = [0]
        for i in range(1, X.size-1):
            if (abs(X[i]-X[i-1]) > 1e-9) or (abs(Y[i]-Y[i-1]) > 1e-9):
                keep.append(i)
        keep.append(X.size-1)
        keep = np.unique(keep)
        X, Y = X[keep], Y[keep]

    return X, Y


# --- helpers used by route_polyline_xy ---------------------------------

def _forward_endpoints_for_lane(Edges, gid: str, ln: int) -> Tuple[Optional[int], Optional[int]]:
    sub = Edges[(Edges["GroupID"]==gid) & (Edges["LaneNo"]==int(ln))]
    if sub.empty:
        return None, None
    row_f = sub[sub["Dir"]=="FORWARD"]
    if not row_f.empty:
        r = row_f.iloc[0]
        return int(r["FromN"]), int(r["ToN"])
    row_b = sub[sub["Dir"]=="BACKWARD"]
    if not row_b.empty:
        r = row_b.iloc[0]
        return int(r["ToN"]), int(r["FromN"])
    return None, None

def _node_distance_xy(Nodes, nid: int, xy: Tuple[float,float]) -> float:
    row = Nodes[Nodes["node_id"]==int(nid)]
    if row.empty: return float("inf")
    x, y = float(row["X"].iloc[0]), float(row["Y"].iloc[0])
    return math.hypot(x-xy[0], y-xy[1])
