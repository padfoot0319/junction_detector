# graph.py
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import math
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from typing import Iterable, Set, Hashable




# Optional retworkx (Rust) backend
try:
    import retworkx as rx
    _HAS_RX = True
except Exception:
    _HAS_RX = False

# Keep import to remain compatible with older callers
from libs.attrs import dir_from_SA_lane  # noqa: F401


# ======================================================================
# Internals / helpers
# ======================================================================

def _canon_id(x) -> str:
    if x is None:
        return ""
    if isinstance(x, (int, float)) and (not math.isfinite(x)):
        return ""
    return str(x).strip()

def _quantize_xy(x: np.ndarray, y: np.ndarray, eps: float):
    """
    If eps == 0: unique on exact floats. Else grid-quantize by eps, then
    represent each grid bucket by the mean of its members. Returns:
      (X_unique, Y_unique, inverse_index)
    """
    if eps <= 0.0:
        XY = np.column_stack((x.astype(np.float64, copy=False),
                              y.astype(np.float64, copy=False)))
        uniq, inv = np.unique(XY, axis=0, return_inverse=True)
        return uniq[:, 0], uniq[:, 1], inv

    qx = np.rint(x / eps).astype(np.int64, copy=False)
    qy = np.rint(y / eps).astype(np.int64, copy=False)
    Q = np.column_stack((qx, qy))
    uq, inv = np.unique(Q, axis=0, return_inverse=True)
    bc = np.bincount(inv)
    Xmean = np.bincount(inv, weights=x) / bc
    Ymean = np.bincount(inv, weights=y) / bc
    return Xmean, Ymean, inv


# ======================================================================
# Public: connector graph (group-level)
# ======================================================================
def build_connector_graph_enu(
    Sxy: Dict[str, Any],
    ST: Dict[str, Any],
    SA: Dict[str, Any] = None,
    Undirected: bool = True,
    backend: str = "networkx",  # "networkx" (default) or "retworkx"
) -> Dict[str, Any]:
    """
    Nodes: unique connector IDs with ENU coords taken from group reference/first lane.
    Edges: lane groups; Weight = lane_group_length_meters (TopoLen) > polyline length (PolyLen) > fallback chord (FallbackLen).

    Uses SA (attributes) to derive allowed travel directions:
      - FORWARD  => start->end
      - BACKWARD => end->start
      - BOTH     => both
      - UNKNOWN/NONE => fallback to Undirected flag (if True)

    backend: "networkx" (default) or "retworkx".
      - If "retworkx" and retworkx is available, returns rx graphs (PyGraph/PyDiGraph) instead of NetworkX.
      - Otherwise falls back to NetworkX.
    """
    SA = SA or {"by_group": {}}
    use_rx = (backend == "retworkx") and _HAS_RX

    # Index reference & lanes per group (reused by public helpers)
    mRef = {
        str(r.get("lane_group_ref")): i
        for i, r in enumerate(Sxy.get("reference", []) or [])
        if r.get("lane_group_ref") is not None
    }
    lanesByGroup: Dict[str, List[int]] = {}
    for i, L in enumerate(Sxy.get("lanes", []) or []):
        gid = str(L.get("lane_group_ref"))
        lanesByGroup.setdefault(gid, []).append(i)

    def _summarize_group_dir(gid: str) -> str:
        g = SA.get("by_group", {}).get(gid)
        if not g:
            return "UNKNOWN"
        dirs_all = set()
        for lane in g.get("lanes", {}).values():
            dirs = {str(d).upper() for d in (lane.get("directions") or [])}
            dirs_all |= dirs
        if "FORWARD" in dirs_all and "BACKWARD" in dirs_all:
            return "BOTH"
        if "FORWARD" in dirs_all:
            return "FWD"
        if "BACKWARD" in dirs_all:
            return "BWD"
        return "NONE"

    nodeMap: Dict[str, Tuple[float, float]] = {}
    Fr: List[str] = []
    To: List[str] = []
    LGID: List[str] = []
    PolyLen: List[float] = []
    FallbackLen: List[float] = []
    AllowedDir: List[str] = []

    gid_to_len = {
        str(g.get("lane_group_id", "")): g.get("lane_group_length_meters")
        for g in (ST.get("groups") or [])
    }

    for g in (ST.get("groups") or []):
        gid = str(g.get("lane_group_id", ""))
        sID = _canon_id(g.get("start_lane_group_connector_id"))
        eID = _canon_id(g.get("end_lane_group_connector_id"))

        xs, ys, xe, ye = endpoints_from_group_xy(Sxy, mRef, lanesByGroup, gid)
        if sID and math.isfinite(xs):
            nodeMap.setdefault(sID, (xs, ys))
        if eID and math.isfinite(xe):
            nodeMap.setdefault(eID, (xe, ye))

        xg, yg = poly_for_group_xy(Sxy, mRef, lanesByGroup, gid)
        if xg.size >= 2:
            L = float(np.sum(np.hypot(np.diff(xg), np.diff(yg))))
        else:
            L = float("nan")
        F = math.hypot(xe - xs, ye - ys) if (math.isfinite(xs) and math.isfinite(xe)) else float("nan")

        Fr.append(sID)
        To.append(eID)
        LGID.append(gid)
        PolyLen.append(L)
        FallbackLen.append(F)
        AllowedDir.append(_summarize_group_dir(gid))

    # Nodes DF
    ids = list(nodeMap.keys())
    X = np.fromiter((nodeMap[k][0] for k in ids), count=len(ids), dtype=np.float64)
    Y = np.fromiter((nodeMap[k][1] for k in ids), count=len(ids), dtype=np.float64)
    Nodes = pd.DataFrame({"ConnectorID": ids, "X": X, "Y": Y})
    name2row = {k: i for i, k in enumerate(Nodes["ConnectorID"])}

    # Edges DF + weights
    Edges = pd.DataFrame(
        {
            "From": Fr,
            "To": To,
            "LaneGroupID": LGID,
            "PolyLen": np.asarray(PolyLen, float),
            "FallbackLen": np.asarray(FallbackLen, float),
            "AllowedDir": AllowedDir,
        }
    )
    topo = np.array(
        [
            float(gid_to_len[g]) if (g in gid_to_len and gid_to_len[g] is not None and np.isfinite(gid_to_len[g]))
            else np.nan
            for g in LGID
        ],
        dtype=float,
    )
    W = topo.copy()
    mask = ~np.isfinite(W)
    W[mask] = Edges.loc[mask, "PolyLen"].to_numpy()
    mask = ~np.isfinite(W)
    W[mask] = Edges.loc[mask, "FallbackLen"].to_numpy()
    Edges["TopoLen"] = topo
    Edges["Weight"] = W

    # ---------------------- Graph builds ----------------------
    if use_rx:
        # retworkx requires integer node IDs; map ConnectorID <-> idx
        id2idx = {cid: i for i, cid in enumerate(ids)}

        # Undirected simple & weighted
        rxG = rx.PyGraph(multigraph=False);  rxG.add_nodes_from(range(len(ids)))
        rxGw = rx.PyGraph(multigraph=False); rxGw.add_nodes_from(range(len(ids)))

        for f, t in zip(Edges["From"], Edges["To"]):
            rxG.add_edge(id2idx[f], id2idx[t], None)
        for f, t, w in zip(Edges["From"], Edges["To"], Edges["Weight"]):
            rxGw.add_edge(id2idx[f], id2idx[t], float(w) if np.isfinite(w) else 0.0)

        # Directed (attribute aware)
        rxGd = rx.PyDiGraph(multigraph=False); rxGd.add_nodes_from(range(len(ids)))
        rxGdw = rx.PyDiGraph(multigraph=False); rxGdw.add_nodes_from(range(len(ids)))
        for f, t, w, dsum in zip(Edges["From"], Edges["To"], Edges["Weight"], Edges["AllowedDir"]):
            fi, ti = id2idx[f], id2idx[t]
            ww = float(w) if np.isfinite(w) else 0.0
            if (dsum in ("UNKNOWN",) and Undirected) or dsum == "BOTH":
                rxGd.add_edge(fi, ti, None); rxGd.add_edge(ti, fi, None)
                rxGdw.add_edge(fi, ti, ww);  rxGdw.add_edge(ti, fi, ww)
            elif dsum == "FWD":
                rxGd.add_edge(fi, ti, None); rxGdw.add_edge(fi, ti, ww)
            elif dsum == "BWD":
                rxGd.add_edge(ti, fi, None); rxGdw.add_edge(ti, fi, ww)
            elif dsum == "NONE":
                pass
            else:
                if Undirected:
                    rxGd.add_edge(fi, ti, None); rxGd.add_edge(ti, fi, None)
                    rxGdw.add_edge(fi, ti, ww);  rxGdw.add_edge(ti, fi, ww)

        return {
            "Nodes": Nodes, "Edges": Edges,
            "G": rxG, "Gw": rxGw, "Gd": rxGd, "Gdw": rxGdw,
            "nodeRow": lambda name: name2row[str(name)],
        }

    # NetworkX (default)
    node_ids = Nodes["ConnectorID"].tolist()
    G = nx.Graph();  Gw = nx.Graph()
    G.add_nodes_from(node_ids); Gw.add_nodes_from(node_ids)
    G.add_edges_from(zip(Edges["From"], Edges["To"]))
    Gw.add_weighted_edges_from(
        (str(f), str(t), float(w) if np.isfinite(w) else 0.0)
        for f, t, w in zip(Edges["From"], Edges["To"], Edges["Weight"])
    )

    Gd = nx.DiGraph(); Gdw = nx.DiGraph()
    Gd.add_nodes_from(node_ids); Gdw.add_nodes_from(node_ids)
    for f, t, w, dsum in zip(Edges["From"], Edges["To"], Edges["Weight"], Edges["AllowedDir"]):
        f = str(f); t = str(t); ww = float(w) if np.isfinite(w) else 0.0
        if (dsum in ("UNKNOWN",) and Undirected) or dsum == "BOTH":
            Gd.add_edge(f, t); Gd.add_edge(t, f)
            Gdw.add_edge(f, t, weight=ww); Gdw.add_edge(t, f, weight=ww)
        elif dsum == "FWD":
            Gd.add_edge(f, t); Gdw.add_edge(f, t, weight=ww)
        elif dsum == "BWD":
            Gd.add_edge(t, f); Gdw.add_edge(t, f, weight=ww)
        elif dsum == "NONE":
            pass
        else:
            if Undirected:
                Gd.add_edge(f, t); Gd.add_edge(t, f)
                Gdw.add_edge(f, t, weight=ww); Gdw.add_edge(t, f, weight=ww)

    return {
        "Nodes": Nodes, "Edges": Edges,
        "G": G, "Gw": Gw, "Gd": Gd, "Gdw": Gdw,
        "nodeRow": lambda name: name2row[str(name)],
    }


# ======================================================================
# Public: lane graph (strict lane-level, using attributes)
# ======================================================================
# ======================================================================
# Public: lane graph (strict lane-level, using attributes)
#   FIX: make CP ids == Nodes.node_id (stable, in same id-space everywhere)
# ======================================================================
def build_simple_lane_graph(
    Sxy: Dict[str, Any],
    SA: Dict[str, Any],
    node_merge_eps: float = 0.0,
    backend: str = "networkx",  # "networkx" (default) or "retworkx"
):
    """
    Per-lane directed graph from attributes (strict (group,lane_number) mapping).
    Returns (G, Nodes, Edges, lane_endpoint_map)

    FIXED BEHAVIOR:
      - CP ids == Nodes['node_id'] == the ids used in Edges['FromN'/'ToN'].
      - IDs are made deterministic by sorting unique CP coordinates (X,Y)
        and remapping all endpoint references accordingly.
    """
    use_rx = (backend == "retworkx") and _HAS_RX

    lanes = Sxy.get("lanes") or []
    nL = len(lanes)
    if nL == 0:
        if use_rx:
            G = rx.PyDiGraph()
        else:
            G = nx.DiGraph()
        Nodes = pd.DataFrame(columns=["node_id", "X", "Y"])
        Edges = pd.DataFrame(columns=[
            "E", "FromN", "ToN", "Weight", "GroupID", "LaneNo", "Dir",
            "left_lane_boundary_number", "right_lane_boundary_number",
            "FromEnd", "ToEnd"
        ])
        lane_endpoint_map = pd.DataFrame(columns=["GroupID", "LaneNo", "Endpoint", "X", "Y", "node_id"])
        return G, Nodes, Edges, lane_endpoint_map

    # ---- 1) Collect lane endpoints quickly ----
    GID, LNO, EP, Xs, Ys = [], [], [], [], []
    bno_map: Dict[Tuple[str, int], Tuple[int, int]] = {}
    for rec in lanes:
        gid = str(rec.get("lane_group_ref"))
        ln = int(
            rec.get("lane_number")
            if rec.get("lane_number") is not None
            else rec.get("lane_index_within_group")
        )
        x = np.asarray(rec["x"], dtype=np.float64).ravel()
        y = np.asarray(rec["y"], dtype=np.float64).ravel()

        lbno = rec.get("left_lane_boundary_number")
        rbno = rec.get("right_lane_boundary_number")
        bno_map[(gid, ln)] = (
            int(lbno) if lbno is not None else np.nan,
            int(rbno) if rbno is not None else np.nan,
        )

        if x.size == 0:
            continue
        xs, ys = x[0], y[0]
        xe, ye = (x[0], y[0]) if x.size == 1 else (x[-1], y[-1])

        GID.extend((gid, gid))
        LNO.extend((ln, ln))
        EP.extend(("start", "end"))
        Xs.extend((xs, xe))
        Ys.extend((ys, ye))

    if not Xs:
        if use_rx:
            G = rx.PyDiGraph()
        else:
            G = nx.DiGraph()
        Nodes = pd.DataFrame(columns=["node_id", "X", "Y"])
        Edges = pd.DataFrame(columns=[
            "E", "FromN", "ToN", "Weight", "GroupID", "LaneNo", "Dir",
            "left_lane_boundary_number", "right_lane_boundary_number",
            "FromEnd", "ToEnd"
        ])
        lane_endpoint_map = pd.DataFrame(columns=["GroupID", "LaneNo", "Endpoint", "X", "Y", "node_id"])
        return G, Nodes, Edges, lane_endpoint_map

    GID = np.asarray(GID, dtype=object)
    LNO = np.asarray(LNO, dtype=np.int32)
    EP  = np.asarray(EP,  dtype=object)
    Xs  = np.asarray(Xs,  dtype=np.float64)
    Ys  = np.asarray(Ys,  dtype=np.float64)

    # ---- 2) Node dedup (vectorized) ----
    # _quantize_xy returns unique coords (Xn,Yn) and inv mapping per endpoint record.
    Xn, Yn, inv = _quantize_xy(Xs, Ys, node_merge_eps)   # inv maps each endpoint -> [0..nUnique-1]
    inv = inv.astype(np.int64, copy=False)

    # ===== FIX: make CP ids deterministic and use them everywhere =====
    # Sort unique points by (X,Y), remap inv accordingly, then CP ids become 0..nUnique-1 in sorted order.
    if len(Xn) > 0:
        order = np.lexsort((Yn, Xn))  # primary X, then Y (lexsort uses last key as primary)
        # order maps new_index -> old_index. We need old_index -> new_index:
        old2new = np.empty_like(order, dtype=np.int64)
        old2new[order] = np.arange(len(order), dtype=np.int64)

        Xn_sorted = Xn[order]
        Yn_sorted = Yn[order]
        Nidx = old2new[inv].astype(np.int32, copy=False)  # endpoint record -> CP id
    else:
        Xn_sorted = Xn
        Yn_sorted = Yn
        Nidx = inv.astype(np.int32, copy=False)

    lane_endpoint_map = pd.DataFrame({
        "GroupID": GID, "LaneNo": LNO, "Endpoint": EP, "X": Xs, "Y": Ys, "node_id": Nidx
    })

    Nodes = pd.DataFrame({"node_id": np.arange(len(Xn_sorted), dtype=np.int32),
                          "X": Xn_sorted.astype(np.float64, copy=False),
                          "Y": Yn_sorted.astype(np.float64, copy=False)})

    # ---- 3) Edges without groupby-apply ----
    df = lane_endpoint_map
    is_start = (df["Endpoint"].values == "start")
    is_end   = ~is_start
    key = pd.MultiIndex.from_arrays([df["GroupID"].values, df["LaneNo"].values])

    def _first_index_per_key(mask: np.ndarray) -> pd.Series:
        mkey = key[mask]
        if mkey.size == 0:
            return pd.Series([], dtype=np.int64)
        order = np.lexsort((mkey.codes[1], mkey.codes[0]))
        mkey_sorted = mkey[order]
        newgrp = np.empty(len(mkey_sorted), dtype=bool)
        newgrp[0] = True
        newgrp[1:] = (
            (mkey_sorted.codes[0][1:] != mkey_sorted.codes[0][:-1])
            | (mkey_sorted.codes[1][1:] != mkey_sorted.codes[1][:-1])
        )
        idx_in_mask = np.nonzero(mask)[0][order]
        return pd.Series(
            idx_in_mask[newgrp],
            index=pd.MultiIndex.from_arrays(
                [
                    mkey_sorted.levels[0][mkey_sorted.codes[0][newgrp]],
                    mkey_sorted.levels[1][mkey_sorted.codes[1][newgrp]],
                ]
            ),
        )

    i_start = _first_index_per_key(is_start)
    i_end   = _first_index_per_key(is_end)
    common  = i_start.index.intersection(i_end.index)

    if len(common) == 0:
        # Graph with nodes, no edges
        if use_rx:
            G = rx.PyDiGraph(); G.add_nodes_from(range(len(Nodes)))
        else:
            G = nx.DiGraph()
            for _, r in Nodes.iterrows():
                G.add_node(int(r["node_id"]), x=float(r["X"]), y=float(r["Y"]))
        Edges = pd.DataFrame(columns=[
            "E", "FromN", "ToN", "Weight", "GroupID", "LaneNo", "Dir",
            "left_lane_boundary_number", "right_lane_boundary_number",
            "FromEnd", "ToEnd"
        ])
        return G, Nodes, Edges, lane_endpoint_map

    idx_s = i_start.loc[common].to_numpy()
    idx_e = i_end.loc[common].to_numpy()

    startN = Nidx[idx_s]   # CP ids (already!)
    endN   = Nidx[idx_e]
    gid_kept = GID[idx_s]
    ln_kept  = LNO[idx_s]

    xs = Nodes["X"].to_numpy()[startN]
    ys = Nodes["Y"].to_numpy()[startN]
    xe = Nodes["X"].to_numpy()[endN]
    ye = Nodes["Y"].to_numpy()[endN]
    dist = np.hypot(xe - xs, ye - ys)

    # Direction map from SA
    dir_map: Dict[Tuple[str, int], str] = {}
    by_group = SA.get("by_group", {})
    for gid in set(map(str, gid_kept)):
        g = by_group.get(gid)
        if not g:
            continue
        for ln, lane in (g.get("lanes", {}) or {}).items():
            dirs = {str(d).upper() for d in (lane.get("directions") or [])}
            if "FORWARD" in dirs and "BACKWARD" in dirs:
                dsum = "BOTH"
            elif "FORWARD" in dirs:
                dsum = "FWD"
            elif "BACKWARD" in dirs:
                dsum = "BWD"
            else:
                dsum = "NONE"
            dir_map[(gid, int(ln))] = dsum

    rows = []
    append = rows.append
    skipped_none = 0
    for s, e, d, g, l in zip(startN, endN, dist, gid_kept, ln_kept):
        lbno, rbno = bno_map.get((str(g), int(l)), (np.nan, np.nan))
        dsum = dir_map.get((str(g), int(l)), "NONE")
        if dsum == "FWD":
            append((s, e, d, g, int(l), "FORWARD", lbno, rbno))
        elif dsum == "BWD":
            append((e, s, d, g, int(l), "BACKWARD", lbno, rbno))
        elif dsum == "BOTH":
            append((s, e, d, g, int(l), "FORWARD",  lbno, rbno))
            append((e, s, d, g, int(l), "BACKWARD", lbno, rbno))
        else:
            skipped_none += 1

    edge_cols = ["FromN", "ToN", "Weight", "GroupID", "LaneNo", "Dir",
                 "left_lane_boundary_number", "right_lane_boundary_number"]
    Edges = pd.DataFrame.from_records(rows, columns=edge_cols)
    if not Edges.empty:
        Edges["E"] = np.arange(len(Edges), dtype=np.int32)
        Edges = Edges[["E", "FromN", "ToN", "Weight", "GroupID", "LaneNo", "Dir",
                       "left_lane_boundary_number", "right_lane_boundary_number"]].copy()
    else:
        Edges = pd.DataFrame(columns=["E", "FromN", "ToN", "Weight", "GroupID", "LaneNo", "Dir",
                                      "left_lane_boundary_number", "right_lane_boundary_number"])

    # ===================== enrich Edges & Nodes in-place =====================
    if not Edges.empty:
        dirv = Edges["Dir"].astype(str).str.upper().values
        Edges["FromEnd"] = np.where(dirv == "FORWARD", "start",
                             np.where(dirv == "BACKWARD", "end", "unknown"))
        Edges["ToEnd"]   = np.where(dirv == "FORWARD", "end",
                             np.where(dirv == "BACKWARD", "start", "unknown"))

    if not lane_endpoint_map.empty:
        lem = lane_endpoint_map.copy()
        lem["GroupID"] = lem["GroupID"].astype(str)
        lem["LaneNo"]  = lem["LaneNo"].astype(int)
        lem["Endpoint"]= lem["Endpoint"].astype(str)
        lane_end_keys = (
            lem.groupby("node_id")[["GroupID","LaneNo","Endpoint"]]
               .apply(lambda df: [ (str(g), int(l), str(ep)) for g,l,ep in df.to_numpy() ])
               .rename("lane_end_keys")
        )
        Nodes = Nodes.merge(lane_end_keys, how="left", left_on="node_id", right_index=True)
        Nodes["lane_end_keys"] = Nodes["lane_end_keys"].apply(lambda v: v if isinstance(v, list) else [])
    else:
        Nodes["lane_end_keys"] = [[] for _ in range(len(Nodes))]

    def _uniq_groups(keys): return sorted({g for (g,_,_) in keys})
    def _uniq_lnos(keys):   return sorted({int(l) for (_,l,_) in keys})
    def _end_counts(keys):
        c = {"start":0, "end":0}
        for _,_,ep in keys:
            if ep in c: c[ep] += 1
        return c

    Nodes["lane_groups"]     = Nodes["lane_end_keys"].apply(_uniq_groups)
    Nodes["lane_lnos"]       = Nodes["lane_end_keys"].apply(_uniq_lnos)
    Nodes["lane_end_counts"] = Nodes["lane_end_keys"].apply(_end_counts)

    if not Edges.empty:
        touch = pd.concat([
            Edges[['FromN','E']].rename(columns={'FromN':'node_id'}),
            Edges[['ToN','E']].rename(columns={'ToN':'node_id'})
        ], ignore_index=True)

        touch['node_id'] = touch['node_id'].astype(Nodes['node_id'].dtype, copy=False)
        incident_edges = (
            touch.groupby('node_id')['E']
                .apply(lambda s: sorted(s.tolist()))
                .rename('incident_edges')
        )
        Nodes = Nodes.merge(incident_edges, how='left', left_on='node_id', right_index=True)
        Nodes['incident_edges'] = Nodes['incident_edges'].apply(lambda v: v if isinstance(v, list) else [])

        meta_from = Edges[['FromN','E','GroupID','LaneNo','Dir','FromEnd']].copy()
        meta_from.columns = ['node_id','E','GroupID','LaneNo','Dir','end_at_node']
        meta_to   = Edges[['ToN','E','GroupID','LaneNo','Dir','ToEnd']].copy()
        meta_to.columns   = ['node_id','E','GroupID','LaneNo','Dir','end_at_node']
        meta = pd.concat([meta_from, meta_to], ignore_index=True)

        meta['node_id'] = meta['node_id'].astype(Nodes['node_id'].dtype, copy=False)
        meta['GroupID'] = meta['GroupID'].astype(str)
        meta['LaneNo']  = meta['LaneNo'].astype(int)
        meta['Dir']     = meta['Dir'].astype(str)
        meta['end_at_node'] = meta['end_at_node'].astype(str)

        incident_edge_meta = (
            meta.groupby('node_id')[['E','GroupID','LaneNo','Dir','end_at_node']]
                .apply(lambda df: [tuple(x) for x in df.to_numpy()])
                .rename('incident_edge_meta')
        )
        Nodes = Nodes.merge(incident_edge_meta, how='left', left_on='node_id', right_index=True)
        Nodes['incident_edge_meta'] = Nodes['incident_edge_meta'].apply(lambda v: v if isinstance(v, list) else [])
    else:
        Nodes['incident_edges'] = [[] for _ in range(len(Nodes))]
        Nodes['incident_edge_meta'] = [[] for _ in range(len(Nodes))]

    # ---- 4) Build graph (backend) ----
    if use_rx:
        G = rx.PyDiGraph(multigraph=False); G.add_nodes_from(range(len(Nodes)))
        for f, t, w in Edges[["FromN", "ToN", "Weight"]].to_numpy():
            G.add_edge(int(f), int(t), float(w))
        return G, Nodes, Edges, lane_endpoint_map

    G = nx.DiGraph()
    G.add_nodes_from((int(n), {"x": float(x), "y": float(y)})
                     for n, x, y in Nodes[["node_id", "X", "Y"]].to_numpy())
    G.add_weighted_edges_from(
        (int(f), int(t), float(w)) for f, t, w in Edges[["FromN", "ToN", "Weight"]].to_numpy()
    )
    for f, t, eidx, gid, ln, d in Edges[["FromN", "ToN", "E", "GroupID", "LaneNo", "Dir"]].itertuples(index=False, name=None):
        G.edges[int(f), int(t)].update({"eidx": int(eidx), "group": str(gid), "lane": int(ln), "dir": str(d)})

    return G, Nodes, Edges, lane_endpoint_map



# ======================================================================
# Plotting (unchanged API; expects NetworkX inputs)
# ======================================================================
def plot_simple_lane_graph(
    ax,
    Sxy,
    Nodes,
    Edges,
    show_node_ids: bool = True,
    show_edge_ids: bool = True,
    lane_gray: float = 0.85,
    node_size: float = 8,
    edge_width: float = 1.5,
    quiver_scale: float = 30,
    quiver_width: float = 0.0025,
    # --- NEW ---
    window_rect: Optional[Tuple[float, float, float, float]] = None,  # ENU (xmin,ymin,xmax,ymax)
    single_color: Optional[Any] = None,  # e.g. "k" or (r,g,b)
):
    """
    Plot a simple lane graph in ENU.

    window_rect: if given, only edges whose FromN or ToN node lies inside the
                 rectangle are shown; node set is limited to endpoints of
                 those visible edges. Lanes (light gray) are still drawn
                 for context (unclipped).
    single_color: if given, use the same color for edges, arrows, and nodes.
    """
    ax.set_aspect("equal", adjustable="datalim")

    # ------------------------------------------------------------------
    # Background lanes (context), unchanged
    # ------------------------------------------------------------------
    for rec in (Sxy.get("lanes") or []):
        x = np.asarray(rec["x"], float).reshape(-1)
        y = np.asarray(rec["y"], float).reshape(-1)
        if x.size >= 2:
            ax.plot(x, y, "-", color=(lane_gray, lane_gray, lane_gray), linewidth=1.0, zorder=1)

    # ------------------------------------------------------------------
    # Build filtered node/edge views (windowed if requested)
    # ------------------------------------------------------------------
    nodes_df = Nodes
    edges_df = Edges

    if window_rect is not None:
        x1, y1, x2, y2 = window_rect
        xmin, xmax = (x1, x2) if x1 <= x2 else (x2, x1)
        ymin, ymax = (y1, y2) if y1 <= y2 else (y2, y1)

        # --- robust, flat arrays ---
        Xv = np.asarray(Nodes["X"].to_numpy(), dtype=float).reshape(-1)
        Yv = np.asarray(Nodes["Y"].to_numpy(), dtype=float).reshape(-1)

        inside_mask = (Xv >= xmin) & (Xv <= xmax) & (Yv >= ymin) & (Yv <= ymax)
        # inside_mask is now a 1-D boolean ndarray aligned to Nodes' index

        if inside_mask.any():
            inside_nodes = set(Nodes.loc[inside_mask, "node_id"].astype(int).tolist())

            e_mask = Edges["FromN"].isin(inside_nodes) | Edges["ToN"].isin(inside_nodes)
            edges_df = Edges.loc[e_mask].copy()

            touched = set(edges_df["FromN"].astype(int).tolist()) | set(edges_df["ToN"].astype(int).tolist())
            nodes_df = Nodes.loc[Nodes["node_id"].isin(touched)].copy()
        else:
            edges_df = Edges.iloc[0:0]
            nodes_df = Nodes.iloc[0:0]

    # Fast maps for coords
    X = nodes_df.set_index("node_id")["X"].to_dict()
    Y = nodes_df.set_index("node_id")["Y"].to_dict()

    # ------------------------------------------------------------------
    # Edges + arrows (optionally single color)
    # ------------------------------------------------------------------
    for _, e in edges_df.iterrows():
        f = int(e["FromN"]); t = int(e["ToN"])
        x0, y0 = X[f], Y[f]
        x1, y1 = X[t], Y[t]
        c = single_color  # None -> Matplotlib default cycle

        ax.plot([x0, x1], [y0, y1], "-", linewidth=edge_width, alpha=0.9, zorder=2, color=c)

        dx, dy = x1 - x0, y1 - y0
        mx, my = x0 + 0.6 * dx, y0 + 0.6 * dy
        ax.quiver(
            mx, my, dx, dy,
            angles="xy", scale_units="xy", scale=quiver_scale, width=quiver_width,
            zorder=3, color=c
        )

    # ------------------------------------------------------------------
    # Nodes + labels (optionally single color)
    # ------------------------------------------------------------------
    if len(nodes_df) > 0:
        ax.scatter(nodes_df["X"], nodes_df["Y"], s=node_size**2 / 4, c=(single_color or "k"), zorder=4)

    if show_node_ids:
        for _, r in nodes_df.iterrows():
            ax.text(r["X"], r["Y"], f'N{int(r["node_id"])}', fontsize=7,
                    color=(single_color or "k"), ha="left", va="bottom", zorder=5)

    if show_edge_ids:
        for _, e in edges_df.iterrows():
            f = int(e["FromN"]); t = int(e["ToN"])
            mx = 0.5 * (X[f] + Y.get(t, 0) * 0 + X[t])  # just use X dict twice to avoid recompute
            my = 0.5 * (Y[f] + X.get(t, 0) * 0 + Y[t])
            ax.text(mx, my, f'E{int(e["E"])}', fontsize=7, ha="center", va="center", zorder=5,
                    color=(single_color or None))

    ax.set_xlabel("E (m)")
    ax.set_ylabel("N (m)")
    ax.grid(True, alpha=0.2)


def _main_endpoints(G, main_id):
    """Return (s,e) of the main edge with this main_id."""
    for u, v, d in G.edges(data=True):
        if d.get('role') == 'main' and d.get('main_id') == main_id:
            return int(d.get('start', u)), int(d.get('end', v))
    raise KeyError(f"main_id={main_id} not found in graph")

def _firsthop_nodes(G, main_id, phase):
    """
    Minimal extension nodes (first-hop) for a main:
      - phase='backward' : edges that START at main start node
      - phase='forward'  : edges that START at main end   node
    If no such extension exists, returns {start} (for backward) or {end} (for forward).
    """
    s, e = _main_endpoints(G, main_id)
    anchor = s if phase == 'backward' else e
    out = set()
    for u, v, d in G.out_edges(anchor, data=True):
        if d.get('role') == 'extension' and d.get('main_id') == main_id and d.get('phase') == phase:
            out.add(int(d.get('end', v)))  # the node reached by the first hop
    if not out:
        out.add(anchor)
    return out


def _coerce_nodeset(x):
    """leaves[...] may be set, list, dict-of-None, etc. → return a set of ints."""
    if x is None:
        return set()
    if isinstance(x, dict):
        # keys are node ids
        return {int(k) for k in x.keys()}
    if isinstance(x, (set, list, tuple)):
        return {int(n) for n in x}
    # single value?
    try:
        return {int(x)}
    except Exception:
        return set()

def _leaf_nodes_for(leaves, link_id, phase):
    """
    Find the leaf-node set for a specific link_id and phase {'backward','forward'}.
    Your leaves looks like:
      leaves = {'S-N': {'backward': {...}, 'forward': {...}, 'link': 711}, 'N-S': {...}, ...}
    """
    link_id = int(link_id)
    for lab, rec in leaves.items():
        if int(rec.get('link', -1)) == link_id:
            return _coerce_nodeset(rec.get(phase)), rec['main_start'], rec['main_end'], rec['fwd_set'], rec['back_set'], lab #rec['fwd_set_end'], rec['back_set_start'], lab
    return set()


def nodes_on_paths_start_to_targets(
    G: nx.DiGraph,
    s: Hashable,
    targets: Iterable[Hashable],
    require_all_targets: bool = False,
) -> Set[Hashable]:
    """
    Return nodes that lie on paths from start node s to the target set.

    If require_all_targets = False (default):
        returns nodes that lie on a path s -> t for at least one t in targets.
    If require_all_targets = True:
        returns nodes that lie on a path s -> t for *every* t in targets.

    Works for DAGs (and any DiGraph). Includes s and any target nodes when appropriate.
    """
    targets = [t for t in targets if t in G]
    if s not in G or not targets:
        return set()

    # Forward reachability from s (include s itself)
    reach_from_s = nx.descendants(G, s) | {s}

    # Nodes that can reach targets:
    #   - any-target mode: union of ancestors of each t (plus t)
    #   - all-targets mode: intersection across targets
    if not require_all_targets:
        reach_to_any = set()
        for t in targets:
            reach_to_any |= nx.ancestors(G, t) | {t}
        can_reach = reach_to_any
    else:
        # Start with all nodes; shrink by intersecting per target
        can_reach = set(G.nodes)
        for t in targets:
            can_reach &= (nx.ancestors(G, t) | {t})

    # Nodes lying on s→(any/all)targets paths are in both sets
    return reach_from_s & can_reach

def plot_lane_background(
    ax,
    Sxy,
    lane_gray: float = 0.85,
):
    """
    Plot a simple lane graph in ENU.

    window_rect: if given, only edges whose FromN or ToN node lies inside the
                 rectangle are shown; node set is limited to endpoints of
                 those visible edges. Lanes (light gray) are still drawn
                 for context (unclipped).
    single_color: if given, use the same color for edges, arrows, and nodes.
    """
    if ax == None:
        fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="datalim")

    # ------------------------------------------------------------------
    # Background lanes (context), unchanged
    # ------------------------------------------------------------------
    for rec in (Sxy.get("lanes") or []):
        x = np.asarray(rec["x"], float).reshape(-1)
        y = np.asarray(rec["y"], float).reshape(-1)
        if x.size >= 2:
            ax.plot(x, y, "-", color=(lane_gray, lane_gray, lane_gray), linewidth=1.0, zorder=1)

    ax.set_xlabel("E (m)")
    ax.set_ylabel("N (m)")
    ax.grid(True, alpha=0.2)
    return ax

# ======================================================================
# Public helpers (module-level, kept for compatibility with exports.py)
# Signature intentionally matches your original usage.
# ======================================================================
def endpoints_from_group_xy(Sxy, mRef, lanesByGroup, gid: str):
    xs = ys = xe = ye = float("nan")
    if gid in mRef:
        r = Sxy["reference"][mRef[gid]]
        x, y = r.get("x"), r.get("y")
        if x is not None and len(x) > 0:
            return float(x[0]), float(y[0]), float(x[-1]), float(y[-1])
    if gid in lanesByGroup:
        for u in lanesByGroup[gid]:
            L = Sxy["lanes"][u]; x, y = L.get("x"), L.get("y")
            if x is not None and len(x) > 0:
                return float(x[0]), float(y[0]), float(x[-1]), float(y[-1])
    return xs, ys, xe, ye

def poly_for_group_xy(Sxy, mRef, lanesByGroup, gid: str):
    if gid in mRef:
        r = Sxy["reference"][mRef[gid]]; x, y = r.get("x"), r.get("y")
        if x is not None and len(x) > 0: return np.asarray(x, float), np.asarray(y, float)
    if gid in lanesByGroup:
        for u in lanesByGroup[gid]:
            L = Sxy["lanes"][u]; x, y = L.get("x"), L.get("y")
            if x is not None and len(x) > 0: return np.asarray(x, float), np.asarray(y, float)
    return np.array([], float), np.array([], float)

def _in_rect_xy(x: float, y: float,
                rect: Optional[Tuple[float, float, float, float]]) -> bool:
    if rect is None or not np.isfinite(x) or not np.isfinite(y):
        return False
    x1, y1, x2, y2 = rect
    xmin, xmax = (x1, x2) if x1 <= x2 else (x2, x1)
    ymin, ymax = (y1, y2) if y1 <= y2 else (y2, y1)
    return (xmin <= x <= xmax) and (ymin <= y <= ymax)

import numpy as np
import pandas as pd

def attach_node_lane_metadata(Nodes: pd.DataFrame,
                              Edges: pd.DataFrame,
                              lane_endpoint_map: pd.DataFrame):
    """
    Augment Nodes and Edges with lane-end metadata.

    Inputs (as returned by build_simple_lane_graph):
      Nodes: columns ['node_id','X','Y']
      Edges: columns ['E','FromN','ToN','Weight','GroupID','LaneNo','Dir', ...]
      lane_endpoint_map: columns ['GroupID','LaneNo','Endpoint','X','Y','node_id']

    Returns:
      Nodes2 (copy with extra columns), Edges2 (copy with FromEnd/ToEnd)
    """
    Nodes2 = Nodes.copy()
    Edges2 = Edges.copy()

    # --- 1) Add FromEnd/ToEnd to Edges (physical lane endpoints w.r.t. direction)
    # Forward edges go start->end; backward edges go end->start (per your builder).
    if not Edges2.empty:
        dirv = Edges2['Dir'].astype(str).str.upper().values
        from_end = np.where(dirv == 'FORWARD', 'start',
                    np.where(dirv == 'BACKWARD', 'end', 'unknown'))
        to_end   = np.where(dirv == 'FORWARD', 'end',
                    np.where(dirv == 'BACKWARD', 'start', 'unknown'))
        Edges2['FromEnd'] = from_end
        Edges2['ToEnd']   = to_end

    # --- 2) Build quick maps from lane endpoints to node ids
    # lane_endpoint_map: one row per (GroupID, LaneNo, Endpoint) with its node N
    lem = lane_endpoint_map.copy()
    if lem.empty:
        # still create empty columns on Nodes
        Nodes2['lane_end_keys']   = [[] for _ in range(len(Nodes2))]
        Nodes2['lane_groups']     = [[] for _ in range(len(Nodes2))]
        Nodes2['lane_lnos']       = [[] for _ in range(len(Nodes2))]
        Nodes2['lane_end_counts'] = [{} for _ in range(len(Nodes2))]
        Nodes2['incident_edges']  = [[] for _ in range(len(Nodes2))]
        Nodes2['incident_edge_meta'] = [[] for _ in range(len(Nodes2))]
        return Nodes2, Edges2

    lem['GroupID'] = lem['GroupID'].astype(str)
    lem['LaneNo']  = lem['LaneNo'].astype(int)
    lem['Endpoint']= lem['Endpoint'].astype(str)

    # Per-node list of (GroupID, LaneNo, Endpoint)
    grouped = (lem.groupby('node_id')[['GroupID','LaneNo','Endpoint']]
                  .apply(lambda df: list(map(tuple, df.to_numpy())))
                  .rename('lane_end_keys'))
    # Merge into Nodes
    Nodes2 = Nodes2.merge(grouped, how='left', left_on='node_id', right_index=True)
    Nodes2['lane_end_keys'] = Nodes2['lane_end_keys'].apply(lambda v: v if isinstance(v, list) else [])

    # Derive convenience lists per node
    def _uniq_groups(keys):
        return sorted({g for (g, _, _) in keys})
    def _uniq_lnos(keys):
        return sorted({int(l) for (_, l, _) in keys})
    def _end_counts(keys):
        c = {'start':0, 'end':0}
        for _,_,ep in keys: 
            if ep in c: c[ep]+=1
        return c

    Nodes2['lane_groups']     = Nodes2['lane_end_keys'].apply(_uniq_groups)
    Nodes2['lane_lnos']       = Nodes2['lane_end_keys'].apply(_uniq_lnos)
    Nodes2['lane_end_counts'] = Nodes2['lane_end_keys'].apply(_end_counts)

    # --- 3) Incident edges and edge meta per node
    if not Edges2.empty:
        # Build lists for each node
        incE_from = Edges2.groupby('FromN')['E'].apply(list)
        incE_to   = Edges2.groupby('ToN')['E'].apply(list)
        # initialize
        Nodes2['incident_edges'] = [[] for _ in range(len(Nodes2))]
        for nid, elist in incE_from.items():
            Nodes2.loc[Nodes2.N==nid, 'incident_edges'].iat[0].extend(elist)
        for nid, elist in incE_to.items():
            Nodes2.loc[Nodes2.N==nid, 'incident_edges'].iat[0].extend(elist)

        # Detailed meta: (E, GroupID, LaneNo, Dir, end_at_node)
        meta_rows = []
        for row in Edges2[['E','FromN','ToN','GroupID','LaneNo','Dir','FromEnd','ToEnd']].itertuples(index=False):
            E, fN, tN, g, l, d, fEnd, tEnd = row
            meta_rows.append((fN, (E, str(g), int(l), str(d), str(fEnd))))
            meta_rows.append((tN, (E, str(g), int(l), str(d), str(tEnd))))
        meta_df = pd.DataFrame(meta_rows, columns=['node_id','meta'])
        meta_grouped = meta_df.groupby('node_id')['meta'].apply(list)

        Nodes2['incident_edge_meta'] = [[] for _ in range(len(Nodes2))]
        for nid, mlist in meta_grouped.items():
            Nodes2.loc[Nodes2.N==nid, 'incident_edge_meta'].iat[0] = mlist
    else:
        Nodes2['incident_edges'] = [[] for _ in range(len(Nodes2))]
        Nodes2['incident_edge_meta'] = [[] for _ in range(len(Nodes2))]

    return Nodes2, Edges2

import pandas as pd
import numpy as np

def print_nodes(Nodes: pd.DataFrame, max_rows: int = 10, max_list_len: int = 3):
    """Pretty print Nodes DataFrame."""
    if Nodes.empty:
        print("⚠️  No nodes.")
        return

    def _fmt(v):
        if isinstance(v, (list, tuple)):
            if len(v) > max_list_len:
                return f"[{', '.join(map(str, v[:max_list_len]))}, …]"
            return str(v)
        if isinstance(v, dict):
            return str({k:v for k,v in list(v.items())[:max_list_len]}) + (" …" if len(v)>max_list_len else "")
        return v

    df = Nodes.copy()
    for c in df.columns:
        df[c] = df[c].apply(_fmt)
    print("=== NODES ===")
    with pd.option_context("display.max_rows", max_rows, "display.max_colwidth", 80):
        print(df.head(max_rows).to_string(index=False))
    if len(df) > max_rows:
        print(f"… ({len(df)} total rows)")

def print_edges(Edges: pd.DataFrame, max_rows: int = 10):
    """Pretty print Edges DataFrame."""
    if Edges.empty:
        print("⚠️  No edges.")
        return
    print("=== EDGES ===")
    cols = ["E","FromN","ToN","Weight","GroupID","LaneNo","Dir","FromEnd","ToEnd"]
    cols = [c for c in cols if c in Edges.columns]
    df = Edges[cols].copy()
    with pd.option_context("display.max_rows", max_rows, "display.precision", 3):
        print(df.head(max_rows).to_string(index=False))
    if len(df) > max_rows:
        print(f"… ({len(df)} total rows)")

import math, shutil, textwrap
import pandas as pd
import numpy as np

# -------------------- Small utilities --------------------
def _term_width(default=120):
    try:
        return shutil.get_terminal_size().columns
    except Exception:
        return default

def _fmt_float(x, prec=3):
    try:
        return f"{float(x): .{prec}f}".strip()
    except Exception:
        return str(x)

def _summ_lane_end_keys(keys, max_pairs=4):
    if not keys:
        return "-"
    d = {}
    for g, l, ep in keys:
        s = d.setdefault((str(g), int(l)), set())
        s.add("s" if str(ep) == "start" else "e" if str(ep) == "end" else "?")
    parts = []
    for (g, l), se in d.items():
        tag = "".join(sorted(se)) or "-"
        parts.append(f"{g}:{l}[{tag}]")
    parts.sort()
    if len(parts) > max_pairs:
        return "; ".join(parts[:max_pairs]) + f"; +{len(parts)-max_pairs} more"
    return "; ".join(parts)

def _summ_incident_edges(lst, max_e=6):
    if not lst: return "deg=0"
    s = f"deg={len(lst)} "
    if len(lst) <= max_e:
        return s + "E" + ",E".join(map(str, lst))
    return s + "E" + ",E".join(map(str, lst[:max_e])) + f", +{len(lst)-max_e}"

def _summ_incident_meta(tuples_, max_items=3):
    if not tuples_: return "-"
    out = []
    for (e, gid, ln, d, end) in tuples_[:max_items]:
        d1 = (str(d)[:1] if isinstance(d, str) else str(d))  # F/B
        end1 = "s" if str(end).startswith("start") else "e" if str(end).startswith("end") else "?"
        out.append(f"E{e}:{d1}@{end1} {gid}:{ln}")
    extra = ""
    if len(tuples_) > max_items:
        extra = f" +{len(tuples_)-max_items}"
    return "; ".join(out) + extra

def _nodes_bbox_mask(Nodes: pd.DataFrame, bbox):
    """Return boolean mask for Nodes within bbox=(xmin, ymin, xmax, ymax)."""
    if Nodes is None or Nodes.empty:
        return np.zeros(0, dtype=bool)
    if bbox is None:
        return np.ones(len(Nodes), dtype=bool)
    xmin, ymin, xmax, ymax = bbox
    x = Nodes["X"].to_numpy(dtype=float, copy=False)
    y = Nodes["Y"].to_numpy(dtype=float, copy=False)
    return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)

def _edges_bbox_mask(Edges: pd.DataFrame, Nodes: pd.DataFrame, bbox, mode="both"):
    """
    mode:
      - 'both'  : print edges when both endpoints are in bbox
      - 'either': print edges when at least one endpoint is in bbox
    """
    if Edges is None or Edges.empty:
        return np.zeros(0, dtype=bool)
    if bbox is None:
        return np.ones(len(Edges), dtype=bool)

    # Build node->in_bbox map
    nmask = _nodes_bbox_mask(Nodes, bbox)
    n_in = set(Nodes.loc[nmask, "node_id"].astype(int).tolist())

    f_in = Edges["FromN"].astype(int).isin(n_in).to_numpy()
    t_in = Edges["ToN"].astype(int).isin(n_in).to_numpy()
    if mode == "either":
        return f_in | t_in
    return f_in & t_in

# -------------------- Pretty Printers WITH bbox --------------------
def print_nodes_pretty(Nodes, max_rows=30, width=None, bbox=None):
    """
    One-line-per-node view, but filtered to bbox=(xmin, ymin, xmax, ymax).
    """
    if Nodes is None or len(Nodes) == 0:
        print("⚠️  No nodes.")
        return

    mask = _nodes_bbox_mask(Nodes, bbox)
    NN = Nodes.loc[mask]
    if NN.empty:
        print("ℹ️  No nodes inside bbox.")
        return

    W = width or _term_width()

    rows = []
    for _, r in NN.head(max_rows).iterrows():
        N  = int(r["node_id"]) if "node_id" in r else "?"
        X  = _fmt_float(r["X"]) if "X" in r else "-"
        Y  = _fmt_float(r["Y"]) if "Y" in r else "-"
        groups = ",".join(r["lane_groups"]) if "lane_groups" in r and isinstance(r["lane_groups"], list) else "-"
        lnos   = ",".join(map(str, r["lane_lnos"])) if "lane_lnos" in r and isinstance(r["lane_lnos"], list) else "-"

        ends = "-"
        if "lane_end_counts" in r and isinstance(r["lane_end_counts"], dict):
            s = int(r["lane_end_counts"].get("start",0))
            e = int(r["lane_end_counts"].get("end",0))
            ends = f"S{s}/E{e}"

        lek = _summ_lane_end_keys(r.get("lane_end_keys", []))
        deg = _summ_incident_edges(r.get("incident_edges", []))
        meta= _summ_incident_meta(r.get("incident_edge_meta", []))

        rows.append({
            "node_id": N, "X": X, "Y": Y,
            "groups": groups, "lnos": lnos, "ends": ends,
            "lane_end_keys": lek, "degree": deg, "incident": meta
        })

    headers = ["node_id","X","Y","groups","lnos","ends","lane_end_keys","degree","incident"]
    colw = {h: len(h) for h in headers}
    for rr in rows:
        for h in headers:
            colw[h] = max(colw[h], len(str(rr[h])))

    fixed = sum(colw[h] for h in ["node_id","X","Y","groups","lnos","ends","degree"]) + 8
    rem = max((width or _term_width()) - fixed, 30)
    lek_target = max(20, rem // 2)
    inc_target = max(20, rem - lek_target)
    colw["lane_end_keys"] = min(colw["lane_end_keys"], lek_target)
    colw["incident"]      = min(colw["incident"], inc_target)

    def pad(s,w): return str(s).ljust(w)
    head = (
        f"{pad('node_id',colw['node_id'])}  {pad('X',colw['X'])}  {pad('Y',colw['Y'])}  "
        f"{pad('groups',colw['groups'])}  {pad('lnos',colw['lnos'])}  {pad('ends',colw['ends'])}  "
        f"{pad('lane_end_keys',colw['lane_end_keys'])}  {pad('degree',colw['degree'])}  {pad('incident',colw['incident'])}"
    )
    print("=== NODES (compact, bbox) ===")
    print(head)
    print("-"*min(len(head), W))

    for rr in rows:
        lek_wrapped = textwrap.wrap(str(rr["lane_end_keys"]), width=colw["lane_end_keys"]) or ["-"]
        inc_wrapped = textwrap.wrap(str(rr["incident"]),      width=colw["incident"]) or ["-"]
        n_lines = max(len(lek_wrapped), len(inc_wrapped))
        for i in range(n_lines):
            left = ""
            if i == 0:
                left = (
                    f"{pad(rr['node_id'],colw['node_id'])}  {pad(rr['X'],colw['X'])}  {pad(rr['Y'],colw['Y'])}  "
                    f"{pad(rr['groups'],colw['groups'])}  {pad(rr['lnos'],colw['lnos'])}  {pad(rr['ends'],colw['ends'])}  "
                )
            else:
                left = (
                    f"{'':{colw['node_id']}}  {'':{colw['X']}}  {'':{colw['Y']}}  "
                    f"{'':{colw['groups']}}  {'':{colw['lnos']}}  {'':{colw['ends']}}  "
                )
            lek_part = lek_wrapped[i] if i < len(lek_wrapped) else ""
            inc_part = inc_wrapped[i] if i < len(inc_wrapped) else ""
            row = f"{left}{pad(lek_part,colw['lane_end_keys'])}  {pad(rr['degree'],colw['degree']) if i==0 else '':{colw['degree']}}  {pad(inc_part,colw['incident'])}"
            print(row)

    if len(NN) > max_rows:
        print(f"… ({len(NN)} nodes in bbox; showing first {max_rows})")

def print_edges_pretty(Edges, max_rows=30, width=None, bbox=None, Nodes=None, mode="both"):
    """
    Compact edge view filtered to bbox.
    - bbox: (xmin, ymin, xmax, ymax) on node coords
    - Nodes: required when bbox is given (to map endpoints to X,Y)
    - mode: 'both' (default) or 'either'
    """
    if Edges is None or len(Edges) == 0:
        print("⚠️  No edges.")
        return

    if bbox is not None:
        if Nodes is None or Nodes.empty:
            print("⚠️  bbox filtering requested but Nodes is missing/empty.")
            return
        mask = _edges_bbox_mask(Edges, Nodes, bbox, mode=mode)
        EE = Edges.loc[mask]
    else:
        EE = Edges

    if EE.empty:
        print("ℹ️  No edges inside bbox.")
        return

    W = width or _term_width()
    order = [c for c in [
        "E","FromN","ToN","Weight","GroupID","LaneNo","Dir","FromEnd","ToEnd",
        "left_lane_boundary_number","right_lane_boundary_number"
    ] if c in EE.columns]

    widths = {
        c: max(len(c), min(20, max((len(str(v)) for v in EE[c].head(max_rows)), default=0)))
        for c in order
    }

    def pad(s,w): return str(s).ljust(w)
    header = "  ".join(pad(c, widths[c]) for c in order)
    print(f"=== EDGES (compact, bbox; mode={mode}) ===")
    print(header)
    print("-"*min(len(header), W))

    for _, r in EE.head(max_rows).iterrows():
        row = []
        for c in order:
            val = _fmt_float(r[c]) if c == "Weight" else r[c]
            row.append(pad(val, widths[c]))
        print("  ".join(map(str,row)))

    if len(EE) > max_rows:
        print(f"… ({len(EE)} edges in bbox; showing first {max_rows})")
