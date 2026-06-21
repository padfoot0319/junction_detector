"""
Lane network builder.

Loads and projects HERE lane geometry/topology tiles into ENU coordinates,
attaches lane attributes, and builds connector and lane-level graphs.
"""
from __future__ import annotations
import json

from libs.geometry_helpers import (
    init_empty_geometry, parse_lane_geometry_tile, merge_geometry_structs,
    resolve_origin, project_polyset_to_xy, project_cellpoly_to_xy
)
from libs.topology import init_empty_topology, parse_lane_topology_tile, merge_topology_structs
from libs.attrs import ATTR_build
from libs.graph import build_connector_graph_enu


def load_config(path="config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_lane_network_enu(geomFiles, topoFiles, OriginLat=None, OriginLon=None,
                           OriginAltCm=0.0, Undirected=True, attrFiles=None):
    geomFiles = [str(p) for p in geomFiles]
    topoFiles = [str(p) for p in topoFiles]
    attrFiles = [str(p) for p in (attrFiles or [])]

    SG = init_empty_geometry()
    for p in geomFiles:
        SG = merge_geometry_structs(SG, parse_lane_geometry_tile(p, origin_alt_cm=OriginAltCm))

    ST = init_empty_topology()
    for p in topoFiles:
        ST = merge_topology_structs(ST, parse_lane_topology_tile(p))

    SA = ATTR_build(attrFiles)

    if (OriginLat is not None) and (OriginLon is not None):
        origin = {"lat0": float(OriginLat), "lon0": float(OriginLon), "h0": float(OriginAltCm) / 100.0}
    else:
        origin = resolve_origin(SG, SG.get("origin_lat"), SG.get("origin_lon"),
                                (float(SG.get("origin_alt_cm", 0)) / 100.0
                                 if SG.get("origin_alt_cm") is not None else 0.0))

    Sxy = dict(SG)
    Sxy["reference"]  = project_polyset_to_xy(SG.get("reference", []),  origin["lat0"], origin["lon0"], origin["h0"])
    Sxy["lanes"]      = project_cellpoly_to_xy(SG.get("lanes", []),      origin["lat0"], origin["lon0"], origin["h0"])
    Sxy["boundaries"] = project_cellpoly_to_xy(SG.get("boundaries", []), origin["lat0"], origin["lon0"], origin["h0"])

    _attach_attrs_to_Sxy(Sxy, SA)

    Graph = build_connector_graph_enu(Sxy, ST, SA, Undirected=Undirected)
    Graph["origin"] = origin
    return Sxy, ST, Graph, SA, origin


def _attach_attrs_to_Sxy(Sxy: dict, SA: dict) -> None:
    by_group = (SA or {}).get("by_group", {})
    Sxy["attrs"] = by_group
    lane_idx = {}
    for gid, G in by_group.items():
        lanes = G.get("lanes", {})
        for ln, L in lanes.items():
            lane_idx[(str(gid), int(ln))] = L
    Sxy["lane_attrs_index"] = lane_idx

    attached = 0
    for lane in (Sxy.get("lanes") or []):
        gid = str(lane.get("lane_group_ref") or lane.get("group_ref") or "")
        ln = lane.get("lane_number", lane.get("number", lane.get("laneNo", None)))
        try:
            ln = int(ln) if ln is not None else None
        except Exception:
            ln = None
        if gid and (ln is not None):
            L = lane_idx.get((gid, ln))
            if L is not None:
                lane["attrs"] = {
                    "directions": set(L.get("directions", [])),
                    "ranges": list(L.get("ranges", [])),
                    "types": set(L.get("types", [])),
                    "width_profiles": list(L.get("width_profiles", [])),
                    "flags": dict(L.get("flags", {})),
                }
                attached += 1

    b_attached = 0
    for b in (Sxy.get("boundaries") or []):
        gid = str(b.get("lane_group_ref") or b.get("group_ref") or "")
        bn = b.get("lane_boundary_number", b.get("number", None))
        try:
            bn = int(bn) if bn is not None else None
        except Exception:
            bn = None
        if gid and (bn is not None):
            G = by_group.get(gid, {})
            B = (G.get("boundaries") or {}).get(bn)
            if B is not None:
                b["attrs"] = {"markings": list(B.get("markings", []))}
                b_attached += 1

    print(f"[ATTR][attach] lanes: {attached}/{len(Sxy.get('lanes') or [])} with attrs, "
          f"boundaries: {b_attached}/{len(Sxy.get('boundaries') or [])} with attrs")
