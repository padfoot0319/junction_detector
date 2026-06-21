# topology.py (optimized + batch helper)
from __future__ import annotations
import os
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----- fast JSON loader (orjson if available) -----
try:
    import orjson as _json
    def _load_json_file(p: str) -> dict:
        with open(p, "rb") as f:
            return _json.loads(f.read())
except Exception:
    import json as _json
    def _load_json_file(p: str) -> dict:
        with open(p, "r", encoding="utf-8") as f:
            return _json.load(f)

def init_empty_topology() -> Dict[str, Any]:
    return {"here_tile_id": "", "groups": [], "groupIndex": {}}

def parse_lane_topology_tile(jsonFile: str) -> Dict[str, Any]:
    raw = _load_json_file(jsonFile)
    T = init_empty_topology()
    T["here_tile_id"] = raw.get("here_tile_id", "")

    LGS = raw.get("lane_groups_starting_in_tile")
    if not LGS:
        return T
    if not isinstance(LGS, list):
        LGS = [LGS]

    groups_out = T["groups"]
    gindex = T["groupIndex"]
    _canon = _canon_id
    _as = _as_id

    for i, g in enumerate(LGS, start=1):
        gid = _as(g.get("lane_group_id"), f"UNKNOWN_{i}")

        endRef = g.get("end_lane_group_connector_ref")
        if isinstance(endRef, dict):
            endConnId  = _canon(endRef.get("lane_group_connector_id"))
            endConnTid = endRef.get("lane_group_connector_here_tile_id")
        else:
            endConnId  = _canon(endRef)
            endConnTid = None

        lanesOut = []
        lanes = g.get("lanes")
        if lanes and isinstance(lanes, list):
            # enumerate provides lane_index_within_group directly
            lanesOut = [{
                "start_lane_connector_number": ln.get("start_lane_connector_number"),
                "end_lane_connector_number":   ln.get("end_lane_connector_number"),
                "lane_index_within_group":     k
            } for k, ln in enumerate(lanes, start=1)]

        lg_len = g.get("lane_group_length_meters")
        item = {
            "lane_group_id": str(gid),
            "start_lane_group_connector_id": str(_canon(g.get("start_lane_group_connector_ref"))),
            "end_lane_group_connector_id":   str(endConnId),
            "end_lane_group_connector_tile": endConnTid,
            "lane_group_length_meters": float(lg_len) if lg_len is not None else None,
            "lanes": lanesOut
        }

        groups_out.append(item)
        gindex[str(gid)] = len(groups_out)
    return T

def merge_topology_structs(ST: Dict[str, Any], t: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(t, dict) or not t or not t.get("groups"):
        return ST
    base = len(ST["groups"])
    ST["groups"].extend(t["groups"])
    gindex = ST["groupIndex"]
    # Build index in one pass
    for i, g in enumerate(t["groups"], start=1):
        gid = str(g.get("lane_group_id", f"UNKNOWN_{i}"))
        gindex[gid] = base + i
    return ST

# ---- batch helper: parse many tiles in parallel & merge ----
def parse_and_merge_topology_tiles(files: List[str], workers: Optional[int] = None) -> Dict[str, Any]:
    ST = init_empty_topology()
    fs = list(files or [])
    if not fs:
        return ST

    max_workers = workers or min(32, (os.cpu_count() or 2) * 2)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(parse_lane_topology_tile, p): p for p in fs}
        for fut in as_completed(futs):
            try:
                t = fut.result()
            except Exception as ex:
                print(f"[TOPO] skip {futs[fut]}: {ex}")
                continue
            merge_topology_structs(ST, t)
    return ST

# ---- small helpers ----
def _canon_id(x) -> str:
    if x is None: return ""
    s = str(x).strip()
    return s

def _as_id(val, defaultVal: str) -> str:
    if val is None: return defaultVal
    s = _canon_id(val)
    return s if s else defaultVal
