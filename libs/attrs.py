# attrs.py (optimized)
from __future__ import annotations
import os, random
from typing import Dict, Any, List, Tuple, Set, Optional
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

def ATTR_build(attr_files: List[str], workers: Optional[int] = None) -> Dict[str, Any]:
    """
    Parallel tile parsing, single-threaded merge (to keep SA consistent).
    """
    SA = {"here_tile_ids": set(), "by_group": {}}
    total_files = 0
    total_groups = 0

    files = list(attr_files or [])
    if not files:
        SA["here_tile_ids"] = []
        return SA

    # I/O + parsing benefit from threads (we keep merge on main thread)
    max_workers = workers or min(32, (os.cpu_count() or 2) * 2)

    def _parse_one(p: str):
        raw = _ATTR_load_raw(p)
        n = _ATTR_count_groups_in_raw(raw)
        return p, raw, n

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_parse_one, p) for p in files]
        for fut in as_completed(futs):
            try:
                p, raw, n = fut.result()
            except Exception as ex:
                print(f"[ATTR] skip {files[futs.index(fut)] if fut in futs else '?'}: {ex}")
                continue
            _ATTR_merge_raw_into_SA(SA, raw)     # merge on main thread: thread-safe
            total_files += 1
            total_groups += n

    SA["here_tile_ids"] = sorted(list(SA["here_tile_ids"]))
    print(f"[ATTR] merged files={total_files}, groups_seen={total_groups}, SA_groups={len(SA['by_group'])}")
    return SA

def _ATTR_load_raw(p: str) -> dict:
    if not os.path.exists(p):
        raise FileNotFoundError(f"[ATTR] not found: {p}")
    raw = _load_json_file(p)
    if not isinstance(raw, dict):
        raise ValueError(f"[ATTR] {p} is not a JSON object")
    return raw

def _ATTR_count_groups_in_raw(raw: dict) -> int:
    groups = raw.get("lane_group_attribution") or []
    return len(groups) if isinstance(groups, list) else 1

# -------- range helpers -------------------------------------------------

def _clip01(x: Optional[float]) -> Optional[float]:
    if x is None: return None
    try:
        v = float(x)
    except Exception:
        return None
    if v < 0.0: return 0.0
    if v > 1.0: return 1.0
    return v

def _read_loc_start(d: Optional[dict]) -> Optional[float]:
    if not isinstance(d, dict): return None
    return _clip01(d.get("location_offset_from_start"))

def _applies_to_range(r: dict) -> Tuple[float, float]:
    if not isinstance(r, dict) or not r:
        return 0.0, 1.0

    # local variables avoid repeated dict lookups
    a = _read_loc_start(r.get("start"))
    b = _read_loc_start(r.get("end"))
    ros = _clip01(r.get("range_offset_from_start"))
    roe = _clip01(r.get("range_offset_from_end"))

    if a is not None or b is not None:
        a = 0.0 if a is None else a
        b = 1.0 if b is None else b
    elif ros is not None:
        a, b = ros, 1.0
    elif roe is not None:
        a, b = 0.0, (1.0 - roe)
        if b <= 0.0: a, b = 0.0, 1.0
    else:
        a, b = 0.0, 1.0

    if a < 0.0: a = 0.0
    elif a > 1.0: a = 1.0
    if b < 0.0: b = 0.0
    elif b > 1.0: b = 1.0
    if b <= a:
        a, b = 0.0, 1.0
    return float(a), float(b)

def _merge_ranges(ranges: List[dict]) -> List[dict]:
    if not ranges: return []
    rs = sorted(ranges, key=lambda z: (z.get("dir",""), z["s0"], z["s1"]))
    out = [dict(rs[0])]
    append = out.append
    for z in rs[1:]:
        last = out[-1]
        if z.get("dir","") == last.get("dir","") and z["s0"] <= last["s1"] + 1e-9:
            last["s1"] = max(last["s1"], z["s1"])
        else:
            append(dict(z))

    # clip & filter
    zz = []
    for r in out:
        a = r["s0"]; b = r["s1"]
        if a < 0.0: a = 0.0
        elif a > 1.0: a = 1.0
        if b < 0.0: b = 0.0
        elif b > 1.0: b = 1.0
        if b > a + 1e-12:
            item = {"s0": a, "s1": b}
            dr = r.get("dir")
            if dr: item["dir"] = dr
            zz.append(item)
    return zz

# -------- core merge ----------------------------------------------------

def _ATTR_merge_raw_into_SA(SA: dict, raw: dict) -> None:
    SA["here_tile_ids"].add(raw.get("here_tile_id"))

    groups = raw.get("lane_group_attribution") or []
    if not isinstance(groups, list):
        groups = [groups]
    by_group = SA["by_group"]
    setdefault = dict.setdefault

    for grp in groups:
        gid = str(grp.get("lane_group_ref", "")).strip()
        if not gid:
            continue

        G = setdefault(by_group, gid, {
            "lanes": {},
            "boundaries": {},
            "group_params": [],
            "summary": {}
        })
        G_lanes = G["lanes"]
        G_bnds  = G["boundaries"]
        G_params = G["group_params"]

        # ---- LANE ATTRIBUTION ----
        for L in (grp.get("lane_attribution") or []):
            ln_raw = L.get("lane_number")
            try:
                ln = int(ln_raw)
            except Exception:
                continue

            lane = setdefault(G_lanes, ln, {
                "directions": set(),
                "ranges": [],
                "types": set(),
                "width_profiles": [],
                "flags": {}
            })
            lane_dirs: Set[str] = lane["directions"]
            lane_ranges: List[dict] = lane["ranges"]
            lane_types: Set[str] = lane["types"]
            lane_wps: List[dict] = lane["width_profiles"]
            lane_flags: dict = lane["flags"]

            for pa in (L.get("parametric_attribution") or []):
                s0, s1 = _applies_to_range(pa.get("applies_to_range") or {})
                lpa_raw = pa.get("lane_parametric_attribution") or []
                lpa_items = [lpa_raw] if isinstance(lpa_raw, dict) else lpa_raw
                for item in lpa_items:
                    # direction_of_travel — string ("FORWARD"/"BACKWARD") or int enum
                    # int encoding: 2=FORWARD, 3=BACKWARD, 4=BOTH
                    _DOT_INT = {2: ("FORWARD",), 3: ("BACKWARD",), 4: ("FORWARD", "BACKWARD")}
                    d = item.get("direction_of_travel")
                    if d:
                        if isinstance(d, int):
                            dot_vals = _DOT_INT.get(d, ())
                        else:
                            dd = str(d).upper()
                            dot_vals = (dd,) if dd in ("FORWARD", "BACKWARD") else ()
                        for dd in dot_vals:
                            lane_dirs.add(dd)
                            lane_ranges.append({"dir": dd, "s0": s0, "s1": s1})

                    # lane_type
                    lt = item.get("lane_type")
                    if lt:
                        lane_types.add(str(lt).upper())

                    # width profile (as-is)
                    lwp = item.get("lane_width_profile")
                    if lwp:
                        lane_wps.append(lwp)

                    # flags
                    if "lane_within_intersection" in item:
                        _flag_add_range(lane_flags, "within_intersection", s0, s1)

            lane["ranges"] = _merge_ranges(lane_ranges)

        # ---- LANE BOUNDARY ATTRIBUTION ----
        for B in (grp.get("lane_boundary_attribution") or []):
            bnum_raw = B.get("lane_boundary_number")
            try:
                bnum = int(bnum_raw)
            except Exception:
                continue

            Bdst = setdefault(G_bnds, bnum, {"markings": []})
            markings = Bdst["markings"]

            pa_list = (B.get("parametric_attribution")
                       or B.get("lane_boundary_point_attribution") or [])
            for pa in pa_list:
                s0, s1 = _applies_to_range(pa.get("applies_to_range") or {})
                lbpa_raw = pa.get("lane_boundary_parametric_attribution") or []
                lbpa_items = [lbpa_raw] if isinstance(lbpa_raw, dict) else lbpa_raw
                for item in lbpa_items:
                    lbm = item.get("lane_boundary_marking")
                    if lbm is not None:
                        markings.append({"s0": s0, "s1": s1, "payload": lbm})

        # ---- GROUP PARAMETRIC ----
        for pa in (grp.get("parametric_attribution") or []):
            s0, s1 = _applies_to_range(pa.get("applies_to_range") or {})
            for item in (pa.get("lane_group_parametric_attribution") or []):
                G_params.append({"s0": s0, "s1": s1, "payload": item})

        # ---- SUMMARY ----
        dirs_all = set()
        for lane in G_lanes.values():
            dirs_all |= lane["directions"]
        G["summary"]["AllowedDir"] = (
            "BOTH" if {"FORWARD","BACKWARD"} <= dirs_all else
            "FWD"  if "FORWARD"  in dirs_all else
            "BWD"  if "BACKWARD" in dirs_all else
            "UNKNOWN"
        )

def _flag_add_range(flags: dict, name: str, s0: float, s1: float) -> None:
    flags.setdefault(name, []).append({"s0": float(s0), "s1": float(s1)})

# -------- utility API ---------------------------------------------------

def dir_from_SA_lane(SA: dict, gid: str, lane_number: int) -> str:
    g = (SA.get("by_group") or {}).get(str(gid), {})
    L = (g.get("lanes") or {}).get(int(lane_number), {})
    dirs = set(str(d).upper() for d in (L.get("directions") or []))
    if "FORWARD" in dirs and "BACKWARD" in dirs: return "BOTH"
    if "FORWARD" in dirs:  return "FWD"
    if "BACKWARD" in dirs: return "BWD"
    return "NONE"

# -------- tiny debug helper --------------------------------------------

def ATTR_debug_print(SA: dict, n_groups: int = 3, seed: Optional[int] = 0) -> None:
    by_group: dict = SA.get("by_group", {})
    gids = list(by_group.keys())
    if not gids:
        print("[ATTR][debug] no groups")
        return
    rnd = random.Random(seed)
    pick = gids if len(gids) <= n_groups else rnd.sample(gids, n_groups)

    print(f"[ATTR][debug] showing {len(pick)}/{len(gids)} groups; here_tile_ids={len(SA.get('here_tile_ids', []))}")
    for gi, gid in enumerate(pick, 1):
        G = by_group[gid]
        print(f"\n[{gi}] group {gid} | AllowedDir={G.get('summary',{}).get('AllowedDir')}")
        lanes = G.get("lanes", {})
        print(f"  lanes: {len(lanes)}")
        for ln in sorted(lanes.keys())[:5]:
            L = lanes[ln]
            dirs = ",".join(sorted(list(L.get("directions", [])))) or "-"
            types = ",".join(sorted(list(L.get("types", [])))) or "-"
            nrng = len(L.get("ranges", []))
            nwp  = len(L.get("width_profiles", []))
            flags = list(L.get("flags", {}).keys())
            print(f"    L{ln}: dirs=[{dirs}] types=[{types}] ranges={nrng} width_profiles={nwp} flags={flags}")
        if len(lanes) > 5:
            print(f"    ... (+{len(lanes)-5} more lanes)")

        bnd = G.get("boundaries", {})
        print(f"  boundaries: {len(bnd)}")
        for bn in sorted(bnd.keys())[:5]:
            M = bnd[bn]
            print(f"    B{bn}: markings={len(M.get('markings', []))}")
        if len(bnd) > 5:
            print(f"    ... (+{len(bnd)-5} more boundaries)")

        gps = G.get("group_params", [])
        print(f"  group_params: {len(gps)}")
