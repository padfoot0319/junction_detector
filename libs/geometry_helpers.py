"""
Geometry helpers for polylines and lane geometries in ENU coordinates.

This module provides:
- Arc-length based utilities (cumulated length, sub-polyline extraction),
- Lane clipping and orientation helpers,
- Simple direction and arrow-placement helpers for plotting.
"""

import json, math
from typing import Any, Dict, List, Tuple
import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points


def _cumlen(poly: List[Tuple[float, float]]) -> List[float]:
    """Cumulative arc-length along a polyline."""
    L = [0.0]
    for i in range(1, len(poly)):
        dx = poly[i][0] - poly[i - 1][0]
        dy = poly[i][1] - poly[i - 1][1]
        L.append(L[-1] + math.hypot(dx, dy))
    return L


def _interp(
    p: Tuple[float, float],
    q: Tuple[float, float],
    t: float,
) -> Tuple[float, float]:
    """Linear interpolation between p and q (0 <= t <= 1)."""
    return (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))


def _subpoly_by_fraction(
    poly: List[Tuple[float, float]],
    s: float,
    e: float,
) -> List[Tuple[float, float]]:
    """
    Return sub-polyline between fractions s and e (0 <= s <= e <= 1)
    measured along arc length.
    """
    if not poly or s >= e:
        return []
    if s <= 0 and e >= 1:
        return list(poly)

    L = _cumlen(poly)
    total = L[-1] if L else 0.0
    if total == 0.0:
        return [poly[0]] if poly else []

    import bisect

    s_abs, e_abs = s * total, e * total
    out: List[Tuple[float, float]] = []

    # place s
    i = bisect.bisect_left(L, s_abs)
    if i == 0:
        out.append(poly[0])
    else:
        p0, p1 = poly[i - 1], poly[i]
        seg = L[i] - L[i - 1]
        t = 0.0 if seg == 0 else (s_abs - L[i - 1]) / seg
        out.append(_interp(p0, p1, t))

    # walk until e
    j = i
    while j < len(poly) and L[j] < e_abs:
        if j > 0:
            out.append(poly[j])
        j += 1

    if j == 0:
        out.append(poly[0])
    else:
        p0, p1 = poly[j - 1], poly[min(j, len(poly) - 1)]
        seg = (L[j] - L[j - 1]) if j < len(L) else 0.0
        base = L[j - 1] if j - 1 < len(L) else L[-1]
        t = 0.0 if seg == 0 else (e_abs - base) / seg
        out.append(_interp(p0, p1, t))

    # dedupe consecutive equal points
    dedup = [out[0]]
    for k in range(1, len(out)):
        if out[k] != dedup[-1]:
            dedup.append(out[k])
    return dedup


def _reverse_lane_in_place(lane: Dict[str, Any]) -> None:
    """Reverse a lane (centerline + boundaries) in-place."""
    lane["centerline"]["x"].reverse()
    lane["centerline"]["y"].reverse()
    lane["left_boundary"]["x"], lane["right_boundary"]["x"] = (
        lane["right_boundary"]["x"],
        lane["left_boundary"]["x"],
    )
    lane["left_boundary"]["y"], lane["right_boundary"]["y"] = (
        lane["right_boundary"]["y"],
        lane["left_boundary"]["y"],
    )


def _clip_lane_by_fraction(lane: Dict[str, Any], s: float, e: float) -> None:
    """Clip lane centerline and boundaries between fractions s and e."""
    cx = list(zip(lane["centerline"]["x"], lane["centerline"]["y"]))
    lx = list(zip(lane["left_boundary"]["x"], lane["left_boundary"]["y"]))
    rx = list(zip(lane["right_boundary"]["x"], lane["right_boundary"]["y"]))

    c_sub = _subpoly_by_fraction(cx, s, e)
    l_sub = _subpoly_by_fraction(lx, s, e)
    r_sub = _subpoly_by_fraction(rx, s, e)
    lane["centerline"]["x"] = [p[0] for p in c_sub]
    lane["centerline"]["y"] = [p[1] for p in c_sub]
    lane["left_boundary"]["x"] = [p[0] for p in l_sub]
    lane["left_boundary"]["y"] = [p[1] for p in l_sub]
    lane["right_boundary"]["x"] = [p[0] for p in r_sub]
    lane["right_boundary"]["y"] = [p[1] for p in r_sub]


def _point_first_last(
    centerline: Dict[str, List[float]]
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Return first and last point of a lane centerline."""
    xs, ys = centerline["x"], centerline["y"]
    return (xs[0], ys[0]), (xs[-1], ys[-1])


def _point_dir_at_fraction(
    poly: List[Tuple[float, float]],
    frac: float = 0.55,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Return (point, unit direction) near a given arc-length fraction
    along the polyline.
    """
    if not poly:
        return (0.0, 0.0), (1.0, 0.0)
    if len(poly) == 1:
        return poly[0], (1.0, 0.0)

    L = _cumlen(poly)
    total = L[-1] if L else 0.0
    if total <= 0:
        return poly[0], (1.0, 0.0)

    frac = max(0.0, min(1.0, frac))
    s_abs = frac * total

    import bisect

    i = min(len(L) - 1, max(1, bisect.bisect_left(L, s_abs)))
    seg = L[i] - L[i - 1]
    t = 0.0 if seg == 0 else (s_abs - L[i - 1]) / seg
    P = _interp(poly[i - 1], poly[i], t)

    dx = poly[i][0] - poly[i - 1][0]
    dy = poly[i][1] - poly[i - 1][1]
    n = math.hypot(dx, dy)
    if n == 0:
        return P, (1.0, 0.0)
    return P, (dx / n, dy / n)

def _wrap_deg(a: float) -> float:
    a = a % 360.0
    return a + 360.0 if a < 0 else a

def _dirless_angle_diff_deg(a_deg: float, b_deg: float) -> float:
    d = abs((a_deg - b_deg + 180.0) % 360.0 - 180.0)
    return d

def _poly_heading_deg(ls: LineString) -> float:
    (x0,y0),(x1,y1) = ls.coords[0], ls.coords[-1]
    return (math.degrees(math.atan2(y1-y0, x1-x0)) + 360.0) % 360.0

def _heading_of_segment(p0, p1) -> float:
    return _wrap_deg(math.degrees(math.atan2(p1[1]-p0[1], p1[0]-p0[0])))

def _endpoints(ls: LineString) -> Tuple[Tuple[float,float], Tuple[float,float]]:
    c = ls.coords
    return (c[0][0], c[0][1]), (c[-1][0], c[-1][1])

def _flip(ls: LineString) -> LineString:
    return LineString(list(ls.coords)[::-1])

def _d2(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    dx = a[0] - b[0]; dy = a[1] - b[1]
    return dx*dx + dy*dy


def _dist2(a, b):
    return (a[0]-b[0])**2 + (a[1]-b[1])**2

def _reverse_linestring(ls):
    return LineString(list(ls.coords)[::-1])

def _project_on_ls(ls, pt):
    # returns (arclen, in_range_bool)
    s = float(ls.project(Point(pt)))
    return s, (0.0 <= s <= float(ls.length)+1e-9)

def _signed_area_xy(coords: List[Tuple[float,float]]) -> float:
    """Shoelace signed area (positive: CCW)."""
    if len(coords) < 3: 
        return 0.0
    x = np.asarray([c[0] for c in coords], dtype=float)
    y = np.asarray([c[1] for c in coords], dtype=float)
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

from shapely.geometry import Point
import numpy as np

def _sample_subsegment(line, s0, s1, n_samples=8):
    """Sample n points along a LineString between curvilinear params s0..s1."""
    if s0 > s1:
        s0, s1 = s1, s0
    if n_samples <= 2:
        return [
            tuple(line.interpolate(s0).coords[0]),
            tuple(line.interpolate(s1).coords[0]),
        ]
    ts = np.linspace(s0, s1, n_samples)
    return [tuple(line.interpolate(t).coords[0]) for t in ts]

def _strip_is_ccw(pi, pj, P0, P1, P2, P3, n_samples=8):
    """
    pi, pj: shapely LineStrings
    P0,P1 on pi; P2,P3 on pj
    Returns True if the curved strip P0→pi→P1→pj→P3→P0 is CCW, False if CW.
    """

    # curvilinear positions of the four anchor points
    t0 = float(pi.project(Point(P0)))
    t1 = float(pi.project(Point(P1)))
    u2 = float(pj.project(Point(P2)))
    u3 = float(pj.project(Point(P3)))

    # sample along the two subsegments
    pi_seg = _sample_subsegment(pi, t0, t1, n_samples)   # P0..P1
    pj_seg = _sample_subsegment(pj, u2, u3, n_samples)   # P2..P3

    ring = pi_seg + pj_seg
    if ring[0] != ring[-1]:
        ring.append(ring[0])

    A = _signed_area_xy(ring)   # your shoelace function
    return A > 0.0


def _segments_cross(seg1, seg2, strict=True, eps=1e-9):
    (ax,ay),(bx,by)=(seg1[0],seg1[1])
    (cx,cy),(dx,dy)=(seg2[0],seg2[1])
    def area2(x1,y1,x2,y2,x3,y3): return (x2-x1)*(y3-y1)-(y2-y1)*(x3-x1)
    a1=area2(ax,ay,bx,by,cx,cy); a2=area2(ax,ay,bx,by,dx,dy)
    a3=area2(cx,cy,dx,dy,ax,ay); a4=area2(cx,cy,dx,dy,bx,by)
    if (a1>eps and a2<-eps or a1<-eps and a2>eps) and (a3>eps and a4<-eps or a3<-eps and a4>eps):
        return True
    if strict: return False
    def onseg(x1,y1,x2,y2,px,py):
        return (min(x1,x2)-eps<=px<=max(x1,x2)+eps and
                min(y1,y2)-eps<=py<=max(y1,y2)+eps and
                abs(area2(x1,y1,x2,y2,px,py))<=eps)
    return (abs(a1)<=eps and onseg(ax,ay,bx,by,cx,cy)) or \
           (abs(a2)<=eps and onseg(ax,ay,bx,by,dx,dy)) or \
           (abs(a3)<=eps and onseg(cx,cy,dx,dy,ax,ay)) or \
           (abs(a4)<=eps and onseg(cx,cy,dx,dy,bx,by))

def _heading_at_end_of_linestring(ls: LineString, end: str) -> float:
    cs = list(ls.coords)
    if len(cs) < 2: 
        return 0.0
    (x0,y0),(x1,y1) = (cs[0], cs[1]) if end == 'start' else (cs[-2], cs[-1])
    return math.atan2(y1-y0, x1-x0)

def _arrow_along(ax, line: LineString, every=150.0, head=12, lw=1.0, color='k', alpha=1.0, z=6):
    L = line.length
    if L <= 1e-6: 
        return
    s = every
    while s < L:
        p0 = line.interpolate(max(0.0, s-0.5*every))
        p1 = line.interpolate(min(L, s+0.5*every))
        ax.annotate("", xy=(p1.x, p1.y), xytext=(p0.x, p0.y),
                    arrowprops=dict(arrowstyle="->", lw=lw, color=color, shrinkA=0, shrinkB=0, alpha=alpha),
                    zorder=z)
        s += every
        
def _closest_end_to_point(g: LineString, pt_xy: Tuple[float,float]) -> str:
    a = Point(g.coords[0]); b = Point(g.coords[-1]); P = Point(pt_xy)
    return 'start' if a.distance(P) <= b.distance(P) else 'end'

def _make_finite_ray_from_end(g: LineString, which_end: str, L: float = 250.0) -> LineString:
    cs = list(g.coords)
    if which_end == 'start':
        p0, p1 = cs[0], cs[1]
        # outward from the interior: opposite of the local tangent (flip)
        vx, vy = p0[0]-p1[0], p0[1]-p1[1]
    else:
        p0, p1 = cs[-1], cs[-2]
        vx, vy = p0[0]-p1[0], p0[1]-p1[1]
    n = math.hypot(vx, vy) or 1.0
    vx, vy = vx/n, vy/n
    q = (p0[0] + L*vx, p0[1] + L*vy)
    return LineString([p0, q])

def _ray_hits_segment(ray: LineString, seg: LineString):
    inter = ray.intersection(seg)
    if inter.is_empty:
        return False, None
    if inter.geom_type == 'Point':
        return True, (inter.x, inter.y)
    # tiny overlap → use midpoint
    g = inter if inter.geom_type == 'LineString' else list(inter.geoms)[0]
    m = g.interpolate(0.5*g.length)
    return True, (m.x, m.y)

def _orient_to_target(poly: LineString, target_deg: float) -> LineString:
    """Flip poly if its end-to-end direction disagrees (>90°) with target heading."""
    if not isinstance(poly, LineString) or len(poly.coords) < 2:
        return poly
    cur = _poly_heading_deg(poly)
    if _dirless_angle_diff_deg(cur, target_deg) > 90.0:
        return LineString(list(poly.coords)[::-1])
    return poly



def intersect_T_pair_unified(A: LineString, B: LineString, *,
                             L_extrap=400.0, max_ray_sep=200.0, tol=1e-9) -> Dict[str,Any]:
    """
    Unified T hit logic: 'main' (segment/segment), 'ray-ray', 'A-ray-B', 'B-ray-A'
    Returns keys:
      ok, center, mode, tA, tB, endA, endB, dmin
    """
    pa, pb = nearest_points(A, B)
    dmin = pa.distance(pb)

    # (a) direct segment hit
    inter = A.intersection(B)
    if not inter.is_empty:
        if inter.geom_type == 'Point':
            X = (inter.x, inter.y)
        else:
            g = inter if inter.geom_type == 'LineString' else list(inter.geoms)[0]
            m = g.interpolate(0.5*g.length)
            X = (m.x, m.y)
        return dict(ok=True, center=X, mode='main', tA=0.0, tB=0.0, endA=None, endB=None, dmin=dmin)

    # build finite rays (outward from the end facing the other)
    endA = _closest_end_to_point(A, (pb.x, pb.y))
    endB = _closest_end_to_point(B, (pa.x, pa.y))
    rayA = _make_finite_ray_from_end(A, endA, L=L_extrap)
    rayB = _make_finite_ray_from_end(B, endB, L=L_extrap)

    # (b) ray-ray (cheap gate by proximity)
    if dmin <= max_ray_sep:
        # solve infinite rays analytically; then bound by [0,L_extrap] via segment hit
        ok, X = _ray_hits_segment(rayA, rayB)
        if ok:
            return dict(ok=True, center=X, mode='ray-ray', tA=None, tB=None, endA=endA, endB=endB, dmin=dmin)

    # (c1) A-ray with B-segment
    ok, X = _ray_hits_segment(rayA, B)
    if ok:
        return dict(ok=True, center=X, mode='A-ray-B', tA=None, tB=0.0, endA=endA, endB=endB, dmin=dmin)

    # (c2) B-ray with A-segment
    ok, X = _ray_hits_segment(rayB, A)
    if ok:
        return dict(ok=True, center=X, mode='B-ray-A', tA=0.0, tB=None, endA=endA, endB=endB, dmin=dmin)

    return dict(ok=False, center=None, mode=None, tA=None, tB=None, endA=endA, endB=endB, dmin=dmin)


# ================== Tile parsing, Morton decoding, WGS-84 / ENU projection ==================

def init_empty_geometry() -> Dict[str, Any]:
    return {
        "here_tile_id": "",
        "origin_morton": np.uint64(0),
        "origin_lat": np.nan,
        "origin_lon": np.nan,
        "origin_alt_cm": np.int64(0),
        "origins": [],
        "reference": [],
        "lanes": [],
        "boundaries": []
    }

def _parse_lane_geometry_tile_predecoded(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Handle geometry files that are already decoded (lat/lon/alt_cm arrays present directly)."""
    S = {
        "here_tile_id": raw.get("here_tile_id", ""),
        "origin_morton": np.uint64(raw.get("origin_morton", 0)),
        "origin_lat": float(raw.get("origin_lat", np.nan)),
        "origin_lon": float(raw.get("origin_lon", np.nan)),
        "origin_alt_cm": int(raw.get("origin_alt_cm", 0)),
        "reference": [],
        "lanes": [],
        "boundaries": []
    }
    LG = raw.get("lane_group_geometries") or []
    for G in LG:
        ref_list = G.get("reference_geometry") or []
        for item in ref_list:
            S["reference"].append({
                "lane_group_ref": item.get("lane_group_ref"),
                "lat": np.asarray(item["lat"], float),
                "lon": np.asarray(item["lon"], float),
                "alt_cm": np.asarray(item["alt_cm"], np.int64),
            })
        for item in (G.get("lane_geometries") or []):
            S["lanes"].append({
                "lane_group_ref": item.get("lane_group_ref"),
                "lat": np.asarray(item["lat"], float),
                "lon": np.asarray(item["lon"], float),
                "alt_cm": np.asarray(item["alt_cm"], np.int64),
                "left_lane_boundary_number": item.get("left_lane_boundary_number"),
                "right_lane_boundary_number": item.get("right_lane_boundary_number"),
                "lane_index_within_group": item.get("lane_index_within_group"),
                "lane_number": item.get("lane_number"),
            })
        for item in (G.get("lane_boundary_geometries") or []):
            S["boundaries"].append({
                "lane_group_ref": item.get("lane_group_ref"),
                "lat": np.asarray(item["lat"], float),
                "lon": np.asarray(item["lon"], float),
                "alt_cm": np.asarray(item["alt_cm"], np.int64),
                "boundary_index_within_group": item.get("boundary_index_within_group"),
            })
    return S


def parse_lane_geometry_tile(jsonFile: str, origin_alt_cm: float = 0.0) -> Dict[str, Any]:
    with open(jsonFile, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if "tile_center_here_2d_coordinate" not in raw:
        if "origin_lat" in raw:
            return _parse_lane_geometry_tile_predecoded(raw)
        raise ValueError("tile_center_here_2d_coordinate missing in JSON.")

    origin_morton = parse_uint64_decimal(raw["tile_center_here_2d_coordinate"])

    t3d = raw.get("tile_center_here_3d_coordinate")
    if t3d and "cm_from_wgs84_ellipsoid" in t3d:
        origin_alt_cm = int(t3d["cm_from_wgs84_ellipsoid"])
    else:
        origin_alt_cm = int(origin_alt_cm)

    origin_lat, origin_lon = morton_to_latlon(origin_morton)

    S = {
        "here_tile_id": raw.get("here_tile_id", ""),
        "origin_morton": origin_morton,
        "origin_lat": origin_lat,
        "origin_lon": origin_lon,
        "origin_alt_cm": origin_alt_cm,
        "reference": [],
        "lanes": [],
        "boundaries": []
    }

    LG = raw.get("lane_group_geometries")
    if not LG:
        return S
    if not isinstance(LG, list):
        LG = list(LG)

    refOut: List[Dict[str, Any]] = []

    for G in LG:
        gid_ref = G.get("lane_group_ref")

        ref = G.get("reference_geometry")
        if ref:
            lat, lon, alt = decode_stream(ref, origin_morton, origin_alt_cm)
            refOut.append({
                "lane_group_ref": gid_ref,
                "lat": lat, "lon": lon, "alt_cm": alt
            })

        lanes = G.get("lane_geometries")
        if lanes:
            if not isinstance(lanes, list):
                lanes = list(lanes)
            for k, Lk in enumerate(lanes, start=1):
                path = Lk.get("lane_path_geometry")
                if not path:
                    continue
                lat, lon, alt = decode_stream(path, origin_morton, origin_alt_cm)
                S["lanes"].append({
                    "lane_group_ref": gid_ref,
                    "lat": lat, "lon": lon, "alt_cm": alt,
                    "left_lane_boundary_number": Lk.get("left_lane_boundary_number"),
                    "right_lane_boundary_number": Lk.get("right_lane_boundary_number"),
                    "lane_index_within_group": k,
                    "lane_number": Lk.get("lane_number", k)
                })

        bgs = G.get("lane_boundary_geometries")
        if bgs:
            if not isinstance(bgs, list):
                bgs = list(bgs)
            for b, B in enumerate(bgs, start=1):
                geom = B.get("geometry")
                if not geom:
                    continue
                lat, lon, alt = decode_stream(geom, origin_morton, origin_alt_cm)
                # Use the HERE-assigned boundary number when present; fall back to
                # sequential position (b) for tiles that omit it.  The fallback is
                # safe only when HERE stores boundaries in slot-number order from 1,
                # which is the case for all tiles seen so far.
                bnum = B.get("lane_boundary_number")
                boundary_idx = int(bnum) if bnum is not None else b
                S["boundaries"].append({
                    "lane_group_ref": gid_ref,
                    "lat": lat, "lon": lon, "alt_cm": alt,
                    "boundary_index_within_group": boundary_idx
                })

    S["reference"] = refOut
    return S

def merge_geometry_structs(SG: Dict[str, Any], s: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(s, dict) or not s:
        return SG

    tile_id       = s.get("here_tile_id")
    origin_morton = s.get("origin_morton")
    origin_lat    = s.get("origin_lat")
    origin_lon    = s.get("origin_lon")
    origin_alt_cm = s.get("origin_alt_cm")

    SG.setdefault("origins", [])
    SG["origins"].append({
        "tile_id": tile_id,
        "origin_morton": origin_morton,
        "origin_lat": origin_lat,
        "origin_lon": origin_lon,
        "origin_alt_cm": origin_alt_cm
    })

    if s.get("reference"):  SG["reference"].extend(s["reference"])
    if s.get("lanes"):      SG["lanes"].extend(s["lanes"])
    if s.get("boundaries"): SG["boundaries"].extend(s["boundaries"])

    if tile_id is not None:       SG["here_tile_id"]  = tile_id
    if origin_morton is not None: SG["origin_morton"] = origin_morton
    if origin_lat is not None:    SG["origin_lat"]    = origin_lat
    if origin_lon is not None:    SG["origin_lon"]    = origin_lon
    if origin_alt_cm is not None: SG["origin_alt_cm"] = origin_alt_cm
    return SG

def resolve_origin(S: Dict[str, Any], lat0, lon0, h0) -> Dict[str, float]:
    if (lat0 is not None) and (lon0 is not None) and not _is_nan(lat0) and not _is_nan(lon0):
        return {"lat0": float(np.array(lat0).flatten()[0]),
                "lon0": float(np.array(lon0).flatten()[0]),
                "h0": float(np.array(h0).flatten()[0]) if h0 is not None else _default_alt(S)}
    if S.get("origin_lat") is not None and not _is_nan(S.get("origin_lat")):
        return {"lat0": float(S["origin_lat"]), "lon0": float(S["origin_lon"]), "h0": _default_alt(S)}
    if S.get("origins"):
        o0 = S["origins"][0]
        cm = o0.get("origin_alt_cm", 0) or 0
        return {"lat0": float(o0.get("origin_lat", 0.0)),
                "lon0": float(o0.get("origin_lon", 0.0)),
                "h0": float(cm)/100.0}
    return {"lat0": 0.0, "lon0": 0.0, "h0": 0.0}

def project_polyset_to_xy(ref_list: List[Dict[str, Any]], lat0, lon0, h0):
    out = []
    for R in ref_list:
        z = float(np.array(R.get("alt_cm", 0)).flatten()[0]) / 100.0 if R.get("alt_cm") is not None else 0.0
        x, y = latlon_to_xy(np.asarray(R["lat"]), np.asarray(R["lon"]), lat0, lon0, h0, z)
        item = dict(R); item["x"] = x; item["y"] = y
        out.append(item)
    return out

def project_cellpoly_to_xy(lanes_or_bounds: List[Dict[str, Any]], lat0, lon0, h0):
    out = []
    for C in lanes_or_bounds:
        z = float(np.array(C.get("alt_cm", 0)).flatten()[0]) / 100.0 if C.get("alt_cm") is not None else 0.0
        x, y = latlon_to_xy(np.asarray(C["lat"]), np.asarray(C["lon"]), lat0, lon0, h0, z)
        item = dict(C); item["x"] = x; item["y"] = y
        out.append(item)
    return out

def decode_stream(node: Dict[str, Any], origin_morton: np.uint64, origin_alt_cm: int):
    xyField = node.get("here_2d_coordinate_diffs")
    if xyField is None:
        return np.array([]), np.array([]), np.array([], dtype=np.int64)
    xyDiffs = strcell_to_uint64(xyField)
    zDiffsField = node.get("cm_from_wgs84_ellipsoid_diffs")
    zDiffs = np.asarray(zDiffsField, dtype=np.int64).reshape(-1) if zDiffsField is not None else np.zeros(len(xyDiffs), dtype=np.int64)

    N = len(xyDiffs)
    lats = np.zeros(N, float); lons = np.zeros(N, float); alts = np.zeros(N, dtype=np.int64)
    prev = np.uint64(origin_morton)
    for i in range(N):
        actual = np.uint64(prev ^ xyDiffs[i])
        prev = actual
        lat, lon = morton_to_latlon(actual)
        lats[i] = lat; lons[i] = lon; alts[i] = int(origin_alt_cm) + int(zDiffs[i])
    return lats, lons, alts

def strcell_to_uint64(x) -> np.ndarray:
    if isinstance(x, (list, tuple)):
        return np.array([parse_uint64_decimal(s) for s in x], dtype=np.uint64)
    if isinstance(x, str):
        return np.array([parse_uint64_decimal(x)], dtype=np.uint64)
    if isinstance(x, np.ndarray):
        return x.astype(np.uint64).reshape(-1)
    raise TypeError("Unsupported coord diffs type.")

def parse_uint64_decimal(s) -> np.uint64:
    s = str(s)
    u = np.uint64(0); ten = np.uint64(10)
    for ch in s:
        if '0' <= ch <= '9':
            u = u*ten + np.uint64(ord(ch) - ord('0'))
    return u

def morton_to_latlon(morton_code: np.uint64) -> Tuple[float, float]:
    mc = np.uint64(morton_code)
    lat_bits = _deinterleave_one_32(mc >> np.uint64(1))
    lon_bits = _deinterleave_one_32(mc)
    if (lat_bits & np.uint32(0x40000000)) != 0:
        lat_bits = np.uint32(lat_bits | np.uint32(0x80000000))
    lat_signed = np.int32(lat_bits.view(np.uint32))
    lon_signed = np.int32(lon_bits.view(np.uint32))
    scale = float(2**31)
    lat = float(lat_signed) * 180.0 / scale
    lon = float(lon_signed) * 180.0 / scale
    return lat, lon

def _deinterleave_one_32(interleaved: np.uint64) -> np.uint32:
    m = np.uint64(interleaved)
    m = m & np.uint64(0x5555555555555555)
    m = (m | (m >> 1))  & np.uint64(0x3333333333333333)
    m = (m | (m >> 2))  & np.uint64(0x0F0F0F0F0F0F0F0F)
    m = (m | (m >> 4))  & np.uint64(0x00FF00FF00FF00FF)
    m = (m | (m >> 8))  & np.uint64(0x0000FFFF0000FFFF)
    m = (m | (m >> 16)) & np.uint64(0x00000000FFFFFFFF)
    return np.uint32(m & np.uint64(0xFFFFFFFF))

def geodetic_to_ecef(lat, lon, h):
    a = 6378137.0; f = 1/298.257223563; e2 = f*(2 - f)
    lat = np.asarray(lat, float); lon = np.asarray(lon, float); h = np.asarray(h, float)
    phi = np.deg2rad(lat); lam = np.deg2rad(lon)
    sphi = np.sin(phi); cphi = np.cos(phi); slam = np.sin(lam); clam = np.cos(lam)
    N = a / np.sqrt(1 - e2*(sphi**2))
    X = (N + h)*cphi*clam
    Y = (N + h)*cphi*slam
    Z = (N*(1 - e2) + h)*sphi
    return X, Y, Z

def ecef_to_geodetic(X, Y, Z):
    a = 6378137.0; f = 1/298.257223563; e2 = f*(2 - f); b = a*(1 - f)
    X = np.asarray(X, float); Y = np.asarray(Y, float); Z = np.asarray(Z, float)
    r = np.hypot(X, Y); E2 = a*a - b*b; F = 54*(b*b)*(Z*Z)
    G = r*r + (1 - e2)*(Z*Z) - e2*E2
    c = (e2*e2*F*(r*r)) / (G*G*G)
    s = np.cbrt(1 + c + np.sqrt(c*c + 2*c))
    P = F / (3*(s + 1/s + 1)**2 * G*G)
    Q = np.sqrt(1 + 2*e2*e2*P)
    r0 = -(P*e2*r)/(1 + Q) + np.sqrt(0.5*a*a*(1 + 1/Q) - (P*(1 - e2)*(Z*Z))/(Q*(1 + Q)) - 0.5*P*r*r)
    U = np.sqrt((r - e2*r0)**2 + Z*Z)
    V = np.sqrt((r - e2*r0)**2 + (1 - e2)*Z*Z)
    z0 = (b*b*Z) / (a*V)
    h  = U*(1 - (b*b)/(a*V))
    lat = np.arctan2(Z + (e2*(b*b)/a)*z0, r)
    lon = np.arctan2(Y, X)
    return np.rad2deg(lat), np.rad2deg(lon), h

# WGS-84
_A = 6378137.0
_F = 1/298.257223563
_E2 = _F * (2 - _F)
_B  = _A * (1 - _F)
_EP2 = (_A*_A - _B*_B) / (_B*_B)

def geodetic_to_ecef_deg(lat_deg, lon_deg, h):
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    h   = np.asarray(h, dtype=float)
    s = np.sin(lat); c = np.cos(lat)
    N = _A / np.sqrt(1.0 - _E2 * s*s)
    X = (N + h) * c * np.cos(lon)
    Y = (N + h) * c * np.sin(lon)
    Z = ((1.0 - _E2) * N + h) * s
    return X, Y, Z

def ecef_to_geodetic_deg(X, Y, Z):
    X = np.asarray(X, float); Y = np.asarray(Y, float); Z = np.asarray(Z, float)
    p = np.hypot(X, Y)
    theta = np.arctan2(Z * _A, p * _B)
    s = np.sin(theta); c = np.cos(theta)
    lat = np.arctan2(Z + _EP2 * _B * s**3, p - _E2 * _A * c**3)
    lon = np.arctan2(Y, X)
    N = _A / np.sqrt(1.0 - _E2 * np.sin(lat)**2)
    h = p / np.cos(lat) - N
    return np.degrees(lat), np.degrees(lon), h

def geodetic_to_enu(lat_deg, lon_deg, h, lat0_deg, lon0_deg, h0):
    X,  Y,  Z  = geodetic_to_ecef_deg(lat_deg, lon_deg, h)
    X0, Y0, Z0 = geodetic_to_ecef_deg(lat0_deg, lon0_deg, h0)
    dX, dY, dZ = np.asarray(X)-X0, np.asarray(Y)-Y0, np.asarray(Z)-Z0
    phi = math.radians(float(lat0_deg)); lam = math.radians(float(lon0_deg))
    sphi, cphi = math.sin(phi), math.cos(phi); slam, clam = math.sin(lam), math.cos(lam)
    R = np.array([
        [-slam,          clam,          0.0],
        [-sphi*clam,    -sphi*slam,     cphi],
        [ cphi*clam,     cphi*slam,     sphi]
    ])
    enu = np.stack([dX, dY, dZ], axis=-1) @ R.T
    return enu[...,0], enu[...,1], enu[...,2]

def enu_to_geodetic(e, n, u, lat0_deg, lon0_deg, h0):
    e = np.asarray(e, float); n = np.asarray(n, float); u = np.asarray(u, float)
    X0, Y0, Z0 = geodetic_to_ecef_deg(lat0_deg, lon0_deg, h0)
    phi = math.radians(float(lat0_deg)); lam = math.radians(float(lon0_deg))
    sphi, cphi = math.sin(phi), math.cos(phi); slam, clam = math.sin(lam), math.cos(lam)
    R = np.array([
        [-slam,          clam,          0.0],
        [-sphi*clam,    -sphi*slam,     cphi],
        [ cphi*clam,     cphi*slam,     sphi]
    ])
    d_ecef = np.stack([e, n, u], axis=-1) @ R
    X = X0 + d_ecef[...,0]; Y = Y0 + d_ecef[...,1]; Z = Z0 + d_ecef[...,2]
    return ecef_to_geodetic_deg(X, Y, Z)

def latlon_to_xy(lat, lon, lat0, lon0, h0, h=None):
    if h is None:
        h = np.zeros_like(np.asarray(lat, dtype=float))
    e, n, _ = geodetic_to_enu(np.asarray(lat, float), np.asarray(lon, float), np.asarray(h, float),
                              float(lat0), float(lon0), float(h0))
    return e, n

def xy_to_latlon(x: np.ndarray, y: np.ndarray, lat0: float, lon0: float, h0: float = 0.0):
    R_earth = 6378137.0
    lat0_rad = np.radians(lat0)
    lat = (y / R_earth) * (180.0 / np.pi) + lat0
    lon = (x / (R_earth * np.cos(lat0_rad))) * (180.0 / np.pi) + lon0
    return np.asarray(lat), np.asarray(lon)

def polyline_length_xy(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2: return 0.0
    dx = np.diff(x); dy = np.diff(y)
    return float(np.sum(np.hypot(dx, dy)))

def _is_nan(x) -> bool:
    try: return bool(np.isnan(x))
    except Exception: return False

def _default_alt(S: Dict[str, Any]) -> float:
    cm = S.get("origin_alt_cm", 0) or 0
    return float(cm)/100.0

def bbox_latlon_from_road(
    road: dict,
    *,
    nodes_key: str = "nodes",
    lat_key: str = "lat",
    lon_key: str = "lon",
    order: str = "latlon",
):
    nodes = road[nodes_key]
    if isinstance(nodes, dict):
        it = nodes.values()
    else:
        it = nodes

    min_lat = min_lon = math.inf
    max_lat = max_lon = -math.inf

    for nd in it:
        if not isinstance(nd, dict):
            continue
        lat = nd.get(lat_key, None)
        lon = nd.get(lon_key, None)
        if lat is None or lon is None:
            continue
        if isinstance(lat, float) and math.isnan(lat):
            continue
        if isinstance(lon, float) and math.isnan(lon):
            continue
        if lat < min_lat: min_lat = lat
        if lat > max_lat: max_lat = lat
        if lon < min_lon: min_lon = lon
        if lon > max_lon: max_lon = lon

    if not all(math.isfinite(v) for v in (min_lat, min_lon, max_lat, max_lon)):
        raise ValueError("No valid (lat, lon) found in road nodes.")

    if order == "latlon":
        return (min_lat, min_lon, max_lat, max_lon)
    elif order == "lonlat":
        return (min_lon, min_lat, max_lon, max_lat)
    else:
        raise ValueError("order must be 'latlon' or 'lonlat'")