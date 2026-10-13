"""
osm_track.py - turn a real race circuit from OpenStreetMap into an EvoDrive track.

Steps:
  1. DOWNLOAD  every road tagged highway=raceway near the circuit, from the
               OpenStreetMap Overpass API (saved to osm_<name>_raw.json, so it
               is downloaded only once).
  2. STITCH    OSM stores a circuit as several pieces ("ways") that share end
               points. Join them into closed loops and keep the loop whose
               length is closest to the circuit's official length (this drops
               pit lanes, karting tracks and other side roads).
  3. PROJECT   latitude/longitude -> meters on a flat local map (north up),
               then meters -> simulator pixels (4 px = 1 m, same as the viewer).
  4. RESAMPLE  evenly spaced points every few meters, lightly smoothed, so the
               walls, sensors and lap counter behave like on random tracks.
  5. SAVE      track_<name>.json, which export_replay.py --track can load.

Map data (c) OpenStreetMap contributors, available under the Open Database
License (ODbL). Anything showing this data must say so.

Run:   python3 osm_track.py                      # Laguna Seca (default)
       python3 osm_track.py --reverse            # drive it the other direction
       python3 osm_track.py --list               # circuits it knows
"""
import argparse
import json
import math
import os
import ssl
import sys
import urllib.parse
import urllib.request

import numpy as np

from track import Track, min_turn_radius

PIXELS_PER_METER = 4.0
EARTH_RADIUS_M = 6371000.0
ATTRIBUTION = "Map data © OpenStreetMap contributors (ODbL)"

# name: centre (lat, lon), search radius (m), official lap length (m)
# Lengths are the published track lengths and only used to pick the right loop.
CIRCUITS = {
    "laguna_seca": dict(title="WeatherTech Raceway Laguna Seca",
                        where="Monterey, California",
                        lat=36.5843, lon=-121.7535, radius_m=1500, length_m=3602),
    "sonoma":      dict(title="Sonoma Raceway", where="Sonoma, California",
                        lat=38.1617, lon=-122.4549, radius_m=1500, length_m=4056),
    "willow_springs": dict(title="Willow Springs International Raceway",
                           where="Rosamond, California",
                           lat=34.8710, lon=-118.2640, radius_m=1800, length_m=4023),
}

OVERPASS_URLS = ["https://overpass-api.de/api/interpreter",
                 "https://overpass.kumi.systems/api/interpreter"]


# ------------------------------------------------------------------ 1. download
def ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def download(c, raw_path):
    if os.path.exists(raw_path):
        print(f"using saved map data {raw_path}")
        with open(raw_path) as f:
            return json.load(f)
    query = (f'[out:json][timeout:60];'
             f'way["highway"="raceway"](around:{c["radius_m"]},{c["lat"]},{c["lon"]});'
             f'out geom;')
    data = urllib.parse.urlencode({"data": query}).encode()
    last_error = None
    for url in OVERPASS_URLS:
        print(f"downloading raceway roads near {c['title']} from {url} ...")
        req = urllib.request.Request(url, data=data,
                                     headers={"User-Agent": "EvoDrive-student-project/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=90, context=ssl_context()) as r:
                osm = json.load(r)
            with open(raw_path, "w") as f:
                json.dump(osm, f)
            print(f"saved {raw_path} ({len(osm.get('elements', []))} roads)")
            return osm
        except ssl.SSLError as e:
            raise SystemExit(
                f"SSL certificate error: {e}\n"
                "On a Mac with Python from python.org, run this once, then try again:\n"
                f'  open "/Applications/Python {sys.version_info.major}.{sys.version_info.minor}'
                '/Install Certificates.command"')
        except Exception as e:                       # network down, server busy, ...
            last_error = e
            if "CERTIFICATE_VERIFY_FAILED" in str(e):
                raise SystemExit(
                    "SSL certificate error. On a Mac with Python from python.org, run once:\n"
                    f'  open "/Applications/Python {sys.version_info.major}.{sys.version_info.minor}'
                    '/Install Certificates.command"')
            print(f"  failed: {e}")
    raise SystemExit(f"Could not download map data ({last_error}). "
                     "Check your internet connection and try again in a minute.")


# ------------------------------------------------------------------ 2. stitch
def is_side_road(tags):
    """Pit lanes and service roads are not part of the racing line."""
    text = " ".join(f"{k}={v}" for k, v in tags.items()).lower()
    return any(w in text for w in ("pit", "service", "paddock", "access"))


def find_loops(ways, max_ways=40):
    """Return closed loops as lists of (way, reversed?) by joining ways at shared
    end nodes. Uses depth-first search; circuits have only a handful of pieces."""
    ends = {}
    for i, w in enumerate(ways):
        for node in (w["nodes"][0], w["nodes"][-1]):
            ends.setdefault(node, []).append(i)
    loops, seen = [], set()

    def walk(start_node, node, path, used):
        if len(path) > max_ways:
            return
        for i in ends.get(node, []):
            if i in used:
                continue
            w = ways[i]
            rev = w["nodes"][-1] == node
            nxt = w["nodes"][0] if rev else w["nodes"][-1]
            step = path + [(i, rev)]
            if nxt == start_node:
                key = frozenset(j for j, _ in step)
                if key not in seen:
                    seen.add(key)
                    loops.append(step)
                continue
            walk(start_node, nxt, step, used | {i})

    for i, w in enumerate(ways):
        if w["nodes"][0] == w["nodes"][-1]:           # a single closed way
            key = frozenset([i])
            if key not in seen:
                seen.add(key)
                loops.append([(i, False)])
        else:
            walk(w["nodes"][0], w["nodes"][-1], [(i, False)], {i})
    return loops


def loop_points(ways, loop):
    pts = []
    for i, rev in loop:
        g = ways[i]["geometry"]
        g = g[::-1] if rev else g
        pts.extend(g if not pts else g[1:])        # skip the shared joint
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    return pts


# ------------------------------------------------------------------ 3. project
def to_meters(points, lat0, lon0):
    """Equirectangular projection: accurate to well under 1% over a few km."""
    lat = np.radians([p["lat"] for p in points])
    lon = np.radians([p["lon"] for p in points])
    x = EARTH_RADIUS_M * (lon - math.radians(lon0)) * math.cos(math.radians(lat0))
    y = -EARTH_RADIUS_M * (lat - math.radians(lat0))      # screen y grows downward
    return np.stack([x, y], axis=1)


def loop_length(xy):
    return float(np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1).sum())


# ------------------------------------------------------------------ 4. resample
def resample(xy, step_m):
    """Points every step_m meters along the closed polyline, then a light
    [1, 2, 1] smoothing pass to soften the corners between OSM nodes."""
    closed = np.vstack([xy, xy[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    keep = seg > 1e-6
    cum = np.concatenate([[0], np.cumsum(seg)])
    total = cum[-1]
    n = max(16, int(round(total / step_m)))
    s = np.linspace(0, total, n, endpoint=False)
    x = np.interp(s, cum, closed[:, 0])
    y = np.interp(s, cum, closed[:, 1])
    pts = np.stack([x, y], axis=1)
    pts = 0.25 * np.roll(pts, 1, axis=0) + 0.5 * pts + 0.25 * np.roll(pts, -1, axis=0)
    return pts, int(keep.sum())


# ------------------------------------------------------------------ load (used by export_replay.py)
def load_track(path):
    """Read track_<name>.json and return (Track, info dict)."""
    with open(path) as f:
        info = json.load(f)
    center = np.array(info["center"], dtype=float)
    return Track(center, info["half_width"]), info


# ------------------------------------------------------------------ main
def build(name, reverse, start_offset, half_width, step_m, raw_path=None, out=None):
    c = CIRCUITS[name]
    raw_path = raw_path or f"osm_{name}_raw.json"
    out = out or f"track_{name}.json"
    osm = download(c, raw_path)

    ways = [e for e in osm.get("elements", [])
            if e.get("type") == "way" and len(e.get("nodes", [])) >= 2
            and len(e.get("geometry", [])) == len(e["nodes"])]
    racing = [w for w in ways if not is_side_road(w.get("tags", {}))]
    print(f"{len(ways)} raceway roads found, {len(racing)} after removing pit/service roads")
    pool = racing
    loops = find_loops(pool)
    if not loops:                                   # side-road filter was too strict
        pool = ways
        loops = find_loops(pool)
    if not loops:
        raise SystemExit("No closed loop found in the map data. The circuit may be "
                         "mapped differently; try another circuit.")

    candidates = []
    for lp in loops:
        xy = to_meters(loop_points(pool, lp), c["lat"], c["lon"])
        length = loop_length(xy)
        candidates.append((abs(length - c["length_m"]), length, xy, lp))
    candidates.sort(key=lambda t: t[0])
    err, length, xy, lp = candidates[0]
    print(f"{len(loops)} closed loop(s); picked one of {len(lp)} piece(s), "
          f"{length:.0f} m (official length {c['length_m']} m, "
          f"difference {100 * err / c['length_m']:.1f}%)")
    if err / c["length_m"] > 0.15:
        print("WARNING: the best loop is more than 15% off the official length - "
              "check the map in the 3D viewer before trusting results.")

    pts_m, _ = resample(xy, step_m)
    if reverse:
        pts_m = pts_m[::-1]
    if start_offset:
        pts_m = np.roll(pts_m, -int(start_offset / step_m), axis=0)
    px = pts_m * PIXELS_PER_METER
    px -= px.min(axis=0) - 2 * half_width - 40           # positive coordinates with a margin

    tightest_m = min_turn_radius(px) / PIXELS_PER_METER
    road_m = 2 * half_width / PIXELS_PER_METER
    print(f"resampled to {len(px)} points every {step_m} m; tightest corner radius "
          f"{tightest_m:.1f} m; road width {road_m:.1f} m")
    if min_turn_radius(px) < half_width:
        print("WARNING: a corner is tighter than half the road width, so the inner wall "
              "folds over itself there. Try a smaller --half-width.")

    track = Track(px, half_width)                         # same class as random tracks
    info = {
        "name": name, "title": c["title"], "where": c["where"],
        "source": "OpenStreetMap", "attribution": ATTRIBUTION,
        "length_m": round(track.length / PIXELS_PER_METER, 1),
        "official_length_m": c["length_m"],
        "half_width": half_width, "pixels_per_meter": PIXELS_PER_METER,
        "reversed": reverse, "start_offset_m": start_offset,
        "tightest_corner_m": round(tightest_m, 1),
        "center": [[round(float(x), 2), round(float(y), 2)] for x, y in px],
    }
    with open(out, "w") as f:
        json.dump(info, f, separators=(",", ":"))
    print(f"saved {out}: {c['title']}, {info['length_m']:.0f} m lap")
    return track, info


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("circuit", nargs="?", default="laguna_seca", choices=list(CIRCUITS))
    p.add_argument("--reverse", action="store_true", help="drive the loop the other way")
    p.add_argument("--start-offset", type=float, default=0,
                   help="move the start this many meters along the track")
    p.add_argument("--half-width", type=float, default=22,
                   help="half road width in pixels (22 = 11 m road, as in training)")
    p.add_argument("--step", type=float, default=6.0, help="meters between track points")
    p.add_argument("--list", action="store_true", help="list known circuits")
    a = p.parse_args()
    if a.list:
        for k, v in CIRCUITS.items():
            print(f"{k:<15} {v['title']}, {v['where']} ({v['length_m']} m)")
        sys.exit()
    build(a.circuit, a.reverse, a.start_offset, a.half_width, a.step)
