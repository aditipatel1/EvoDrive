"""
export_replay.py - record one run of an evolved driver as replay.json,
for the 3D viewer (viewer.html).

The simulation is still 2D. This file records exactly what the car did on
every frame (position, heading, speed, its 5 sensor readings) plus the
track's walls, so the 3D viewer can redraw the SAME run from any camera.
Nothing is re-simulated in 3D: what you watch is what was scored.

Run:   python3 export_replay.py                                # runs/seed1, random normal track
       python3 export_replay.py --stress                       # narrow, bendy road
       python3 export_replay.py --any-result                   # keep the first track even if it crashes
       python3 export_replay.py --track track_laguna_seca.json --out replay_laguna.json
"""
import argparse
import json

import numpy as np

from car import Car, SENSOR_ANGLES, SENSOR_RANGE, CAR_RADIUS, GRIP_G
from track import generate_track
from train import network_driver, load_network, MAX_FRAMES, GRIP_MAX_FRAMES, STALL_FRAMES

TRACK_SEED = 7
PIXELS_PER_METER = 4.0     # scale used by the viewer: 1 sim pixel = 0.25 m


def run(driver, track, laps_target, max_frames):
    """Drive one car and record every frame. Stops on a crash, on reaching
    laps_target, on the time limit, or if the car stops making progress."""
    car = Car(track, grip=getattr(driver, "grip", False))
    frames = []
    result = "time limit"
    while True:
        obs = car.observe()
        readings = obs[:5]
        frames.append([round(car.x, 2), round(car.y, 2), round(car.angle, 2),
                       round(car.speed, 3), round(max(car.laps, 0), 4),
                       [round(float(r), 1) for r in readings], int(car.sliding)])
        if car.crashed:
            result = "crashed"; break
        if car.laps >= laps_target:
            result = "finished"; break
        if car.frames_since_best >= STALL_FRAMES:
            result = "stalled"; break
        if car.frames >= max_frames:
            break
        car.update(*driver(obs))
    return frames, car, result


def main(path, out, stress, any_result, track_file, laps_target, max_frames):
    driver = network_driver(load_network(path))      # arcade or grip, from the file

    extra = {}
    if track_file:                                   # a real circuit from osm_track.py
        from osm_track import load_track
        track, info = load_track(track_file)
        laps_target = laps_target or 1
        # enough time for the laps at an average of ~3 px per frame
        max_frames = max_frames or int(laps_target * track.length / 3) + 600
        frames, car, result = run(driver, track, laps_target, max_frames)
        hw, wb, attempt = info["half_width"], None, 1
        extra = {k: info[k] for k in ("title", "where", "source", "attribution",
                                      "length_m", "official_length_m") if k in info}
    else:                                            # a random track, as in training
        hw, wb = (18, 4.0) if stress else (22, 3.0)
        laps_target = laps_target or 2
        max_frames = max_frames or (GRIP_MAX_FRAMES if driver.grip else MAX_FRAMES)
        rng = np.random.default_rng(TRACK_SEED)
        for attempt in range(1, 31):
            track = generate_track(rng, half_width=hw, wobble=wb)
            frames, car, result = run(driver, track, laps_target, max_frames)
            if any_result or car.laps >= 1.0:
                break
        else:
            raise SystemExit("Driver didn't finish any of 30 tracks - try --any-result.")

    rnd = lambda a: [[round(float(x), 1), round(float(y), 1)] for x, y in a]
    replay = {
        "meta": {"driver": path, "half_width": hw, "wobble": wb,
                 "track_attempt": attempt, "fps": 60,
                 "pixels_per_meter": PIXELS_PER_METER,
                 "car_radius": CAR_RADIUS, "sensor_range": SENSOR_RANGE,
                 "sensor_angles_deg": [round(float(np.degrees(a))) for a in SENSOR_ANGLES],
                 "laps_target": laps_target, "result": result,
                 "physics": "grip" if driver.grip else "arcade",
                 "grip_g": GRIP_G if driver.grip else None,
                 "laps": round(max(car.laps, 0), 3), **extra},
        "track": {"center": rnd(track.center), "wall_a": rnd(track.wall_a),
                  "wall_b": rnd(track.wall_b)},
        # each frame: x, y, heading (deg), speed (px/frame), laps, [5 sensor distances], sliding 0/1
        "frames": frames,
    }
    with open(out, "w") as f:
        json.dump(replay, f, separators=(",", ":"))
    name = extra.get("title", f"random track (attempt {attempt})")
    print(f"saved {out}: {name}, {len(frames)} frames, result {result} "
          f"({replay['meta']['laps']:.2f} of {laps_target} laps)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--file", default="runs/seed1/best_network.npy",
                   help="a trained driver (arcade or grip; detected from the file)")
    p.add_argument("--out", default="replay.json")
    p.add_argument("--stress", action="store_true", help="narrow, bendy random road")
    p.add_argument("--any-result", action="store_true",
                   help="keep the first random track even if the car crashes")
    p.add_argument("--track", help="a real circuit made by osm_track.py, e.g. track_laguna_seca.json")
    p.add_argument("--laps", type=int, help="laps to drive (default 2 random, 1 real)")
    p.add_argument("--max-frames", type=int, help="time limit in frames (60 per second)")
    a = p.parse_args()
    main(a.file, a.out, a.stress, a.any_result, a.track, a.laps, a.max_frames)
