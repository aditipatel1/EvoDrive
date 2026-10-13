"""
compare.py - rule-based driver vs evolved drivers on FRESH test tracks.

Why fresh tracks? The 8 validation tracks were used to CHOOSE which car
got saved, so a score on them is biased upward (like grading yourself on
the questions you picked). The test tracks here use a different seed
(TEST_SEED) and are never seen during training or selection.

Why several seeds? One training run can be lucky or unlucky. Training
with seeds 1, 2, 3 ... and reporting mean +/- spread shows whether the
result is repeatable.

First train some seeds, e.g.:
    python3 train.py --seed 1 --out runs/seed1
    python3 train.py --seed 2 --out runs/seed2
    python3 train.py --seed 3 --out runs/seed3
Then:
    python3 compare.py
    python3 compare.py --runs runs_grip          # grip-mode drivers
Results are printed and saved to compare_results.csv.
The rule-based driver always races under the same physics as the evolved
drivers being compared (arcade or grip), so the comparison stays fair.
"""
import argparse
import csv
import glob
import os

import numpy as np

from car import rule_based_driver
from track import generate_track
from train import simulate, network_driver, load_network, SUCCESS_LAPS, VAL_SEED

TEST_SEED = 2024        # different from VAL_SEED (12345)
N_TEST = 100


def evaluate(driver, tracks):
    """Laps on each track for one driver (one car per track)."""
    return np.array([simulate(t, [driver])[0] for t in tracks])


def main(files, n_test=N_TEST, half_width=22, wobble=3.0):
    assert TEST_SEED != VAL_SEED
    if not files:
        raise SystemExit("No trained networks found. Train first, e.g.\n"
                         "  python3 train.py --seed 1 --out runs/seed1")
    drivers = [network_driver(load_network(path)) for path in files]
    kinds = {d.grip for d in drivers}
    if len(kinds) > 1:
        raise SystemExit("These files mix arcade and grip drivers; compare one kind at a time "
                         "(e.g. --runs runs or --runs runs_grip).")
    grip = kinds.pop()
    baseline = lambda readings: rule_based_driver(readings)
    baseline.grip = grip                      # same physics as the evolved drivers

    rng = np.random.default_rng(TEST_SEED)
    tracks = [generate_track(rng, half_width=half_width, wobble=wobble)
              for _ in range(n_test)]
    print(f"{n_test} fresh test tracks (seed {TEST_SEED}, "
          f"half_width={half_width}, wobble={wobble}); "
          f"{'grip' if grip else 'arcade'} physics; "
          f"'completed' = at least {SUCCESS_LAPS:.0f} lap\n")

    rows = []
    rb = evaluate(baseline, tracks)
    rows.append(dict(driver="rule-based", completed=np.mean(rb >= SUCCESS_LAPS),
                     mean_laps=rb.mean(), beats_rule_based=""))

    evolved = []
    for path, driver in zip(files, drivers):
        ev = evaluate(driver, tracks)
        evolved.append(ev)
        # paired comparison: same track, who drove further?
        wins = np.mean(ev > rb + 1e-9)
        rows.append(dict(driver=f"evolved ({path})",
                         completed=np.mean(ev >= SUCCESS_LAPS),
                         mean_laps=ev.mean(), beats_rule_based=wins))

    print(f"{'driver':<40} {'completed':>9} {'mean laps':>10} {'beats rule':>11}")
    for r in rows:
        beats = "" if r["beats_rule_based"] == "" else f"{r['beats_rule_based']*100:.0f}%"
        print(f"{r['driver']:<40} {r['completed']*100:8.0f}% {r['mean_laps']:10.2f} {beats:>11}")

    done = np.array([np.mean(ev >= SUCCESS_LAPS) for ev in evolved]) * 100
    laps = np.array([ev.mean() for ev in evolved])
    print(f"\nevolved, {len(evolved)} seed(s): completed {done.mean():.0f}% "
          f"(min {done.min():.0f}%, max {done.max():.0f}%), "
          f"mean laps {laps.mean():.2f} (min {laps.min():.2f}, max {laps.max():.2f})")
    print(f"rule-based:          completed {np.mean(rb >= SUCCESS_LAPS)*100:.0f}%, "
          f"mean laps {rb.mean():.2f}")
    if len(evolved) < 3:
        print("note: fewer than 3 seeds - treat this as weak evidence.")

    with open("compare_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("saved compare_results.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="*",
                   help="best_network.npy files (default: runs/*/best_network.npy)")
    p.add_argument("--n-test", type=int, default=N_TEST)
    # stress test: narrower, bendier tracks than training (training uses 22, 3.0)
    p.add_argument("--half-width", type=float, default=22)
    p.add_argument("--wobble", type=float, default=3.0)
    p.add_argument("--runs", default="runs",
                   help="folder of training runs to compare (default runs; grip: runs_grip)")
    a = p.parse_args()
    main(a.files or sorted(glob.glob(os.path.join(a.runs, "*", "best_network.npy"))),
         a.n_test, a.half_width, a.wobble)
