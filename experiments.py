"""
experiments.py - which training settings actually matter?

Trains every setting in EXPERIMENTS with several seeds, then scores each
saved driver on two sets of fresh test tracks:
  normal : same difficulty as training (half_width 22, wobble 3.0)
  stress : narrower, bendier roads it never trained on (18, 4.0)

Why several seeds per setting? A difference between two settings only
means something if it's bigger than the difference between two seeds of
the SAME setting. The summary prints min-max across seeds so you can see
whether the ranges overlap.

Runs go to runs/exp/<setting>/seed<N>/. A run that already finished is
skipped, so you can stop with Ctrl+C and start again later.

Run:   python3 experiments.py                 # everything (~25 min on an M-series Mac)
       python3 experiments.py --seeds 1 2     # fewer seeds, faster
       python3 experiments.py --only baseline tracks4
Results: experiments_results.csv (one row per run) and a printed summary.
"""
import argparse
import contextlib
import csv
import inspect
import json
import os
import time

import numpy as np

import train
from car import rule_based_driver
from compare import evaluate, TEST_SEED
from track import generate_track

# name: settings passed to train.train()
EXPERIMENTS = {
    "baseline":      dict(tracks_per_gen=2, mutation_rate=0.1,  mutation_strength=0.4),
    "tracks4":       dict(tracks_per_gen=4, mutation_rate=0.1,  mutation_strength=0.4),
    "low_mutation":  dict(tracks_per_gen=2, mutation_rate=0.05, mutation_strength=0.2),
    "high_mutation": dict(tracks_per_gen=2, mutation_rate=0.2,  mutation_strength=0.8),
}
TESTS = {"normal": (22, 3.0), "stress": (18, 4.0)}
N_TEST = 100


def check_train_version():
    params = inspect.signature(train.train).parameters
    if "mutation_rate" not in params:
        raise SystemExit("Your train.py is the OLD version (no mutation settings).\n"
                         "Replace it with the new train.py first, then check:\n"
                         "  grep -c mutation train.py     (should print 10)")


def run_one(name, settings, seed, generations):
    out = os.path.join("runs", "exp", name, f"seed{seed}")
    done_marker = os.path.join(out, "history.csv")
    if os.path.exists(done_marker) and os.path.exists(os.path.join(out, "best_network.npy")):
        with open(os.path.join(out, "config.json")) as f:
            cfg = json.load(f)
        if cfg.get("generations") == generations:
            print(f"  {name} seed {seed}: already done, skipping")
            return out
    t0 = time.time()
    os.makedirs(out, exist_ok=True)
    # training prints a line per generation; send it to a log file instead
    with open(os.path.join(out, "train_log.txt"), "w") as log, \
            contextlib.redirect_stdout(log):
        train.train(generations=generations, seed=seed, out=out, **settings)
    print(f"  {name} seed {seed}: trained in {time.time() - t0:.0f}s")
    return out


def main(names, seeds, generations):
    check_train_version()
    test_tracks = {}
    for test, (hw, wb) in TESTS.items():
        rng = np.random.default_rng(TEST_SEED)
        test_tracks[test] = [generate_track(rng, half_width=hw, wobble=wb)
                             for _ in range(N_TEST)]

    rows = []
    for name in names:
        print(f"{name}: {EXPERIMENTS[name]}")
        for seed in seeds:
            out = run_one(name, EXPERIMENTS[name], seed, generations)
            net = train.load_network(os.path.join(out, "best_network.npy"))
            row = dict(setting=name, seed=seed, **EXPERIMENTS[name])
            for test, tracks in test_tracks.items():
                laps = evaluate(train.network_driver(net), tracks)
                row[f"{test}_completed"] = round(float(np.mean(laps >= train.SUCCESS_LAPS)), 4)
                row[f"{test}_laps"] = round(float(laps.mean()), 4)
            rows.append(row)

    with open("experiments_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{N_TEST} test tracks per test, {len(seeds)} seed(s) per setting, "
          f"'completed' = at least 1 lap")
    print(f"{'setting':<15} {'normal: mean (min-max)':>24} {'stress: mean (min-max)':>24}")
    for name in names:
        r = [x for x in rows if x["setting"] == name]
        cells = []
        for test in TESTS:
            v = np.array([x[f"{test}_completed"] for x in r]) * 100
            cells.append(f"{v.mean():5.0f}% ({v.min():3.0f}-{v.max():3.0f}%)")
        print(f"{name:<15} {cells[0]:>24} {cells[1]:>24}")
    for test, tracks in test_tracks.items():
        rb = evaluate(rule_based_driver, tracks)
        print(f"rule-based on {test}: {np.mean(rb >= train.SUCCESS_LAPS)*100:.0f}%")
    if len(seeds) < 3:
        print("note: fewer than 3 seeds per setting - treat differences as weak evidence.")
    print("saved experiments_results.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--only", nargs="+", choices=list(EXPERIMENTS),
                   default=list(EXPERIMENTS))
    p.add_argument("--generations", type=int, default=train.GENERATIONS)
    a = p.parse_args()
    main(a.only, a.seeds, a.generations)
