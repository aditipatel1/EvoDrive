"""
train.py - evolve neural-network drivers on random tracks.

Each generation:
  1. Make TRACKS_PER_GEN brand-new random training tracks.
  2. Every car drives every training track. Fitness = average laps
     (capped at MAX_LAPS), so a car must handle tracks it has never seen
     rather than memorise one.
  3. The generation's best car is tested on 8 FIXED validation tracks
     (seed 12345) that are never used for training. The car with the best
     validation score so far is saved to best_network.npy.
  4. evolution.next_generation() builds the next population.

Everything is logged to history.csv.

Run:
    python3 train.py                      # headless (fast)
    python3 train.py --window             # watch the first track each generation
    python3 train.py --seed 2 --generations 60 --tracks-per-gen 4 --out runs/seed2
    python3 train.py --mutation-rate 0.2 --mutation-strength 0.2 --out runs/mut_test

The settings of every run are saved to config.json in its --out folder,
so you can always tell later how a result was produced.
"""
import argparse
import csv
import json
import os
import time

import numpy as np

from car import Car, SENSOR_RANGE
from evolution import next_generation, MUTATION_RATE, MUTATION_STRENGTH
from network import NeuralNetwork
from track import generate_track, WIDTH, HEIGHT

POPULATION = 50
GENERATIONS = 45
TRACKS_PER_GEN = 2
MAX_FRAMES = 1200      # time limit per track (20 s at 60 fps)
MAX_LAPS = 2.0         # fitness cap; a car stops once it reaches this
STALL_FRAMES = 120     # stop a car that makes no new progress for 2 s
SUCCESS_LAPS = 1.0     # a track counts as "completed" after one full lap
VAL_SEED = 12345
VAL_TRACKS = 8


def network_driver(net):
    """Turn a network into a driver: sensors in, (steer, throttle) out.
    Sensor distances are divided by SENSOR_RANGE so inputs are 0..1."""
    return lambda readings: net.forward(readings / SENSOR_RANGE)


def simulate(track, drivers, max_frames=MAX_FRAMES, on_frame=None):
    """Drive every driver on `track` at the same time; return laps per driver.

    A car stops when it crashes, reaches MAX_LAPS, or stalls. on_frame
    (optional) is called every frame with the list of cars, for drawing;
    if it returns False the run is aborted early.
    """
    cars = [Car(track) for _ in drivers]
    active = list(range(len(cars)))
    for _ in range(max_frames):
        still = []
        for i in active:
            car = cars[i]
            car.update(*drivers[i](car.sense()))
            if not (car.crashed or car.laps >= MAX_LAPS
                    or car.frames_since_best >= STALL_FRAMES):
                still.append(i)
        active = still
        if on_frame is not None and on_frame(cars) is False:
            break
        if not active:
            break
    return np.array([min(max(c.laps, 0.0), MAX_LAPS) for c in cars])


def validation_tracks():
    rng = np.random.default_rng(VAL_SEED)
    return [generate_track(rng) for _ in range(VAL_TRACKS)]


def make_window_callback(track, generation):
    """Draw all cars each frame (only used with --window)."""
    import pygame
    if not pygame.get_init():
        pygame.init()
        pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("EvoDrive - training")
    screen = pygame.display.get_surface()
    font = pygame.font.SysFont("consolas", 16)
    clock = pygame.time.Clock()

    def draw(cars):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                raise KeyboardInterrupt
        screen.fill((30, 30, 35))
        track.draw(screen)
        for car in cars:
            car.draw(screen)
        alive = sum(not c.crashed for c in cars)
        text = f"generation {generation}   alive {alive}/{len(cars)}   (ESC = stop)"
        screen.blit(font.render(text, True, (230, 230, 230)), (10, 10))
        pygame.display.flip()
        clock.tick(120)
    return draw


def train(generations=GENERATIONS, seed=1, tracks_per_gen=TRACKS_PER_GEN,
          population=POPULATION, window=False, out=".",
          mutation_rate=MUTATION_RATE, mutation_strength=MUTATION_STRENGTH):
    os.makedirs(out, exist_ok=True)
    config = dict(generations=generations, seed=seed, tracks_per_gen=tracks_per_gen,
                  population=population, mutation_rate=mutation_rate,
                  mutation_strength=mutation_strength)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    rng = np.random.default_rng(seed)
    nets = [NeuralNetwork(rng=rng) for _ in range(population)]
    dnas = [n.to_vector() for n in nets]
    val = validation_tracks()
    best_val = (-1.0, -1.0)            # (success rate, mean laps)
    history = []

    print(f"seed {seed}: {population} cars, {generations} generations, "
          f"{tracks_per_gen} training tracks/generation, "
          f"mutation rate {mutation_rate}, strength {mutation_strength}")
    print(" gen | train best | train mean | val laps | val done")
    try:
        for gen in range(1, generations + 1):
            t0 = time.time()
            for net, dna in zip(nets, dnas):
                net.set_vector(dna)
            drivers = [network_driver(n) for n in nets]

            # ---- fitness on fresh training tracks
            scores = np.zeros(population)
            for k in range(tracks_per_gen):
                track = generate_track(rng)
                cb = make_window_callback(track, gen) if (window and k == 0) else None
                scores += simulate(track, drivers, on_frame=cb)
            fitness = scores / tracks_per_gen

            # ---- validate the generation's best on the fixed unseen tracks
            best_i = int(np.argmax(fitness))
            val_laps = np.array([simulate(t, [drivers[best_i]])[0] for t in val])
            val_success = float(np.mean(val_laps >= SUCCESS_LAPS))
            if (val_success, val_laps.mean()) > best_val:
                best_val = (val_success, val_laps.mean())
                np.save(os.path.join(out, "best_network.npy"), dnas[best_i])

            row = dict(generation=gen, train_best=round(fitness.max(), 4),
                       train_mean=round(fitness.mean(), 4),
                       val_laps=round(val_laps.mean(), 4),
                       val_success=round(val_success, 4))
            history.append(row)
            print(f" {gen:3d} |   {row['train_best']:6.2f}   |   {row['train_mean']:6.2f}   |"
                  f"  {row['val_laps']:5.2f}   |  {val_success*100:4.0f}%"
                  f"   ({time.time() - t0:.1f}s)")

            dnas = next_generation(dnas, fitness, rng, rate=mutation_rate,
                                   strength=mutation_strength)
    except KeyboardInterrupt:
        print("stopped early")

    with open(os.path.join(out, "history.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]) if history else
                                ["generation", "train_best", "train_mean",
                                 "val_laps", "val_success"])
        writer.writeheader()
        writer.writerows(history)
    print(f"best validation: {best_val[0]*100:.0f}% completed, "
          f"{best_val[1]:.2f} laps -> saved {os.path.join(out, 'best_network.npy')}")
    return history


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--generations", type=int, default=GENERATIONS)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--tracks-per-gen", type=int, default=TRACKS_PER_GEN)
    p.add_argument("--population", type=int, default=POPULATION)
    p.add_argument("--window", action="store_true", help="watch training")
    p.add_argument("--out", default=".", help="folder for best_network.npy and history.csv")
    p.add_argument("--mutation-rate", type=float, default=MUTATION_RATE,
                   help="chance each of the 50 numbers is nudged (default 0.1)")
    p.add_argument("--mutation-strength", type=float, default=MUTATION_STRENGTH,
                   help="size of each nudge, std of the noise (default 0.4)")
    a = p.parse_args()
    train(a.generations, a.seed, a.tracks_per_gen, a.population, a.window, a.out,
          a.mutation_rate, a.mutation_strength)