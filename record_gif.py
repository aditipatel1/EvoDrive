"""
record_gif.py - record a demo GIF of an evolved driver (for the README).

Runs headlessly (no window): draws each frame in memory, keeps every
FRAME_STEP-th frame, and saves an animated GIF with Pillow.

Run:   python3 record_gif.py                                  # runs/seed1, normal track
       python3 record_gif.py --file runs/seed2/best_network.npy
       python3 record_gif.py --stress                         # narrow, bendy road
Saves demo.gif (or the name given with --out).
"""
import argparse
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame
from PIL import Image

from car import Car
from track import generate_track, WIDTH, HEIGHT
from train import network_driver, load_network, MAX_FRAMES, GRIP_MAX_FRAMES

FRAME_STEP = 3          # keep every 3rd simulation frame
GIF_WIDTH = 640         # output size (keeps the file small)
TRACK_SEED = 7          # which demo track; change it to try another

BACKGROUND = (30, 30, 35)
# every color the simulation draws (see track.py / car.py)
SIM_COLORS = [(220, 220, 220), (255, 255, 255), (230, 230, 230), (70, 140, 255),
              (110, 60, 60), (80, 200, 120), (255, 200, 60), (90, 200, 90)]


def make_palette():
    """A fixed GIF palette: each drawn color blended with the background in
    8 steps (for smooth, anti-aliased edges). A GIF can only hold 256 colors;
    choosing them ourselves keeps the small blue car and green rays from
    being turned gray by the automatic palette."""
    colors = []
    for c in SIM_COLORS:
        for t in np.linspace(0, 1, 8):
            colors.append(tuple(int(BACKGROUND[i] + t * (c[i] - BACKGROUND[i]))
                                for i in range(3)))
    flat = [v for c in colors for v in c]
    flat += flat[-3:] * (256 - len(colors))      # pad to 256 entries
    pal = Image.new("P", (1, 1))
    pal.putpalette(flat)
    return pal


def find_demo_track(driver, half_width, wobble, laps=1.0, tries=30):
    """Pick a track the driver actually completes, so the demo shows a full lap.
    (Honest note for your README: the GIF shows a success; the success RATE
    comes from compare.py, not from this video.)"""
    rng = np.random.default_rng(TRACK_SEED)
    for _ in range(tries):
        track = generate_track(rng, half_width=half_width, wobble=wobble)
        car = Car(track, grip=driver.grip)
        limit = GRIP_MAX_FRAMES if driver.grip else MAX_FRAMES
        while not car.crashed and car.frames < limit and car.laps < laps:
            car.update(*driver(car.observe()))
        if car.laps >= laps:
            return track
    raise SystemExit("Driver didn't finish any of the demo tracks - try another --file.")


def main(path, out, stress):
    hw, wb = (18, 4.0) if stress else (22, 3.0)
    driver = network_driver(load_network(path))
    track = find_demo_track(driver, hw, wb)

    pygame.init()
    screen = pygame.Surface((WIDTH, HEIGHT))
    font = pygame.font.SysFont("consolas", 22)
    car = Car(track, grip=driver.grip)
    frames = []
    palette = make_palette()
    limit = GRIP_MAX_FRAMES if driver.grip else MAX_FRAMES
    while not car.crashed and car.laps < 1.0 and car.frames < limit:
        obs = car.observe()
        readings = obs[:5]
        car.update(*driver(obs))
        if car.frames % FRAME_STEP:
            continue
        screen.fill(BACKGROUND)
        track.draw(screen)
        car.draw(screen, readings=readings)
        label = f"EvoDrive - evolved driver, unseen track   laps {max(car.laps, 0):.2f}"
        screen.blit(font.render(label, True, (230, 230, 230)), (16, 12))
        raw = pygame.image.tostring(screen, "RGB")
        img = Image.frombytes("RGB", (WIDTH, HEIGHT), raw)
        img = img.resize((GIF_WIDTH, GIF_WIDTH * HEIGHT // WIDTH), Image.LANCZOS)
        frames.append(img.quantize(palette=palette, dither=Image.Dither.NONE))
    pygame.quit()

    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=1000 // 20, loop=0, optimize=True)  # 20 fps
    size_mb = os.path.getsize(out) / 1e6
    print(f"saved {out}: {len(frames)} frames, {size_mb:.1f} MB")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--file", default="runs/seed1/best_network.npy")
    p.add_argument("--out", default="demo.gif")
    p.add_argument("--stress", action="store_true", help="narrow, bendy road")
    a = p.parse_args()
    main(a.file, a.out, a.stress)
