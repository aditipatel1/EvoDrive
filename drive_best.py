"""
drive_best.py - watch your best evolved driver on brand-new random tracks.

Needs best_network.npy (made by train.py).

Controls:  N = new random track   R = restart   S = toggle sensor rays
           ESC = quit
Run:       python3 drive_best.py
           python3 drive_best.py --file runs/seed2/best_network.npy
"""
import argparse

import numpy as np
import pygame

from car import Car
from network import NeuralNetwork
from track import generate_track, WIDTH, HEIGHT
from train import network_driver


def load_driver(path):
    net = NeuralNetwork()
    net.set_vector(np.load(path))
    return network_driver(net)


def main(path="best_network.npy", max_frames=None):
    """max_frames is only for automatic testing (check_setup.py)."""
    driver = load_driver(path)
    rng = np.random.default_rng()
    track = generate_track(rng)
    car = Car(track)
    show_rays = True

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("EvoDrive - best driver")
    font = pygame.font.SysFont("consolas", 16)
    clock = pygame.time.Clock()

    frame = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_n:
                    track = generate_track(rng)
                    car = Car(track)
                elif event.key == pygame.K_r:
                    car = Car(track)
                elif event.key == pygame.K_s:
                    show_rays = not show_rays

        readings = car.sense()                       # SENSE
        if not car.crashed:
            car.update(*driver(readings))            # THINK + ACT

        screen.fill((30, 30, 35))
        track.draw(screen)
        car.draw(screen, readings=readings if show_rays else None)
        lines = [f"laps: {max(car.laps, 0):.2f}   speed: {car.speed:.1f}",
                 "N = new track   R = restart   S = rays   ESC = quit"]
        if car.crashed:
            lines.append("CRASHED - press R or N")
        for i, text in enumerate(lines):
            screen.blit(font.render(text, True, (230, 230, 230)), (10, 10 + i * 20))
        pygame.display.flip()
        clock.tick(60)

        frame += 1
        if max_frames is not None and frame >= max_frames:
            running = False
    pygame.quit()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--file", default="best_network.npy")
    main(p.parse_args().file)