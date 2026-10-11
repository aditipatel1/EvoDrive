"""
Demo: 50 cars, each driven by a RANDOM neural network.

Watch what happens: nearly all of them crash almost immediately.
That's generation 0 - the starting point evolution will improve on.

Needs main.py (your renamed stage1.py) in the same folder.

Controls:  R = new random population    ESC = quit
"""
import math
import numpy as np
import pygame

from main import Car, WALLS, WIDTH, HEIGHT, SENSOR_ANGLES, SENSOR_RANGE
from network import NeuralNetwork

POPULATION = 50
MAX_FRAMES = 900          # end the round even if some cars are still alive


def new_population():
    return [(Car(), NeuralNetwork()) for _ in range(POPULATION)]


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Random networks - generation 0")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)

    population = new_population()
    frame = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    population = new_population()
                    frame = 0

        # SENSE -> THINK -> ACT, once per car
        for car, net in population:
            if car.crashed:
                continue
            readings = car.sense()
            inputs = np.array(readings) / SENSOR_RANGE      # scale to 0..1
            steer, throttle = net.forward(inputs)
            car.update(steer, throttle)
        frame += 1

        # draw
        screen.fill((30, 30, 35))
        for p1, p2 in WALLS:
            pygame.draw.line(screen, (220, 220, 220), p1, p2, 3)
        for car, _ in population:
            color = (90, 90, 100) if car.crashed else (70, 140, 255)
            pygame.draw.circle(screen, color, (car.x, car.y), 6)

        alive = sum(not car.crashed for car, _ in population)
        best = max(car.distance for car, _ in population)
        for i, text in enumerate([f"alive: {alive}/{POPULATION}",
                                  f"best distance: {best:.0f}",
                                  f"frame: {frame}/{MAX_FRAMES}",
                                  "R = new random population"]):
            screen.blit(font.render(text, True, (230, 230, 230)),
                        (270, 230 + i * 22))

        if alive == 0 or frame >= MAX_FRAMES:
            screen.blit(font.render("Round over - press R", True, (255, 200, 60)),
                        (270, 330))

        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()