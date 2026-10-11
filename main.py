"""
Self-driving car simulation - Stage 1
Car + track + ray-cast sensors + collision detection + rule-based baseline driver.

Controls:
    Arrow keys : drive manually (Up/Down = speed, Left/Right = steer)
    SPACE      : toggle the rule-based autopilot
    R          : reset the car
    ESC        : quit

Install:  pip install pygame
Run:      python stage1.py
"""
import math
import pygame

WIDTH, HEIGHT = 1000, 700
SENSOR_ANGLES = [-60, -30, 0, 30, 60]   # degrees relative to the car's heading
SENSOR_RANGE = 200                      # max distance a sensor can "see"
CAR_RADIUS = 10
MAX_SPEED = 5.0
START = (150, 125, 0)                   # x, y, heading (degrees)


# ---------------------------------------------------------------- track
def rect_walls(x, y, w, h):
    """Turn a rectangle into 4 wall segments: ((x1, y1), (x2, y2))."""
    p = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    return [(p[i], p[(i + 1) % 4]) for i in range(4)]


# An outer rectangle and an inner rectangle form a looping corridor.
WALLS = rect_walls(50, 50, 900, 600) + rect_walls(250, 200, 500, 300)


# ---------------------------------------------------------------- geometry
def ray_distance(ox, oy, dx, dy, seg):
    """Distance along a ray (origin ox,oy, direction dx,dy) to a wall segment.
    Returns None if the ray doesn't hit it.

    Solves  O + t*D = P1 + u*S  using 2D cross products, where S is the
    segment's direction. t = distance along the ray, u = position on the
    segment (must be between 0 and 1 for a real hit).
    """
    (x1, y1), (x2, y2) = seg
    sx, sy = x2 - x1, y2 - y1
    denom = dx * sy - dy * sx
    if abs(denom) < 1e-9:               # ray is parallel to the wall
        return None
    t = ((x1 - ox) * sy - (y1 - oy) * sx) / denom
    u = ((x1 - ox) * dy - (y1 - oy) * dx) / denom
    if t >= 0 and 0 <= u <= 1:
        return t
    return None


def point_to_segment(px, py, seg):
    """Shortest distance from a point to a wall segment (for collisions)."""
    (x1, y1), (x2, y2) = seg
    sx, sy = x2 - x1, y2 - y1
    length_sq = sx * sx + sy * sy
    t = max(0, min(1, ((px - x1) * sx + (py - y1) * sy) / length_sq))
    return math.hypot(px - (x1 + t * sx), py - (y1 + t * sy))


# ---------------------------------------------------------------- car
class Car:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x, self.y, self.angle = START
        self.speed = 2.0
        self.crashed = False
        self.distance = 0.0             # will become the fitness score later

    def sense(self):
        """Fire one ray per sensor angle; return distances (the car's 'vision')."""
        readings = []
        for offset in SENSOR_ANGLES:
            a = math.radians(self.angle + offset)
            dx, dy = math.cos(a), math.sin(a)
            nearest = SENSOR_RANGE
            for wall in WALLS:
                d = ray_distance(self.x, self.y, dx, dy, wall)
                if d is not None and d < nearest:
                    nearest = d
            readings.append(nearest)
        return readings

    def update(self, steer, throttle):
        """steer: -1 (left) .. +1 (right).  throttle: -1 .. +1."""
        if self.crashed:
            return
        self.angle += steer * 4.0                           # turn
        self.speed = max(0.5, min(MAX_SPEED, self.speed + throttle * 0.1))
        a = math.radians(self.angle)
        self.x += math.cos(a) * self.speed                  # move forward
        self.y += math.sin(a) * self.speed
        self.distance += self.speed
        if any(point_to_segment(self.x, self.y, w) < CAR_RADIUS for w in WALLS):
            self.crashed = True


# ---------------------------------------------------------------- baseline driver
def rule_based_driver(readings):
    """Hand-written 'think' step: steer toward the side with more open space,
    slow down when something is close ahead. This is the baseline the
    neural network will have to beat in later stages."""
    far_left, left, front, right, far_right = readings
    steer = ((right + far_right) - (left + far_left)) / (2 * SENSOR_RANGE)
    steer = max(-1, min(1, steer * 3))
    throttle = 1 if front > 100 else -1
    return steer, throttle


# ---------------------------------------------------------------- main loop
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Self-driving sim - stage 1")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)
    car = Car()
    autopilot = False

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    autopilot = not autopilot
                elif event.key == pygame.K_r:
                    car.reset()

        # SENSE
        readings = car.sense()

        # THINK
        if autopilot:
            steer, throttle = rule_based_driver(readings)
        else:
            keys = pygame.key.get_pressed()
            steer = keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]
            throttle = keys[pygame.K_UP] - keys[pygame.K_DOWN]

        # ACT
        car.update(steer, throttle)

        # ---- draw
        screen.fill((30, 30, 35))
        for (p1, p2) in WALLS:
            pygame.draw.line(screen, (220, 220, 220), p1, p2, 3)
        for offset, d in zip(SENSOR_ANGLES, readings):
            a = math.radians(car.angle + offset)
            end = (car.x + math.cos(a) * d, car.y + math.sin(a) * d)
            pygame.draw.line(screen, (80, 200, 120), (car.x, car.y), end, 1)
            pygame.draw.circle(screen, (255, 200, 60), end, 3)
        color = (230, 70, 70) if car.crashed else (70, 140, 255)
        pygame.draw.circle(screen, color, (car.x, car.y), CAR_RADIUS)
        nose = (car.x + math.cos(math.radians(car.angle)) * CAR_RADIUS * 1.6,
                car.y + math.sin(math.radians(car.angle)) * CAR_RADIUS * 1.6)
        pygame.draw.line(screen, (255, 255, 255), (car.x, car.y), nose, 2)

        mode = "AUTOPILOT" if autopilot else "MANUAL"
        lines = [f"mode: {mode}   speed: {car.speed:.1f}   distance: {car.distance:.0f}",
                 "sensors: " + "  ".join(f"{r:5.0f}" for r in readings),
                 "SPACE = autopilot   R = reset"]
        if car.crashed:
            lines.append("CRASHED - press R")
        for i, text in enumerate(lines):
            screen.blit(font.render(text, True, (230, 230, 230)), (270, 230 + i * 22))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()