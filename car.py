"""
car.py - a car that can drive on ANY Track from track.py.

Same physics and sensors as stage1.py, with three upgrades:
  1. Works on any track (walls come from track.segments).
  2. Sensors are vectorized with NumPy: all 5 rays are tested against all
     wall segments in one batch of array maths instead of nested loops.
  3. Measures PROGRESS along the track's centerline, in laps. Driving
     backwards counts as negative progress, so a car can't cheat by
     circling in place or reversing.
"""
import math
import numpy as np

SENSOR_ANGLES = np.radians([-60, -30, 0, 30, 60])  # relative to heading
SENSOR_RANGE = 200.0
CAR_RADIUS = 8
MAX_SPEED = 5.0
MIN_SPEED = 0.5
TURN_RATE = 4.0            # degrees per frame at full steer


class Car:
    def __init__(self, track):
        self.track = track
        self.reset()

    def reset(self):
        self.x, self.y, self.angle = self.track.start
        self.speed = 2.0
        self.crashed = False
        self.frames = 0
        self.index = self.track.nearest_index(self.x, self.y)
        self.progress = 0.0      # pixels driven ALONG the track (signed)
        self.best_progress = 0.0
        self.frames_since_best = 0

    @property
    def laps(self):
        return self.progress / self.track.length

    # ---------------------------------------------------------- SENSE
    def sense(self):
        """Distance from the car to the nearest wall along each of the 5 rays.

        Each ray: O + t*D (origin O, direction D, t = distance).
        Each wall segment: P1 + u*S (u from 0 to 1).
        Solving for where they meet gives t and u for EVERY ray/segment pair
        at once. A real hit needs t >= 0 and 0 <= u <= 1; we keep the
        smallest such t per ray.
        """
        seg = self.track.segments                   # (M, 4)
        x1, y1 = seg[:, 0], seg[:, 1]
        sx, sy = seg[:, 2] - x1, seg[:, 3] - y1      # segment directions (M,)

        a = math.radians(self.angle) + SENSOR_ANGLES
        dx, dy = np.cos(a)[:, None], np.sin(a)[:, None]   # (5, 1)

        denom = dx * sy - dy * sx                    # (5, M)
        qx, qy = x1 - self.x, y1 - self.y            # (M,)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (qx * sy - qy * sx) / denom
            u = (qx * dy - qy * dx) / denom
        hit = (np.abs(denom) > 1e-9) & (t >= 0) & (u >= 0) & (u <= 1)
        t = np.where(hit, t, SENSOR_RANGE)
        return np.minimum(t.min(axis=1), SENSOR_RANGE)  # (5,)

    def distance_to_wall(self):
        """Shortest distance from the car's centre to any wall segment."""
        seg = self.track.segments
        x1, y1 = seg[:, 0], seg[:, 1]
        sx, sy = seg[:, 2] - x1, seg[:, 3] - y1
        t = ((self.x - x1) * sx + (self.y - y1) * sy) / (sx * sx + sy * sy)
        t = np.clip(t, 0, 1)
        return float(np.hypot(self.x - (x1 + t * sx),
                              self.y - (y1 + t * sy)).min())

    # ---------------------------------------------------------- ACT
    def update(self, steer, throttle):
        """steer: -1 (left) .. +1 (right).  throttle: -1 .. +1."""
        if self.crashed:
            return
        steer = max(-1.0, min(1.0, float(steer)))
        throttle = max(-1.0, min(1.0, float(throttle)))
        self.angle += steer * TURN_RATE
        self.speed = max(MIN_SPEED, min(MAX_SPEED, self.speed + throttle * 0.1))
        a = math.radians(self.angle)
        self.x += math.cos(a) * self.speed
        self.y += math.sin(a) * self.speed
        self.frames += 1
        self._update_progress()
        if self.distance_to_wall() < CAR_RADIUS:
            self.crashed = True

    def _update_progress(self):
        """Move the 'progress meter' to the nearest centerline point.
        The distance between old and new points (with wrap-around at the
        finish line) is added to progress, negative if going backwards."""
        new = self.track.nearest_index(self.x, self.y)
        cum, length = self.track.cum_length, self.track.length
        delta = cum[new] - cum[self.index]
        if delta > length / 2:        # crossed the start line backwards
            delta -= length
        elif delta < -length / 2:     # crossed the start line forwards
            delta += length
        self.progress += delta
        self.index = new
        if self.progress > self.best_progress + 1e-6:
            self.best_progress = self.progress
            self.frames_since_best = 0
        else:
            self.frames_since_best += 1

    # ---------------------------------------------------------- DRAW
    def draw(self, screen, color=(70, 140, 255), readings=None):
        import pygame
        if readings is not None:
            for off, d in zip(SENSOR_ANGLES, readings):
                a = math.radians(self.angle) + off
                end = (self.x + math.cos(a) * d, self.y + math.sin(a) * d)
                pygame.draw.line(screen, (80, 200, 120), (self.x, self.y), end, 1)
                pygame.draw.circle(screen, (255, 200, 60), end, 3)
        if self.crashed:
            color = (110, 60, 60)
        pygame.draw.circle(screen, color, (self.x, self.y), CAR_RADIUS)
        a = math.radians(self.angle)
        nose = (self.x + math.cos(a) * CAR_RADIUS * 1.6,
                self.y + math.sin(a) * CAR_RADIUS * 1.6)
        pygame.draw.line(screen, (255, 255, 255), (self.x, self.y), nose, 2)


def rule_based_driver(readings):
    """Your stage-1 hand-written driver, unchanged: steer toward the side
    with more open space, slow down when something is close ahead.
    It's the baseline the evolved network has to beat (see compare.py later)."""
    far_left, left, front, right, far_right = readings
    steer = ((right + far_right) - (left + far_left)) / (2 * SENSOR_RANGE)
    steer = max(-1, min(1, steer * 3))
    throttle = 1 if front > 100 else -1
    return steer, throttle