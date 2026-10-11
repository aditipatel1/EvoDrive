"""
track.py - random closed-loop race tracks for EvoDrive.

How a track is built
--------------------
1. Centerline: walk once around a circle, but let the radius "wobble":
       r(theta) = R * (1 + sum_k a_k * sin(k*theta + phase_k))
   A few random sine waves (k = 2..5) bend the circle into a random smooth
   loop. Because r is always positive and we only go around once, the
   centerline can never cross itself.
2. Walls: push every centerline point sideways (along its "normal", the
   direction perpendicular to the road) by +half_width and -half_width.
   That gives an outer wall and an inner wall.
3. Validity check: on a very tight bend, the inner wall would fold over
   itself (like a pinched hose). That happens when the bend's radius of
   curvature is smaller than half the road width. We reject such tracks,
   and tracks that don't fit on screen, and simply try again.

The centerline is also the car's "progress meter": the nearest centerline
point tells us how far around the loop the car has driven.

Only NumPy is needed. Pygame is imported inside draw() only, so headless
training doesn't need a screen.
"""
import numpy as np

WIDTH, HEIGHT = 1000, 700


class Track:
    def __init__(self, centerline, half_width):
        self.half_width = half_width
        self.center = centerline                          # (N, 2) points
        nxt = np.roll(centerline, -1, axis=0)
        prv = np.roll(centerline, 1, axis=0)

        # Direction of travel at each point (unit tangent), then turn it
        # 90 degrees to get the sideways direction (unit normal).
        tangent = nxt - prv
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)

        self.wall_a = centerline + normal * half_width     # one side
        self.wall_b = centerline - normal * half_width     # other side

        # Every wall as a list of segments, one row = (x1, y1, x2, y2).
        # A single NumPy array lets car.py test all walls at once.
        segs = []
        for wall in (self.wall_a, self.wall_b):
            segs.append(np.hstack([wall, np.roll(wall, -1, axis=0)]))
        self.segments = np.vstack(segs)                   # (2N, 4)

        # Cumulative distance along the centerline, for measuring progress.
        step = np.linalg.norm(nxt - centerline, axis=1)
        self.cum_length = np.concatenate([[0.0], np.cumsum(step)[:-1]])
        self.length = step.sum()                          # one full lap

        # Start at point 0, facing toward point 1.
        d = centerline[1] - centerline[0]
        self.start = (float(centerline[0, 0]), float(centerline[0, 1]),
                      float(np.degrees(np.arctan2(d[1], d[0]))))

    def nearest_index(self, x, y):
        """Index of the centerline point closest to (x, y)."""
        d2 = (self.center[:, 0] - x) ** 2 + (self.center[:, 1] - y) ** 2
        return int(np.argmin(d2))

    def draw(self, screen, color=(220, 220, 220)):
        import pygame
        for wall in (self.wall_a, self.wall_b):
            pygame.draw.lines(screen, color, True, wall.tolist(), 3)
        sx, sy, _ = self.start
        pygame.draw.circle(screen, (90, 200, 90), (int(sx), int(sy)), 5)


def min_turn_radius(points):
    """Smallest radius of curvature along a closed loop of points.
    Uses the circle through each point and its two neighbours:
    R = (a*b*c) / (4 * triangle_area)."""
    p0, p1, p2 = np.roll(points, 1, axis=0), points, np.roll(points, -1, axis=0)
    a = np.linalg.norm(p1 - p0, axis=1)
    b = np.linalg.norm(p2 - p1, axis=1)
    c = np.linalg.norm(p2 - p0, axis=1)
    cross = np.abs((p1 - p0)[:, 0] * (p2 - p0)[:, 1]
                   - (p1 - p0)[:, 1] * (p2 - p0)[:, 0])
    area = cross / 2
    with np.errstate(divide="ignore"):
        radius = np.where(area > 1e-9, a * b * c / (4 * area), np.inf)
    return float(radius.min())


def is_valid(centerline, half_width, margin=20):
    walls_fit = (centerline.min(axis=0) > half_width + margin).all() and \
                (centerline[:, 0].max() < WIDTH - half_width - margin) and \
                (centerline[:, 1].max() < HEIGHT - half_width - margin)
    no_pinch = min_turn_radius(centerline) > 1.5 * half_width
    return bool(walls_fit and no_pinch)

def generate_track(rng, half_width=22, wobble=3.0, n_points=120,
                   max_tries=1000):
    """Make one random valid track.

    rng        : np.random.Generator, so results are reproducible by seed
    half_width : half the road width in pixels (smaller = harder)
    wobble     : how bendy the loop is (0 = perfect circle, bigger = harder)
    """
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    for _ in range(max_tries):
        radius = np.ones(n_points)
        for k in range(2, 6):
            amp = rng.uniform(0, 0.06) * wobble
            radius += amp * np.sin(k * theta + rng.uniform(0, 2 * np.pi))
        rx, ry = rng.uniform(300, 420), rng.uniform(200, 290)  # oval size
        cx, cy = WIDTH / 2, HEIGHT / 2
        center = np.stack([cx + rx * radius * np.cos(theta),
                           cy + ry * radius * np.sin(theta)], axis=1)
        if radius.min() > 0.2 and is_valid(center, half_width):
            return Track(center, half_width)
    raise RuntimeError("Could not make a valid track; lower 'wobble' "
                       "or 'half_width'.")


if __name__ == "__main__":
    # Quick self-test:  python3 track.py
    rng = np.random.default_rng(0)
    for i in range(5):
        t = generate_track(rng)
        print(f"track {i}: lap length {t.length:6.0f} px, "
              f"tightest bend radius {min_turn_radius(t.center):5.1f} px, "
              f"{len(t.segments)} wall segments")