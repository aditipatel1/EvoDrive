"""
check_setup.py - verify the whole EvoDrive project works on this machine.

Run:  python3 check_setup.py
Every check prints PASS or FAIL. Paste the full output if anything fails.
Drawing is tested "headlessly" (no window opens), so this works anywhere.
"""
import os
import sys
import tempfile
import traceback

os.environ["SDL_VIDEODRIVER"] = "dummy"     # draw to memory, not a window
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

results = []


def check(name):
    def wrap(fn):
        try:
            detail = fn()
            results.append(True)
            print(f"PASS  {name}" + (f"  ({detail})" if detail else ""))
        except Exception as e:
            results.append(False)
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
        return fn
    return wrap


@check("Python 3.9 or newer")
def _():
    assert sys.version_info >= (3, 9), sys.version
    return sys.version.split()[0]


@check("NumPy and Pygame installed")
def _():
    import numpy, pygame
    return f"numpy {numpy.__version__}, pygame {pygame.version.ver}"


@check("network.py: 50 parameters, DNA round-trip")
def _():
    import numpy as np
    from network import NeuralNetwork
    a, b = NeuralNetwork(), NeuralNetwork()
    assert a.num_parameters() == 50, a.num_parameters()
    b.set_vector(a.to_vector())
    x = np.array([.2, .4, .6, .8, 1])
    assert np.allclose(a.forward(x), b.forward(x))


@check("track.py: valid tracks, same seed -> same track")
def _():
    import numpy as np
    from track import generate_track, min_turn_radius
    t1 = generate_track(np.random.default_rng(12345))
    t2 = generate_track(np.random.default_rng(12345))
    assert np.allclose(t1.center, t2.center)
    rng = np.random.default_rng(0)
    for _ in range(50):
        t = generate_track(rng)
        assert min_turn_radius(t.center) > 1.5 * t.half_width
    return f"50 tracks ok, lap length ~{t1.length:.0f} px"


@check("car.py: sensors, crash detection, lap counting")
def _():
    import numpy as np
    from track import generate_track
    from car import Car, SENSOR_RANGE, rule_based_driver
    track = generate_track(np.random.default_rng(0))
    car = Car(track)
    r = car.sense()
    assert r.shape == (5,) and (r > 0).all() and (r <= SENSOR_RANGE).all()
    car.angle += 90                      # point at a wall and floor it
    for _ in range(100):
        car.update(0, 1)
    assert car.crashed, "car drove through a wall"
    car = Car(track)
    for _ in range(2000):
        if car.crashed or car.laps >= 1:
            break
        car.update(*rule_based_driver(car.sense()))
    assert car.laps > 0.1, f"rule-based driver made no progress ({car.laps:.2f})"
    return f"rule-based driver reached {car.laps:.2f} laps on test track"


@check("evolution.py: elitism, mutation, crossover")
def _():
    import numpy as np
    from evolution import next_generation, mutate, crossover
    rng = np.random.default_rng(0)
    dnas = [rng.normal(size=50) for _ in range(20)]
    fit = np.arange(20, dtype=float)
    new = next_generation(dnas, fit, rng)
    assert len(new) == 20
    assert np.allclose(new[0], dnas[19]), "best driver was not kept"
    changed = np.mean(mutate(np.zeros(5000), rng) != 0)
    assert 0.07 < changed < 0.13, changed
    c = crossover(np.zeros(50), np.ones(50), rng)
    assert set(np.unique(c)) <= {0.0, 1.0}
    return f"mutation changed {changed*100:.0f}% of numbers"


@check("train.py: 2 tiny generations, files written")
def _():
    import train
    with tempfile.TemporaryDirectory() as d:
        train.train(generations=2, seed=0, population=8, out=d)
        assert os.path.exists(os.path.join(d, "best_network.npy"))
        lines = open(os.path.join(d, "history.csv")).read().strip().splitlines()
        assert lines[0] == "generation,train_best,train_mean,val_laps,val_success"
        assert len(lines) == 3


@check("drawing (headless): train.py --window, drive_best.py")
def _():
    import numpy as np
    import pygame
    import train, drive_best
    from network import NeuralNetwork
    from track import generate_track
    track = generate_track(np.random.default_rng(1))
    cb = train.make_window_callback(track, 1)
    train.simulate(track, [train.network_driver(NeuralNetwork())], 30, on_frame=cb)
    pygame.quit()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "net.npy")
        np.save(path, NeuralNetwork().to_vector())
        drive_best.main(path, max_frames=30)


@check("early files: main.py and demo_random.py")
def _():
    import py_compile
    if os.path.exists("stage1.py") and not os.path.exists("main.py"):
        raise FileNotFoundError("rename stage1.py to main.py "
                                "(demo_random.py imports 'main')")
    for f in ("main.py", "demo_random.py"):
        if os.path.exists(f):
            py_compile.compile(f, doraise=True)
    return "both compile"


print(f"\n{sum(results)}/{len(results)} checks passed")
if all(results):
    print("All good. Next: python3 train.py")