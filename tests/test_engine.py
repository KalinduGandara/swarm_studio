import math

import numpy as np

from swarm_studio.engine import Swarm


def make(n: int, seed: int = 1, floor: float = 900.0, r: float = 1.5) -> Swarm:
    s = Swarm(80.0, 500.0, 590.0, floor, r, speed=400.0)
    s.scatter(n, np.random.default_rng(seed), (590.0, floor))
    return s


def test_balls_stay_inside_the_chamber():
    s = make(3000)
    for _ in range(400):
        s.step(1 / 240)
    r = s.radius
    assert s.x.min() >= 80 + r - 1e-9 and s.x.max() <= 500 - r + 1e-9
    assert s.y.min() >= 590 + r - 1e-9 and s.y.max() <= 900 - r + 1e-9


def test_speed_stays_near_target_and_no_clumps():
    s = make(8000)
    for _ in range(480):
        s.step(1 / 240)
    sp = np.hypot(s.vx, s.vy)
    assert abs(np.median(sp) - 400) < 20
    assert sp.max() <= 1.5 * 400 + 1e-6
    hist, _, _ = np.histogram2d(s.x, s.y, bins=(42, 31), range=((80, 500), (590, 900)))
    assert hist.max() < 6 * hist.mean()  # evenly spread, no collapsed blobs


def test_same_seed_same_state():
    a, b = make(2000, seed=7), make(2000, seed=7)
    for _ in range(200):
        a.step(1 / 240)
        b.step(1 / 240)
    assert np.array_equal(a.x, b.x) and np.array_equal(a.vy, b.vy)


def test_hit_rate_follows_gas_formula():
    # hits per ball per second ≈ speed / (pi * chamber height)
    s = make(4000, floor=1100.0, r=1.0)
    for _ in range(120):
        s.step(1 / 240)
    hits = sum(s.step(1 / 240) for _ in range(480))
    per_ball = hits / 2.0 / s.n
    expected = 400 / (math.pi * (1100 - 590))
    assert 0.8 * expected < per_ball < 1.4 * expected


def test_lone_ball_keeps_hitting_the_floor():
    s = make(1, r=14.0, floor=711.0)
    hits = sum(s.step(1 / 240) for _ in range(240 * 4))
    assert hits >= 4  # never skims between the side walls for ever


def test_multiply_adds_copies():
    s = make(100)
    s.multiply(600, np.random.default_rng(2))
    assert s.n == 700


def test_gravity_keeps_hits_per_ball_independent_of_count_and_no_heap():
    rates = []
    for n in (300, 6000):
        s = Swarm(80.0, 500.0, 590.0, 1000.0, 1.2, speed=250.0, gravity=700.0)
        s.scatter(n, np.random.default_rng(4), (590.0, 1000.0))
        for _ in range(240):
            s.step(1 / 240)
        hits = sum(s.step(1 / 240) for _ in range(480))
        rates.append(hits / 2.0 / n)
        rows = np.histogram(s.y, bins=5, range=(590.0, 1000.0))[0]
        assert rows[-1] <= rows[0]  # densest near the top of the bounce, never a heap
    assert abs(rates[0] - rates[1]) < 0.25 * rates[1]
