"""Deterministic 2D swarm engine: N equal discs in one tank chamber (numba).

The chamber is the box ``x0..x1`` × ``top..floor``. Balls fall under gravity and bounce off
the walls, the lid, the floor and each other. Every step each ball is nudged back toward the
speed an energy-conserving bouncing ball would have at its height (``speed`` at the lid), so
collisions never drain the swarm into a heap: every ball keeps travelling lid to floor, and hits
per second on the floor grow with the number of balls.

Determinism: the grid is built serially (counting sort); the parallel kernels only write the
entries of the ball they process (Jacobi style), reading the previous state, so the result does
not depend on thread scheduling. Wall bounces add a small deterministic angle jitter from a hash
of (ball index, step) so a lone ball never locks into one repeating path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numba import njit, prange


@njit(cache=True, inline="always")
def _hash01(i: int, step: int) -> float:
    """Deterministic pseudo-random number in [0, 1) from two integers (31-bit integer hash;
    every product stays inside int64)."""
    z = ((i * 73856093) ^ (step * 19349663)) & 0x7FFFFFFF
    z = (((z >> 16) ^ z) * 0x45D9F3B) & 0x7FFFFFFF
    z = (((z >> 16) ^ z) * 0x45D9F3B) & 0x7FFFFFFF
    z = (z >> 16) ^ z
    return z / 2147483648.0


@njit(cache=True)
def _build_grid(x, y, x0, top, cell, gw, gh, cell_of, start, order):
    n = x.size
    counts = np.zeros(gw * gh + 1, np.int64)
    for i in range(n):
        cx = min(max(int((x[i] - x0) / cell), 0), gw - 1)
        cy = min(max(int((y[i] - top) / cell), 0), gh - 1)
        c = cy * gw + cx
        cell_of[i] = c
        counts[c + 1] += 1
    for c in range(gw * gh):
        counts[c + 1] += counts[c]
    for c in range(gw * gh + 1):
        start[c] = counts[c]
    fill = counts[:-1].copy()
    for i in range(n):
        c = cell_of[i]
        order[fill[c]] = i
        fill[c] += 1


@njit(cache=True, parallel=True)
def _collide(x, y, vx, vy, r, gw, gh, cell_of, start, order, exch, nx_out, ny_out, nvx, nvy):
    """Jacobi pass: each ball's position push and velocity change from its overlaps."""
    n = x.size
    rr = 2.0 * r
    rr2 = rr * rr
    for i in prange(n):
        px = x[i]
        py = y[i]
        pvx = vx[i]
        pvy = vy[i]
        dxs = 0.0
        dys = 0.0
        dvx = 0.0
        dvy = 0.0
        m = 0
        c = cell_of[i]
        cx = c % gw
        cy = c // gw
        for oy in range(-1, 2):
            yy = cy + oy
            if yy < 0 or yy >= gh:
                continue
            for ox in range(-1, 2):
                xx = cx + ox
                if xx < 0 or xx >= gw:
                    continue
                cc = yy * gw + xx
                for k in range(start[cc], start[cc + 1]):
                    j = order[k]
                    if j == i:
                        continue
                    dx = x[j] - px
                    dy = y[j] - py
                    d2 = dx * dx + dy * dy
                    if d2 >= rr2:
                        continue
                    if d2 < 1e-12:
                        # exactly on top of each other: split along a fixed per-pair direction
                        ii = np.int64(i)
                        a = 6.283185307179586 * _hash01(min(ii, j), max(ii, j))
                        ux = math.cos(a)
                        uy = math.sin(a)
                        if ii > j:
                            ux = -ux
                            uy = -uy
                        d = 0.0
                    else:
                        d = math.sqrt(d2)
                        ux = dx / d
                        uy = dy / d
                    over = rr - d
                    dxs -= 0.5 * over * ux
                    dys -= 0.5 * over * uy
                    rel = (vx[j] - pvx) * ux + (vy[j] - pvy) * uy
                    if rel < 0.0:  # approaching: equal masses swap normal velocity
                        dvx += rel * ux
                        dvy += rel * uy
                        m += 1
        # limit the push so dense spawns relax over a few steps instead of exploding
        push = math.sqrt(dxs * dxs + dys * dys)
        lim = r
        if push > lim:
            dxs *= lim / push
            dys *= lim / push
        nx_out[i] = px + dxs
        ny_out[i] = py + dys
        if m > 1:  # several contacts at once: average, so the Jacobi sum never amplifies
            dvx /= m
            dvy /= m
        nvx[i] = pvx + exch * dvx
        nvy[i] = pvy + exch * dvy


@njit(cache=True, parallel=True)
def _move(
    x, y, vx, vy, r, x0, x1, top, floor, g, speed, relax, jitter, min_hit, min_vy, dt, step, hit
):
    """Gravity, speed relaxation, integration and wall/floor bounces. hit[i]=1 on a floor hit."""
    n = x.size
    for i in prange(n):
        hit[i] = 0
        # energy-conserving target speed: `speed` at the lid, faster lower down (a bouncing
        # ball under gravity), so balls fall and bounce but never settle into a heap
        tgt = math.sqrt(speed * speed + 2.0 * g * max(y[i] - top, 0.0))
        pvx = vx[i]
        pvy = vy[i] + g * dt
        s = math.sqrt(pvx * pvx + pvy * pvy)
        if s > 1.5 * tgt:  # hard cap: no ball runs away
            pvx *= 1.5 * tgt / s
            pvy *= 1.5 * tgt / s
            s = 1.5 * tgt
        if s > 1e-9:
            f = 1.0 + relax * (tgt / s - 1.0)
            pvx *= f
            pvy *= f
        else:
            a = 6.283185307179586 * _hash01(np.int64(i), np.int64(step))
            pvx = tgt * math.cos(a)
            pvy = tgt * math.sin(a)
        px = x[i] + pvx * dt
        py = y[i] + pvy * dt
        bounced = False
        if px < x0 + r:
            px = x0 + r
            pvx = abs(pvx)
            bounced = True
        elif px > x1 - r:
            px = x1 - r
            pvx = -abs(pvx)
            bounced = True
        if py < top + r:
            py = top + r
            pvy = abs(pvy)
            bounced = True
        elif py > floor - r:
            py = floor - r
            if pvy > 0.0:
                if pvy > min_hit * tgt:
                    hit[i] = 1
                pvy = -pvy
            bounced = True
        if bounced and jitter > 0.0:
            t = (2.0 * _hash01(np.int64(i), np.int64(step) + 7919) - 1.0) * jitter
            c = math.cos(t)
            sn = math.sin(t)
            qx = pvx * c - pvy * sn
            qy = pvx * sn + pvy * c
            # keep the bounce direction (never turn back into the wall)
            same_x = (qx > 0.0) == (pvx > 0.0) or abs(pvx) < 1e-9
            same_y = (qy > 0.0) == (pvy > 0.0) or abs(pvy) < 1e-9
            if same_x and same_y:
                pvx = qx
                pvy = qy
        # never skim: keep at least `min_vy` of the speed vertical, so no ball (a lone one
        # especially) can bounce between the side walls for ever without reaching the floor
        sp = math.sqrt(pvx * pvx + pvy * pvy)
        if sp > 1e-9 and abs(pvy) < min_vy * sp:
            sy = 1.0 if pvy >= 0.0 else -1.0
            pvy = sy * min_vy * sp
            pvx = math.copysign(math.sqrt(max(sp * sp - pvy * pvy, 0.0)), pvx)
        x[i] = px
        y[i] = py
        vx[i] = pvx
        vy[i] = pvy


@dataclass
class Swarm:
    """All balls of one tank. Positions in screen pixels (+y down)."""

    x0: float
    x1: float
    top: float
    floor: float
    radius: float
    speed: float = 520.0
    gravity: float = 0.0  # px/s²; the target speed grows with depth, so no heap forms
    exchange: float = 0.0  # share of the elastic velocity swap (1 = billiards; >0 clumps)
    min_vy: float = 0.35  # sin of the flattest allowed angle (~20°)
    min_hit: float = 0.15  # a floor contact counts as a hit only above this share of `speed`
    relax: float = 0.05  # share of the gap to the target speed closed per step
    jitter: float = math.radians(10.0)  # max random turn per wall bounce
    x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    vx: np.ndarray = field(default_factory=lambda: np.zeros(0))
    vy: np.ndarray = field(default_factory=lambda: np.zeros(0))
    steps: int = 0

    @property
    def n(self) -> int:
        return int(self.x.size)

    def add(self, x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray) -> None:
        r = self.radius
        x = np.clip(x, self.x0 + r, self.x1 - r)
        y = np.clip(y, self.top + r, self.floor - r)
        self.x = np.concatenate([self.x, x])
        self.y = np.concatenate([self.y, y])
        self.vx = np.concatenate([self.vx, vx])
        self.vy = np.concatenate([self.vy, vy])

    def scatter(self, n: int, rng: np.random.Generator, y_band: tuple[float, float]) -> None:
        """Add ``n`` balls at random spots in the band ``y_band``, moving in random directions."""
        r = self.radius
        x = rng.uniform(self.x0 + r, self.x1 - r, n)
        y = rng.uniform(max(y_band[0], self.top + r), min(y_band[1], self.floor - r), n)
        a = rng.uniform(0.0, 2.0 * math.pi, n)
        self.add(x, y, self.speed * np.cos(a), self.speed * np.sin(a))

    def multiply(self, extra: int, rng: np.random.Generator) -> None:
        """Add ``extra`` copies of existing balls (cycling through them), jittered by ~1 radius."""
        if extra <= 0 or self.n == 0:
            return
        src = np.arange(extra) % self.n
        src = rng.permutation(self.n)[src] if extra < self.n else src
        r = max(self.radius, 1.0)
        x = self.x[src] + rng.uniform(-r, r, extra)
        y = self.y[src] + rng.uniform(-r, r, extra)
        a = rng.uniform(0.0, 2.0 * math.pi, extra)
        self.add(x, y, self.speed * np.cos(a), self.speed * np.sin(a))

    def step(self, dt: float, collide: bool = True) -> int:
        """Advance one step; return the number of floor hits in it."""
        n = self.n
        hit = np.zeros(n, np.int8)
        if n == 0:
            self.steps += 1
            return 0
        if collide and n > 1:
            cell = max(2.0 * self.radius, 1.0)
            gw = int((self.x1 - self.x0) / cell) + 1
            gh = int((self.floor - self.top) / cell) + 1
            cell_of = np.empty(n, np.int64)
            start = np.empty(gw * gh + 1, np.int64)
            order = np.empty(n, np.int64)
            _build_grid(self.x, self.y, self.x0, self.top, cell, gw, gh, cell_of, start, order)
            nx_ = np.empty(n)
            ny_ = np.empty(n)
            nvx = np.empty(n)
            nvy = np.empty(n)
            _collide(
                self.x,
                self.y,
                self.vx,
                self.vy,
                self.radius,
                gw,
                gh,
                cell_of,
                start,
                order,
                self.exchange,
                nx_,
                ny_,
                nvx,
                nvy,
            )
            self.x, self.y, self.vx, self.vy = nx_, ny_, nvx, nvy
        _move(
            self.x,
            self.y,
            self.vx,
            self.vy,
            self.radius,
            self.x0,
            self.x1,
            self.top,
            self.floor,
            self.gravity,
            self.speed,
            self.relax,
            self.jitter,
            self.min_hit,
            self.min_vy,
            dt,
            self.steps,
            hit,
        )
        self.steps += 1
        self.last_hit = hit
        return int(hit.sum())
