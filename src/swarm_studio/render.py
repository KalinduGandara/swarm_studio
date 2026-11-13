"""Draw one race frame (numpy RGB) and write PNGs; text with pygame's default font (the same
font bouncekit uses, so the Channel S look stays identical)."""

from __future__ import annotations

import math
import os
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit

from .race import BOTTOM, HEIGHT, TANK_X, TOP, TRAY, WIDTH, Race

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402
import pygame.freetype  # noqa: E402

BG = "#0b0b14"
TEXT = "#ffffff"
MUTED = "#9a9ab0"
GOLD = "#ffd43b"
WALL = 6
BAR_Y = TRAY + 22


def rgb(color: str) -> np.ndarray:
    return np.array([int(color[i : i + 2], 16) for i in (1, 3, 5)], np.float64)


def mix(a: str, b: str, t: float) -> np.ndarray:
    """a*(1-t) + b*t."""
    return rgb(a) * (1.0 - t) + rgb(b) * t


@njit(cache=True)
def _discs(img, xs, ys, r, col, flash, fcol, alpha):
    """Anti-aliased discs of radius r (serial: overlapping writes stay deterministic)."""
    h, w, _ = img.shape
    rd = max(r, 0.7)
    for i in range(xs.size):
        cx = xs[i]
        cy = ys[i]
        c0 = fcol if flash[i] else col
        x0 = max(int(cx - rd - 1), 0)
        x1 = min(int(cx + rd + 2), w)
        y0 = max(int(cy - rd - 1), 0)
        y1 = min(int(cy + rd + 2), h)
        for py in range(y0, y1):
            for px in range(x0, x1):
                d = math.sqrt((px + 0.5 - cx) ** 2 + (py + 0.5 - cy) ** 2)
                cov = rd + 0.5 - d
                if cov <= 0.0:
                    continue
                a = min(cov, 1.0) * alpha
                for ch in range(3):
                    img[py, px, ch] = img[py, px, ch] * (1.0 - a) + c0[ch] * a


@njit(cache=True)
def _glow(img, xs, ys, r, col, strength):
    """Soft halo around big balls (only used while a side has few balls)."""
    h, w, _ = img.shape
    rg = r * 2.6
    for i in range(xs.size):
        cx = xs[i]
        cy = ys[i]
        for py in range(max(int(cy - rg), 0), min(int(cy + rg + 1), h)):
            for px in range(max(int(cx - rg), 0), min(int(cx + rg + 1), w)):
                d = math.sqrt((px + 0.5 - cx) ** 2 + (py + 0.5 - cy) ** 2)
                if d >= rg or d <= r:
                    continue
                a = strength * (1.0 - (d - r) / (rg - r)) ** 2
                for ch in range(3):
                    img[py, px, ch] = img[py, px, ch] * (1.0 - a) + col[ch] * a


class TextCache:
    def __init__(self) -> None:
        if not pygame.freetype.get_init():
            pygame.freetype.init()
        self._fonts: dict[int, pygame.freetype.Font] = {}
        self._cache: dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray]] = {}

    def _font(self, size: int) -> pygame.freetype.Font:
        f = self._fonts.get(size)
        if f is None:
            f = pygame.freetype.Font(None, size)
            f.antialiased = True
            self._fonts[size] = f
        return f

    def get(self, text: str, size: int, color: str) -> tuple[np.ndarray, np.ndarray]:
        key = (text, size, color)
        hit = self._cache.get(key)
        if hit is None:
            surf, _ = self._font(size).render(text, fgcolor=pygame.Color(color))
            rgb_ = pygame.surfarray.array3d(surf).swapaxes(0, 1).astype(np.float64)
            a = pygame.surfarray.array_alpha(surf).swapaxes(0, 1).astype(np.float64) / 255.0
            hit = (rgb_, a)
            if len(self._cache) > 4000:
                self._cache.clear()
            self._cache[key] = hit
        return hit

    def draw(
        self, img: np.ndarray, text: str, center: tuple[float, float], size: int, color: str
    ) -> None:
        if not text:
            return
        col, a = self.get(text, size, color)
        h, w = a.shape
        x0 = int(round(center[0] - w / 2))
        y0 = int(round(center[1] - h / 2))
        ys, xs = max(y0, 0), max(x0, 0)
        ye, xe = min(y0 + h, img.shape[0]), min(x0 + w, img.shape[1])
        if ye <= ys or xe <= xs:
            return
        sa = a[ys - y0 : ye - y0, xs - x0 : xe - x0, None]
        sc = col[ys - y0 : ye - y0, xs - x0 : xe - x0]
        region = img[ys:ye, xs:xe]
        region *= 1.0 - sa
        region += sc * sa


def rect(
    img: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: np.ndarray,
    alpha: float = 1.0,
) -> None:
    xa, xb = max(int(round(x0)), 0), min(int(round(x1)), img.shape[1])
    ya, yb = max(int(round(y0)), 0), min(int(round(y1)), img.shape[0])
    if xb <= xa or yb <= ya:
        return
    region = img[ya:yb, xa:xb]
    region *= 1.0 - alpha
    region += color * alpha


@dataclass
class Burst:
    t0: float
    x: np.ndarray
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    color: np.ndarray


class Painter:
    """Draws frames of one race; keeps particle bursts and wall flashes between frames."""

    LIFE = 0.8

    def __init__(self, race: Race) -> None:
        self.race = race
        self.text = TextCache()
        self.bursts: list[Burst] = []
        self.flashes: list[tuple[float, int, float, str]] = []  # (t0, side, seconds, colour)
        self._seen = 0
        self.fx = np.random.default_rng(race.cfg.seed + 1_000_003)  # visual-only randomness

    def _take_events(self) -> None:
        race = self.race
        cfg = race.cfg
        for ev in race.events[self._seen :]:
            s = race.sides[ev.side]
            if ev.kind == "break":
                x0, x1 = TANK_X[ev.side]
                n = 90
                a = self.fx.uniform(0, 2 * math.pi, n)
                sp = self.fx.uniform(150, 650, n)
                self.bursts.append(
                    Burst(
                        ev.t,
                        self.fx.uniform(x0, x1, n),
                        np.full(n, cfg.floor_y(ev.floor)),
                        sp * np.cos(a),
                        sp * np.sin(a) - 200,
                        rgb(s.team.color),
                    )
                )
            elif ev.kind == "turn":
                self.flashes.append((ev.t, ev.side, 0.6, TEXT))
            elif ev.kind == "lead":
                self.flashes.append((ev.t, ev.side, 0.5, GOLD))
            elif ev.kind == "win":
                self.flashes.append((ev.t, ev.side, 1.2, GOLD))
        self._seen = len(race.events)

    def headline(self) -> str:
        race = self.race
        if race.finished:
            return ""
        if race.turn_t is not None:
            return f"{race.sides[race.underdog].team.name} HAS MORE BALLS!"
        return race.cfg.hook

    def frame(self, clear_live: bool = False) -> np.ndarray:
        race = self.race
        cfg = race.cfg
        t = race.t
        self._take_events()
        img = np.empty((HEIGHT, WIDTH, 3), np.float64)
        img[:] = rgb(BG)
        done = race.finished

        for i, s in enumerate(race.sides):
            x0, x1 = TANK_X[i]
            col = s.team.color
            wall = rgb(col)
            for t0, side, dur, fc in self.flashes:
                if side == i and t0 <= t < t0 + dur:
                    wall = mix(col, fc, 0.85 * (1.0 - (t - t0) / dur))
            # walls, lid, tray
            rect(img, x0 - WALL, TOP - WALL, x0, TRAY + WALL, wall)
            rect(img, x1, TOP - WALL, x1 + WALL, TRAY + WALL, wall)
            rect(img, x0 - WALL, TOP - WALL, x1 + WALL, TOP, wall)
            rect(img, x0 - WALL, TRAY, x1 + WALL, TRAY + WALL, wall)
            # floors still standing: solid lines; the active one thins and dims as HP falls
            for k in range(s.floor, cfg.floors):
                y = cfg.floor_y(k)
                if k == s.floor and not done:
                    frac = max(s.hp, 0.0) / cfg.hp[k]
                    th = 2.0 + 5.0 * frac
                    c = mix(BG, col, 0.35 + 0.65 * frac)
                else:
                    th = 7.0
                    c = rgb(col)
                rect(img, x0, y - th / 2, x1, y + th / 2, c)
            # balls
            sw = s.swarm
            if sw.n:
                c = mix(BG, col, 0.3) if done else rgb(col)
                if sw.radius >= 4.0 and not done:
                    _glow(img, sw.x, sw.y, sw.radius, rgb(col), 0.35)
                flash = s.flash if (s.flash.size == sw.n and not done) else np.zeros(sw.n, np.int8)
                _discs(img, sw.x, sw.y, sw.radius, c, flash, rgb(TEXT), 1.0)

        # particles
        keep = []
        for b in self.bursts:
            age = t - b.t0
            if age < 0 or age > self.LIFE:
                continue
            keep.append(b)
            a = 1.0 - age / self.LIFE
            px = b.x + b.vx * age
            py = b.y + b.vy * age + 0.5 * 900 * age * age
            for xx, yy in zip(px, py, strict=True):
                rect(img, xx - 2, yy - 2, xx + 2, yy + 2, b.color, a)
        self.bursts = keep

        # labels below every standing floor (in the empty space under the swarm)
        for i, s in enumerate(race.sides):
            x0, x1 = TANK_X[i]
            cx = (x0 + x1) / 2
            for k in range(s.floor, cfg.floors):
                if done or clear_live:
                    break
                value = s.hp if k == s.floor else cfg.hp[k]
                y = cfg.floor_y(k) + (20 if k < cfg.floors - 1 else 22)
                self.text.draw(img, f"{math.ceil(max(value, 0)):,}", (cx, y), 30, TEXT)

        # gauges: one segment per floor still standing; the last drains with the active HP
        for i, s in enumerate(race.sides):
            x0, x1 = TANK_X[i]
            seg = (x1 - x0) / cfg.floors
            for k in range(cfg.floors):
                rect(
                    img,
                    x0 + k * seg + 2,
                    BAR_Y - 7,
                    x0 + (k + 1) * seg - 2,
                    BAR_Y + 7,
                    mix(BG, s.team.color, 0.25),
                )
            left = cfg.floors - s.floor
            for j in range(left):
                a0 = x0 + j * seg + 2
                a1 = x0 + (j + 1) * seg - 2
                if j == left - 1:
                    a1 = a0 + (a1 - a0) * max(s.hp, 0.0) / cfg.hp[s.floor]
                rect(img, a0, BAR_Y - 7, a1, BAR_Y + 7, rgb(s.team.color))

        # header and labels
        tx = self.text
        tx.draw(img, cfg.tag.upper(), (540, 270), 38, MUTED)
        if not clear_live:
            tx.draw(img, self.headline(), (540, 345), 66, TEXT)
        tx.draw(img, cfg.rule_line, (540, 412), 38, TEXT)
        tx.draw(img, cfg.win_line, (540, 456), 38, TEXT)
        for i, s in enumerate(race.sides):
            x0, x1 = TANK_X[i]
            cx = (x0 + x1) / 2
            if clear_live:
                tx.draw(img, s.team.rule_text(), (cx, 1610), 40, s.team.color)
                continue
            if race.leader == i and not done:
                tx.draw(img, "LEADER", (cx, 508), 28, GOLD)
            tx.draw(img, f"{s.count:,}", (cx, 548), 50, TEXT)
            under = (
                "BOTTOM!"
                if s.floor >= cfg.floors
                else f"FLOOR {s.floor + 1}/{cfg.floors} · {math.ceil(max(s.hp, 0)):,} HP"
            )
            tx.draw(img, under, (cx, 1562), 34, TEXT)
            tx.draw(img, s.team.rule_text(), (cx, 1610), 40, s.team.color)
        if done and not clear_live:
            w = race.sides[race.winner]
            lo = race.sides[1 - race.winner]
            tx.draw(img, f"{w.team.name} WINS!", (540, 960), 140, GOLD)
            short = math.ceil(max(lo.hp, 0.0)) + sum(cfg.hp[lo.floor + 1 :])
            tx.draw(img, f"{lo.team.name} WAS {short:,} HP SHORT", (540, 1080), 50, TEXT)
        return np.clip(img, 0, 255).astype(np.uint8)


def write_png(path: Path, rgb_: np.ndarray) -> None:
    """Minimal dependency-free RGB PNG writer."""
    h, w, _ = rgb_.shape
    raw = b"".join(b"\x00" + rgb_[r].tobytes() for r in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6))
    png += chunk(b"IEND", b"")
    Path(path).write_bytes(png)


_ = BOTTOM  # layout constant re-exported for callers
