"""The growth race: two tanks of floors with HP; every floor hit takes 1 HP.

`Race.advance()` runs one video frame of physics (several substeps) and updates the race state
(HP, breaks, growth, turning point, leader, winner). It never draws; `render.py` reads it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .engine import Swarm

# 1080x1920 layout (Channel S brand: action between y 520 and 1450, text inside y 250-1650)
WIDTH, HEIGHT = 1080, 1920
TOP, BOTTOM = 590.0, 1440.0
TRAY = BOTTOM + 50.0  # catch tray under the bottom floor (the winner's balls land here)
TANK_X = ((80.0, 500.0), (580.0, 1000.0))
R_MIN, R_MAX = 0.6, 14.0


@dataclass(frozen=True)
class Team:
    name: str
    start: int
    rule: str  # "mul:N" | "add:N" | "pow:N"
    color: str  # "#rrggbb"

    @classmethod
    def parse(cls, text: str, color: str) -> Team:
        """``NAME:START:KIND:N`` e.g. ``ONE:1:mul:7``."""
        name, start, kind, n = text.split(":")
        team = cls(name.upper(), int(start), f"{kind}:{n}", color)
        grow(team.rule, 1)  # validates the rule
        return team

    def rule_text(self) -> str:
        kind, n = self.rule.split(":")
        power = "SQUARED" if n == "2" else f"^{n}"
        labels = {"mul": f"×{n}", "add": f"+{int(n):,}", "pow": power}
        return labels[kind] + " EVERY FLOOR"


def grow(rule: str, count: int) -> int:
    kind, n = rule.split(":")
    k = int(n)
    if kind == "mul":
        return count * k
    if kind == "add":
        return count + k
    if kind == "pow":
        return count**k
    raise ValueError(f"unknown rule {rule!r} (use mul:N, add:N or pow:N)")


@dataclass
class Config:
    left: Team
    right: Team
    hp: tuple[int, ...]
    seed: int = 1
    fps: int = 60
    substeps: int = 4
    gate: float = 0.5  # seconds before hits count
    hold: float = 2.2  # seconds on the result
    loop_fade: float = 0.4
    max_seconds: float = 90.0
    speed: float = 150.0  # speed at the top of a bounce (px/s); faster lower down
    gravity: float = 700.0  # px/s²
    fill: float = 0.3  # share of the chamber area the balls cover (sets the ball radius)
    tag: str = "One vs Many"
    hook: str = ""
    rule_line: str = "Every hit takes 1 HP"
    win_line: str = "First to the bottom wins"

    @property
    def floors(self) -> int:
        return len(self.hp)

    def floor_y(self, k: int) -> float:
        """y of floor k (0 = top floor); k == floors means the tray."""
        if k >= self.floors:
            return TRAY
        return TOP + (k + 1) * (BOTTOM - TOP) / self.floors


@dataclass
class Side:
    team: Team
    swarm: Swarm
    count: int
    floor: int = 0
    hp: float = 0.0
    armed: bool = False  # hits on a new floor count once its first ball lands there
    flash: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int8))
    hits_frame: int = 0
    breaks: list[float] = field(default_factory=list)

    def mark_hits(self, hit: np.ndarray) -> None:
        """OR this substep's hits into the frame's flash mask (balls may have been added)."""
        if self.flash.size < hit.size:
            self.flash = np.concatenate([self.flash, np.zeros(hit.size - self.flash.size, np.int8)])
        self.flash[: hit.size] |= hit


@dataclass
class Event:
    t: float
    kind: str  # "break" | "turn" | "lead" | "win"
    side: int
    floor: int = 0


class Race:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.frame = 0
        self.t = 0.0
        self.sides: list[Side] = []
        for i, team in enumerate((cfg.left, cfg.right)):
            x0, x1 = TANK_X[i]
            floor = cfg.floor_y(0)
            sw = Swarm(
                x0,
                x1,
                TOP,
                floor,
                self.radius(team.start, 0),
                speed=cfg.speed,
                gravity=cfg.gravity,
            )
            sw.scatter(team.start, self.rng, (TOP, floor))
            self.sides.append(Side(team, sw, team.start, hp=float(cfg.hp[0])))
        self.events: list[Event] = []
        self.winner: int | None = None
        self.win_t: float | None = None
        self.leader = 0 if cfg.left.start >= cfg.right.start else 1
        self.underdog = 1 - self.leader
        self.turn_t: float | None = None
        self.overtake_t: float | None = None

    # -- geometry ----------------------------------------------------------------------
    def radius(self, count: int, floor: int) -> float:
        x0, x1 = TANK_X[0]
        area = (x1 - x0) * (self.cfg.floor_y(floor) - TOP)
        r = math.sqrt(self.cfg.fill * area / (math.pi * max(count, 1)))
        return min(R_MAX, max(R_MIN, r))

    def progress(self, i: int) -> float:
        s = self.sides[i]
        if s.floor >= self.cfg.floors:
            return float(self.cfg.floors)
        return s.floor + 1.0 - max(s.hp, 0.0) / self.cfg.hp[s.floor]

    @property
    def finished(self) -> bool:
        return self.winner is not None

    @property
    def end_t(self) -> float | None:
        return None if self.win_t is None else self.win_t + self.cfg.hold

    # -- simulation ---------------------------------------------------------------------
    def _break(self, i: int) -> None:
        s = self.sides[i]
        cfg = self.cfg
        k = s.floor
        s.breaks.append(round(self.t, 3))
        self.events.append(Event(self.t, "break", i, k))
        s.floor = k + 1
        s.armed = False
        s.swarm.floor = cfg.floor_y(s.floor)
        if s.floor >= cfg.floors:
            s.hp = 0.0
            if self.winner is None:
                self.winner = i
                self.win_t = self.t
                self.events.append(Event(self.t, "win", i, k))
            return
        new = grow(s.team.rule, s.count)
        if new > 400_000:
            raise ValueError(f"{s.team.name} would reach {new:,} balls; keep counts under ~70,000")
        s.swarm.multiply(new - s.count, self.rng)
        s.count = new
        s.swarm.radius = self.radius(new, s.floor)
        s.hp += cfg.hp[s.floor]  # leftover damage carries to the next floor

    def advance(self) -> None:
        """One video frame: substeps of physics, hits → HP, breaks, turn, leader."""
        cfg = self.cfg
        dt = 1.0 / (cfg.fps * cfg.substeps)
        for s in self.sides:
            s.flash = np.zeros(s.swarm.n, np.int8)
            s.hits_frame = 0
        for _ in range(cfg.substeps):
            for i, s in enumerate(self.sides):
                hits = s.swarm.step(dt)
                s.mark_hits(s.swarm.last_hit)
                if hits:
                    s.armed = True
                if self.winner is not None or self.t < cfg.gate or s.floor >= cfg.floors:
                    continue
                if not s.armed:
                    continue
                s.hp -= hits
                s.hits_frame += hits
                while s.hp <= 0 and s.floor < cfg.floors and self.winner is None:
                    self._break(i)
            self.t += dt
        self.frame += 1
        self.t = self.frame / cfg.fps  # no float drift
        if self.winner is None:
            u, o = self.underdog, 1 - self.underdog
            if self.turn_t is None and self.sides[u].count > self.sides[o].count:
                self.turn_t = self.t
                self.events.append(Event(self.t, "turn", u))
            lead = 0 if self.progress(0) >= self.progress(1) else 1
            if lead != self.leader:
                self.leader = lead
                if lead == self.underdog and self.overtake_t is None:
                    self.overtake_t = self.t
                self.events.append(Event(self.t, "lead", lead))

    def summary(self) -> dict:
        loser = None if self.winner is None else 1 - self.winner
        out = {
            "winner": None if self.winner is None else self.sides[self.winner].team.name,
            "win_t": self.win_t,
            "turn_t": self.turn_t,
            "overtake_t": self.overtake_t,
            "breaks": {s.team.name: s.breaks for s in self.sides},
            "counts": {s.team.name: s.count for s in self.sides},
        }
        if loser is not None:
            ls = self.sides[loser]
            out["loser_floor"] = ls.floor + 1
            out["loser_hp_left"] = math.ceil(max(ls.hp, 0.0))
        return out
