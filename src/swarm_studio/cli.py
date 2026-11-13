"""swarm CLI: race (render + export pack), plan (physics only), frames (stills), list-presets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .race import Config, Team
from .studio import run

BLUE, ORANGE = "#4d7cff", "#ff9f1c"  # brand.TEAMS blue / orange (One vs Many colours)

# The first three videos (user, 2026-11-13). HP is tuned with `swarm plan` so the underdog
# wins late and close; re-tune after changing the engine.
PRESETS: dict[str, dict] = {
    "x7": {
        "left": "MANY:500:mul:2",
        "right": "ONE:1:mul:7",
        "hp": "5,23,72,390,2600,11700,350000",
        "hook": "Can 1 ball ×7 beat 500?",
        "tag": "One vs Many · #2",
    },
    "add": {
        "left": "PLUS:1000:add:1000",
        "right": "TIMES:1:mul:6",
        "hp": "5,12,43,180,780,3300,23500",
        "hook": "+1,000 a floor or ×6?",
        "tag": "One vs Many · #3",
    },
    "square": {
        "left": "MANY:1000:mul:2",
        "right": "SQUARE:2:pow:2",
        "hp": "6,8,17,180,68000",
        "hook": "Can 2 balls squared beat 1,000?",
        "tag": "One vs Many · #4",
    },
}


def _config(a: argparse.Namespace) -> Config:
    p = dict(PRESETS[a.preset]) if a.preset else {}
    for key in ("left", "right", "hp", "hook", "tag"):
        v = getattr(a, key)
        if v is not None:
            p[key] = v
    missing = [k for k in ("left", "right", "hp") if k not in p]
    if missing:
        raise SystemExit(f"missing {', '.join('--' + m for m in missing)} (or use --preset)")
    return Config(
        left=Team.parse(p["left"], BLUE),
        right=Team.parse(p["right"], ORANGE),
        hp=tuple(int(x) for x in str(p["hp"]).split(",")),
        seed=a.seed,
        fps=a.fps,
        gate=a.gate,
        hold=a.hold,
        max_seconds=a.max_seconds,
        speed=a.speed,
        gravity=a.gravity,
        tag=p.get("tag", "One vs Many"),
        hook=p.get("hook", ""),
    )


def _stem(a: argparse.Namespace) -> str:
    return a.stem or f"race_{a.preset or 'custom'}_seed{a.seed}"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="swarm", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("race", "plan", "frames"):
        sp = sub.add_parser(name)
        sp.add_argument("--preset", choices=sorted(PRESETS))
        sp.add_argument("--left", help='NAME:START:RULE:N, e.g. "MANY:500:mul:2"')
        sp.add_argument("--right", help='NAME:START:RULE:N, e.g. "ONE:1:mul:7"')
        sp.add_argument("--hp", help="floor HP, top to bottom, e.g. 8,25,110,450")
        sp.add_argument("--hook")
        sp.add_argument("--tag")
        sp.add_argument("--seed", type=int, default=1)
        sp.add_argument("--fps", type=int, default=60)
        sp.add_argument("--gate", type=float, default=0.5)
        sp.add_argument("--hold", type=float, default=2.2)
        sp.add_argument("--speed", type=float, default=250.0, help="px/s at the lid")
        sp.add_argument("--gravity", type=float, default=700.0, help="px/s²")
        sp.add_argument("--max-seconds", type=float, default=90.0)
        sp.add_argument("--out", type=Path, default=Path("out"))
        sp.add_argument("--stem")
        sp.add_argument("--quiet", action="store_true")
        if name == "race":
            sp.add_argument("--no-sound", action="store_true")
        if name == "frames":
            sp.add_argument("--at", required=True, help="seconds, e.g. 0.1,10,20")
    sub.add_parser("list-presets")
    a = ap.parse_args(argv)
    if a.cmd == "list-presets":
        print(json.dumps(PRESETS, indent=2, ensure_ascii=False))
        return
    cfg = _config(a)
    if a.cmd == "plan":
        summary = run(cfg, a.out, _stem(a), render=False, quiet=a.quiet)
    elif a.cmd == "frames":
        at = [float(x) for x in a.at.split(",")]
        summary = run(cfg, a.out, _stem(a), render=False, at=at, quiet=a.quiet)
    else:
        summary = run(cfg, a.out, _stem(a), sound=not a.no_sound, quiet=a.quiet)
        print(f"wrote {a.out / (_stem(a) + '.mp4')}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
