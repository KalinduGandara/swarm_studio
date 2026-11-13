"""Pipeline: race → frames (streamed into ffmpeg) → soundtrack → mux → export pack."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .audio import soundtrack, write_wav
from .post import build_post
from .race import HEIGHT, WIDTH, Config, Race
from .render import Painter, write_png

THUMB_T = 0.3  # the frame-0 stakes, never the result


def ffmpeg_bin() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not found on PATH")
    return exe


class Encoder:
    """Raw RGB frames → H.264 (video only; audio is muxed afterwards)."""

    def __init__(self, out: Path, fps: int, crf: int = 18) -> None:
        cmd = [
            ffmpeg_bin(),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{WIDTH}x{HEIGHT}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-vf",
            "scale=out_color_matrix=bt709:out_range=tv:flags=bicubic",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-profile:v",
            "high",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            str(out),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg encoding failed")


def mux(video: Path, wav: Path, out: Path) -> None:
    cmd = [
        ffmpeg_bin(),
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-i",
        str(wav),
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-af",
        "loudnorm=I=-14:TP=-1.5:LRA=11",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-shortest",
        "-movflags",
        "+faststart",
        str(out),
    ]
    if subprocess.run(cmd).returncode != 0:
        raise RuntimeError("ffmpeg mux failed")


def _progress(race: Race, wall0: float, quiet: bool) -> None:
    if quiet or race.frame % (race.cfg.fps * 2):
        return
    counts = ", ".join(f"{s.team.name} {s.count:,} f{s.floor + 1}" for s in race.sides)
    print(
        f"  t={race.t:5.1f}s  {counts}  ({time.perf_counter() - wall0:.0f}s)",
        file=sys.stderr,
        flush=True,
    )


def run(
    cfg: Config,
    out_dir: Path,
    stem: str,
    *,
    render: bool = True,
    sound: bool = True,
    at: list[float] | None = None,
    quiet: bool = False,
) -> dict:
    """Run the race. render=False is `plan` (physics only); ``at`` saves stills instead."""
    out_dir.mkdir(parents=True, exist_ok=True)
    race = Race(cfg)
    painter = Painter(race) if (render or at) else None
    stills = sorted(at or [])
    enc = None
    tmp_video = out_dir / f"{stem}.video.mp4"
    if render and not at:
        enc = Encoder(tmp_video, cfg.fps)
    log: list[dict] = []
    frame0 = None
    thumb_done = False
    wall0 = time.perf_counter()
    while True:
        race.advance()
        log.append({"t": race.t, "hits": [s.hits_frame for s in race.sides]})
        _progress(race, wall0, quiet)
        end = race.end_t
        if painter is not None and (enc is not None or (stills and race.t + 1e-9 >= stills[0])):
            clear = end is not None and race.t >= end - cfg.loop_fade - 0.1
            img = painter.frame(clear_live=clear)
            if frame0 is None:
                frame0 = img
            if end is not None and race.t > end - cfg.loop_fade and enc is not None:
                w = min(1.0, (race.t - (end - cfg.loop_fade)) / cfg.loop_fade)
                img = (img * (1 - w) + frame0 * w).astype(np.uint8)
            if enc is not None:
                enc.write(img)
                if not thumb_done and race.t >= THUMB_T:
                    write_png(out_dir / f"{stem}_thumb.png", img)
                    thumb_done = True
            while stills and race.t + 1e-9 >= stills[0]:
                write_png(out_dir / f"{stem}_t{stills.pop(0):g}.png", img)
        if end is not None and race.t >= end - 1e-9:
            break
        if at is not None and not stills:
            break
        if race.t >= cfg.max_seconds:
            break
    seconds = race.frame / cfg.fps
    summary = race.summary()
    summary["seconds"] = round(seconds, 3)
    summary["render_seconds"] = round(time.perf_counter() - wall0, 1)
    if enc is not None:
        enc.close()
        if race.winner is None:
            raise RuntimeError(
                f"no winner by {cfg.max_seconds:g} s; lower the HP (run `swarm plan` first)"
            )
        wav = out_dir / f"{stem}.wav"
        audio = (
            soundtrack(log, race.events, seconds, cfg.fps, cfg.loop_fade)
            if sound
            else (np.zeros((int(seconds * 48000), 2)))
        )
        write_wav(wav, audio)
        mux(tmp_video, wav, out_dir / f"{stem}.mp4")
        tmp_video.unlink()
        wav.unlink()
        meta = {
            "tool": "swarm_studio",
            "stem": stem,
            "seed": cfg.seed,
            "params": {k: v for k, v in asdict(cfg).items() if k not in ("left", "right")}
            | {"left": asdict(cfg.left), "right": asdict(cfg.right)},
            "duration": round(seconds, 3),
            "size": [WIDTH, HEIGHT],
            "fps": cfg.fps,
            "race": summary,
            "thumb_at": THUMB_T,
        }
        (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2) + "\n")
        post = build_post(cfg, race, seconds)
        (out_dir / f"{stem}_post.json").write_text(
            json.dumps(post, indent=2, ensure_ascii=False) + "\n"
        )
    return summary
