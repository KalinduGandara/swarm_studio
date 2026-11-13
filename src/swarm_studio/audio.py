"""Procedural soundtrack from the race log: one pluck tone for every floor hit, marimba for
breaks, stings for the turning point, the lead change and the win. 48 kHz stereo float, no
randomness."""

from __future__ import annotations

import math
import wave
from functools import lru_cache
from pathlib import Path

import numpy as np

SR = 48_000
MAJOR = (0, 2, 4, 5, 7, 9, 11)


def midi_hz(m: float) -> float:
    return 440.0 * 2.0 ** ((m - 69) / 12)


@lru_cache(maxsize=256)
def _voice(kind: str, midi: int) -> np.ndarray:
    f = midi_hz(midi)
    if kind == "pluck":
        n = int(0.45 * SR)
        t = np.arange(n) / SR
        env = np.exp(-t / 0.11) * np.minimum(1.0, t / 0.002)
        sig = sum(a * np.sin(2 * np.pi * f * h * t) for h, a in ((1, 1.0), (2, 0.45), (3, 0.2)))
    elif kind == "marimba":
        n = int(1.0 * SR)
        t = np.arange(n) / SR
        env = np.minimum(1.0, t / 0.003)
        sig = np.exp(-t / 0.35) * np.sin(2 * np.pi * f * t) + 0.35 * np.exp(-t / 0.06) * np.sin(
            2 * np.pi * 4 * f * t
        )
    else:  # bell for stings
        n = int(1.6 * SR)
        t = np.arange(n) / SR
        env = np.minimum(1.0, t / 0.004) * np.exp(-t / 0.5)
        sig = np.sin(2 * np.pi * f * t + 1.2 * np.exp(-t / 0.4) * np.sin(2 * np.pi * 2.0 * f * t))
    return (env * sig).astype(np.float64)


class Mixer:
    def __init__(self, seconds: float) -> None:
        self.buf = np.zeros((int(math.ceil(seconds * SR)) + SR * 2, 2))

    def add(self, t: float, kind: str, midi: int, gain: float, pan: float) -> None:
        w = _voice(kind, midi)
        start = int(round(t * SR))
        end = min(start + w.size, self.buf.shape[0])
        if end <= start:
            return
        lg = gain * math.cos((pan + 1) * math.pi / 4)
        rg = gain * math.sin((pan + 1) * math.pi / 4)
        self.buf[start:end, 0] += w[: end - start] * lg
        self.buf[start:end, 1] += w[: end - start] * rg

    def finish(self, seconds: float, fade: float, peak_db: float = -1.0) -> np.ndarray:
        out = self.buf[: int(round(seconds * SR))].copy()
        if fade > 0:
            n = min(int(fade * SR), out.shape[0])
            out[-n:] *= np.linspace(1.0, 0.0, n)[:, None]
        peak = float(np.abs(out).max()) or 1.0
        return out * (10 ** (peak_db / 20) / peak)


def soundtrack(log: list[dict], events: list, seconds: float, fps: int, fade: float) -> np.ndarray:
    """``log``: one dict per frame {t, hits: [left, right], floor: [left, right]}."""
    mix = Mixer(seconds)
    last = [-1.0, -1.0]
    hit_note = 72  # every hit plays the same tone (C5), panned to its tank
    pans = (-0.55, 0.55)
    for row in log:
        for i in (0, 1):
            h = row["hits"][i]
            if h <= 0 or row["t"] - last[i] < 0.075:
                continue
            last[i] = row["t"]
            gain = 0.22 * min(1.0, 0.45 + 0.12 * math.log10(1 + h))
            mix.add(row["t"], "pluck", hit_note, gain, pans[i])
    for ev in events:
        if ev.kind == "break":
            midi = 60 + MAJOR[ev.floor % 7] + 12 * (ev.floor // 7)
            mix.add(ev.t, "marimba", midi, 0.55, pans[ev.side])
        elif ev.kind == "turn":
            for m in (84, 88):
                mix.add(ev.t, "bell", m, 0.3, 0.0)
        elif ev.kind == "lead":
            for m in (79, 84):
                mix.add(ev.t, "bell", m, 0.28, pans[ev.side])
        elif ev.kind == "win":
            for k, m in enumerate((72, 76, 79, 84)):
                mix.add(ev.t + 0.09 * k, "bell", m, 0.35, 0.0)
    return mix.finish(seconds, fade)


def write_wav(path: Path, samples: np.ndarray) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).round().astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
