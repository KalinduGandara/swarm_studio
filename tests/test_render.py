"""One short real render through the CLI, checked with ffprobe and `release check`."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from swarm_studio.cli import main

ROOT = Path(__file__).resolve().parents[2]
needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffprobe"), reason="ffmpeg not installed")


@needs_ffmpeg
def test_short_render_writes_a_valid_export_pack(tmp_path, capsys):
    main(
        [
            "race",
            "--left",
            "MANY:60:mul:2",
            "--right",
            "ONE:1:mul:6",
            "--hp",
            "3,12,45,200",
            "--hook",
            "Can 1 ball ×6 beat 60?",
            "--hold",
            "1.5",
            "--out",
            str(tmp_path),
            "--stem",
            "t",
            "--quiet",
        ]
    )
    mp4 = tmp_path / "t.mp4"
    for name in ("t.mp4", "t_thumb.png", "t_post.json", "t.json"):
        assert (tmp_path / name).exists(), name
    probe = json.loads(
        subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(mp4)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    v = next(s for s in probe["streams"] if s["codec_type"] == "video")
    a = next(s for s in probe["streams"] if s["codec_type"] == "audio")
    assert (v["codec_name"], v["width"], v["height"], v["pix_fmt"]) == (
        "h264",
        1080,
        1920,
        "yuv420p",
    )
    assert (a["codec_name"], a["sample_rate"], a["channels"]) == ("aac", "48000", 2)
    meta = json.loads((tmp_path / "t.json").read_text())
    assert abs(float(probe["format"]["duration"]) - meta["duration"]) < 0.2
    post = json.loads((tmp_path / "t_post.json").read_text())
    assert len(post["title"]) <= 100 and "Made with code." in post["description"]
    assert all(not h.startswith("#") for h in post["hashtags"])
    release = ROOT / "release"
    if release.exists() and shutil.which("uv"):
        r = subprocess.run(
            ["uv", "run", "--project", str(release), "release", "check", str(mp4)],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, r.stdout + r.stderr
