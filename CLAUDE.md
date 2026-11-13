# swarm_studio

Growth-race videos with tens of thousands of real balls for Channel S (Kinetic Code, series
*One vs Many*). Two tanks of floors with HP; every floor hit takes 1 HP; a broken floor grows the
side's ball count by its rule. Writes the workspace **export pack**. Spec: `README.md`.

## Stack
Python 3.12 (uv), numpy, numba (parallel `prange` kernels), pygame-ce (text only: its default
freetype font, the same one bouncekit draws with), ffmpeg via subprocess. Dev: pytest, ruff.
No other runtime dependencies without asking the user.

## Commands
- Install: `uv sync` · Tests: `uv run pytest` · Lint: `uv run ruff check src tests` · Format: `uv run ruff format src tests`
- Physics only (tuning, ~1-2 min): `uv run swarm plan --preset x7 [--hp ...] [--seed N]` → floor-break times, turn, lead change, winner, `loser_hp_left`
- Stills: `uv run swarm frames --preset x7 --at 0.1,10,20` → `out/<stem>_t<s>.png`
- Render: `uv run swarm race --preset x7 [--seed N] [--out DIR] [--stem NAME]` (~1 s of wall time per s of video at 100k+ balls)
- Custom: `--left "MANY:500:mul:2" --right "ONE:1:mul:7" --hp 5,18,55,250,1500,7500,160000 --hook "..." --tag "One vs Many · #N"`
- Flags: `--fps 60 --gate 0.5 --hold 2.2 --speed 400 --max-seconds 90 --no-sound --quiet`

## Layout (src/swarm_studio)
- `engine.py` `Swarm`: one tank's balls. Jacobi collision pass (push apart only; `exchange=0`
  because a Jacobi velocity swap clumps the swarm), speed relaxation toward `speed`, a minimum
  vertical share `min_vy` (no ball skims between the side walls for ever), wall jitter from an
  integer hash, floor hits counted when moving down faster than `min_hit × speed`.
- `race.py` `Team` (`NAME:START:mul|add|pow:N`), `Config` (layout constants, HP list), `Race`
  (`advance()` = one video frame: substeps → hits → HP → breaks → growth, turn, leader, win).
- `render.py` `Painter.frame()`: tanks, floors (active one thins/dims with HP, number just below
  each line), discs (numba, serial), glow for big balls, break particles, gauges (one segment per
  standing floor), text. `audio.py` plucks/marimba/bells from the race log. `post.py` copy.
  `studio.py` pipeline (stream frames → ffmpeg, mix audio, mux, export pack). `cli.py` presets.

## Physics facts (use them to set HP)
The swarm is a gas, so hits per ball per second ≈ `speed / (π × chamber height)` (chamber = lid
to the active floor; checked in `tests/test_engine.py`). Floor time ≈ HP ÷ (count × that rate)
plus ~0.3-0.5 s for the balls to reach a new floor. A lone ball hits a bit less often. Design HP
from the floor times you want, then confirm with `swarm plan` on 2-3 seeds.
Keep every count ≲ 120,000 (ball radius bottoms out at 0.6 px; render time grows with count).

## Export pack contract (must not break)
`<out>/<stem>.mp4` (H.264 yuv420p +faststart, AAC 48 kHz stereo, 1080x1920@60),
`<stem>_thumb.png` (frame at 0.3 s: the stakes, never the result), `<stem>_post.json` (title
≤ 100, description with "Made with code.", hashtags without `#`, `platforms`), `<stem>.json`
(params, seed, race summary: breaks, turn, overtake, winner, loser HP left). Default out `out/`.

## Conventions
- Determinism: same args + seed → same frames and byte-identical audio. Simulation randomness only
  from `numpy.random.default_rng(seed)` and the integer hash; visual-only randomness (particles)
  from a separate rng. Parallel kernels write only their own ball (no data races).
- Honesty: counts on screen are the real ball counts; HP falls by exactly the hits. Never scale
  damage per ball or draw fewer balls than the count.
- Look at frames (`swarm frames` or ffmpeg) and check the audio stream before calling a video done.
- Commits: conventional (`feat(engine): ...`), ending with the Co-Authored-By line. Never push.
