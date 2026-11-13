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
- Custom: `--left "MANY:500:mul:2" --right "ONE:1:mul:7" --hp 5,23,72,390,2600,11700,350000 --hook "..." --tag "One vs Many · #N"`
- Flags: `--fps 60 --gate 0.5 --hold 2.2 --speed 250 --gravity 700 --max-seconds 90 --no-sound --quiet`

## Layout (src/swarm_studio)
- `engine.py` `Swarm`: one tank's balls. Jacobi collision pass (push apart only; `exchange=0`
  because a Jacobi velocity swap clumps the swarm), speed relaxation toward `speed`, a minimum
  vertical share `min_vy` (no ball skims between the side walls for ever), wall jitter from an
  integer hash, floor hits counted when moving down faster than `min_hit × speed`.
- `race.py` `Team` (`NAME:START:mul|add|pow:N`), `Config` (layout constants, HP list), `Race`
  (`advance()` = one video frame: substeps → hits → HP → breaks → growth, turn, leader, win).
- `render.py` `Painter.frame()`: tanks, floors (active one thins/dims with HP, number just below
  each line), discs (numba, serial), glow for big balls, break particles, gauges (one segment per
  standing floor), text. `audio.py` one pluck tone for every hit, marimba per break, bells from the race log. `post.py` copy.
  `studio.py` pipeline (stream frames → ffmpeg, mix audio, mux, export pack). `cli.py` presets.

## Physics facts (use them to set HP)
Gravity is energy-conserving: each ball is nudged toward the speed a bouncing ball has at its
height (`speed` at the lid, `sqrt(speed² + 2·g·depth)` lower down), so balls fall, bounce and
keep reaching the lid; no heap forms (densest near the lid). Hits per ball per second depend on
the chamber height but **not on the count** (tested), so damage grows with the number of balls.
Measured at speed 250, gravity 700: ~1.0 hits/ball/s on floor 1 of 7, ~0.25 in the full tank
(7 floors). Floor time ≈ HP ÷ (count × rate) + ~0.4 s for balls to reach a new floor. A lone
ball's floor 1 varies by ~1 s between seeds. Design HP, then confirm with `swarm plan` on 3 seeds.
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
