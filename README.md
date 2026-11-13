# swarm_studio

Growth-race videos for Channel S (Kinetic Code, series *One vs Many*): two tanks, each split
into floors with HP. **Every ball on screen is a real ball and every hit on a floor takes 1 HP.**
When a floor breaks, that side's ball count grows by its rule (×n, +n, squared) and the balls
move down into the freed space. First side to break the bottom floor wins.

Tens of thousands of balls are simulated with a small deterministic numba particle engine
(bouncekit/pymunk tops out near a thousand). Output is the workspace export pack.

## What a video is (spec)

- **Format:** 1080x1920 @ 60 fps Short, 30-45 s, wordless, Channel S brand look (ink `#0b0b14`,
  white hook, muted series tag, gold result, team colours from `py_animation/scenes/brand.py`).
- **Tanks:** 2 side by side (`left`, `right`), each `name`, `start` count, growth `rule`, colour.
  Rules: `mul:N` (×N), `add:N` (+N), `pow:N` (count^N).
- **Floors:** one HP list shared by both tanks (same race for both). The number sits just
  below each floor line and counts down with every hit; the line thins and dims as its HP falls.
- **Physics (gas, not a pile):** balls keep a steady speed (like bouncekit's `constant_speed`)
  under light gravity, collide elastically with each other and the walls, and fill their
  chamber. Hits per second on a floor therefore grow with the number of balls, which is what
  makes "more balls = faster" real. A hit is a ball reaching the active floor while moving down.
- **Ball size:** all balls of a side share one radius that shrinks as the count grows (a lone
  ball is drawn big; 60,000 balls are ~1.5 px dots). Keep the biggest count ≲ 70,000 per side.
- **Growth on a break:** new balls are copies of existing balls with a small jitter, so the
  swarm visibly multiplies.
- **Beats:** start gate; floor breaks (particles + marimba); the turning point when the side
  that started behind has more balls ("ONE HAS MORE BALLS!" + sting); the lead change; the win
  ("ONE WINS!" + margin "MANY WAS N HP SHORT"); hold; loop fade into frame 0.
- **Sound (procedural):** a soft pluck per batch of hits per side (rate-limited, pitch rises
  with the floor), marimba on breaks, stings on turn/lead/win. 48 kHz stereo, peak -1 dBFS.
- **Determinism:** same args + seed → same frames and byte-identical audio. Seeded numpy RNG
  only; the parallel kernels write each ball's own result only (no races).

First three videos (user, 2026-11-13): 1) 1 ball ×7 vs 1,000 ×2; 2) adding vs multiplying
(5,000 +5,000 per floor vs 1 ×6); 3) squaring (2 balls squared every floor vs 1,000 ×2).

## CLI

```bash
uv sync
uv run swarm race --preset x7 --seed 1                  # render + export pack into out/
uv run swarm race --left "MANY:1000:mul:2" --right "ONE:1:mul:7" --hp 3,20,140,1000,6500,42000 \
    --hook "Can 1 ball ×7 beat 1,000?" --tag "One vs Many · #2" --seed 1
uv run swarm plan --preset x7 --seed 1                  # physics only: floor times, winner, margin
uv run swarm frames --preset x7 --seed 1 --at 0.1,10,20 # PNG stills for checking
uv run swarm list-presets
```

Common flags: `--fps 60 --gate 0.5 --hold 2.2 --max-seconds 60 --out out/ --stem NAME --no-sound`.

## Export pack

`<out>/<stem>.mp4` (H.264 yuv420p +faststart, AAC 48 kHz stereo), `<stem>_thumb.png` (the
frame-0 stakes, never the result), `<stem>_post.json` (title ≤ 100, description with the
"Made with code." line, hashtags without `#`, `platforms` block), `<stem>.json` (params, seed,
floor-break times, winner, margin, render timings). Contract: `docs/export-pack.md` in the
workspace root.

## Tests

`uv run pytest` (engine conservation/containment/determinism, race rules, audio determinism,
one short real render checked with ffprobe).
