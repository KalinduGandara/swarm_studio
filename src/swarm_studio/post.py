"""Default social copy for the export pack (<stem>_post.json). Honest: the title is the hook
question; the description states the rule and that every ball is a real ball."""

from __future__ import annotations

from .race import Config, Race

HASHTAGS = ["simulation", "satisfying", "exponentialgrowth", "physics", "onevsmany"]


def build_post(cfg: Config, race: Race, seconds: float) -> dict:
    lt, rt = cfg.left, cfg.right
    title = cfg.hook or f"{rt.start:,} vs {lt.start:,}: who reaches the bottom first?"
    title = title if len(title) <= 100 else title[:99].rstrip() + "…"

    def says(t) -> str:
        kind, n = t.rule.split(":")
        one = t.start == 1
        verb = {
            "mul": f"{'grows' if one else 'grow'} ×{n}",
            "add": f"{'gains' if one else 'gain'} {int(n):,}",
            "pow": (
                f"{'squares its' if one else 'square their'} count"
                if n == "2"
                else f"{'raises its' if one else 'raise their'} count to the power {n}"
            ),
        }[kind]
        unit = "ball" if one else "balls"
        return f"{t.start:,} {unit} that {verb} on every floor"

    description = (
        f"{says(lt).capitalize()} vs {says(rt)}. Every hit on a floor takes 1 HP; first to the "
        "bottom wins. Every ball on screen is a real ball. Made with code."
    )
    caption = f"{title}\n\n{description}\n\n" + " ".join(f"#{h}" for h in HASHTAGS)
    return {
        "title": title,
        "description": description,
        "hashtags": HASHTAGS,
        "pinned_comment": "Which side did you pick before it started?",
        "platforms": {
            "youtube": {
                "title": title,
                "description": f"{description}\n\n"
                + " ".join(f"#{h}" for h in HASHTAGS + ["shorts"]),
                "tags": ["physics simulation", "exponential growth", "one vs many", "satisfying"],
                "category_id": "24",
                "made_for_kids": False,
                "playlist": "One vs Many",
            },
            "tiktok": {"caption": caption, "cover_timestamp_ms": 300},
            "instagram": {"caption": caption, "thumb_offset_ms": 300},
            "facebook": {"title": title, "description": description},
        },
    }
