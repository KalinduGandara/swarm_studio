import numpy as np
import pytest

from swarm_studio.audio import soundtrack
from swarm_studio.race import Config, Race, Team, grow


def test_growth_rules():
    assert grow("mul:7", 343) == 2401
    assert grow("add:5000", 5000) == 10000
    assert grow("pow:2", 16) == 256
    with pytest.raises(ValueError):
        grow("div:2", 4)


def test_team_parse_and_rule_text():
    t = Team.parse("one:1:mul:7", "#ff9f1c")
    assert (t.name, t.start, t.rule) == ("ONE", 1, "mul:7")
    assert t.rule_text() == "×7 EVERY FLOOR"
    assert Team.parse("P:5:add:5000", "#000000").rule_text() == "+5,000 EVERY FLOOR"
    assert Team.parse("S:2:pow:2", "#000000").rule_text() == "SQUARED EVERY FLOOR"


def small_race(seed: int = 1) -> Race:
    cfg = Config(
        left=Team.parse("MANY:60:mul:2", "#4d7cff"),
        right=Team.parse("ONE:1:mul:6", "#ff9f1c"),
        hp=(2, 6, 20, 60),
        seed=seed,
        hold=0.2,
    )
    return Race(cfg)


def run(race: Race) -> Race:
    while race.end_t is None or race.t < race.end_t:
        race.advance()
        assert race.t < 60
    return race


def test_race_breaks_floors_grows_and_finishes():
    race = run(small_race())
    assert race.winner is not None
    w = race.sides[race.winner]
    assert w.floor == 4 and len(w.breaks) == 4
    assert w.breaks == sorted(w.breaks)
    # counts follow the rules: one growth per break except the last
    many, one = race.sides
    assert many.count == 60 * 2 ** min(many.floor, 3)
    assert one.count == 6 ** min(one.floor, 3)
    kinds = {e.kind for e in race.events}
    assert {"break", "win"} <= kinds


def test_same_seed_same_race():
    a, b = run(small_race(3)), run(small_race(3))
    assert a.summary() == b.summary()


def test_soundtrack_is_deterministic_and_peaks_at_minus_1_db():
    race = run(small_race())
    log = [{"t": i / 60, "hits": [i % 3, i % 5]} for i in range(120)]
    a = soundtrack(log, race.events, 2.0, 60, 0.4)
    b = soundtrack(log, race.events, 2.0, 60, 0.4)
    assert np.array_equal(a, b)
    assert a.shape == (96000, 2)
    assert abs(np.abs(a).max() - 10 ** (-1 / 20)) < 1e-6
