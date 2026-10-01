from dataclasses import FrozenInstanceError, replace
from functools import partial
import json
from types import FunctionType

import numpy as np
import pytest

from aeromill import session
from aeromill.config import ToolProfile
from aeromill.scenarios import Region, Scenario


def parts_api():
    from aeromill import parts

    return parts


def assert_no_hidden_references(value, seen=None):
    from aeromill.engine import Engine
    from aeromill.parts import Part
    from aeromill.simulator import Simulator

    seen = set() if seen is None else seen
    if id(value) in seen:
        return
    seen.add(id(value))
    assert not isinstance(value, (Part, Scenario, Region, Engine, Simulator))
    if isinstance(value, dict):
        children = list(value.values())
    elif isinstance(value, (tuple, list)):
        children = list(value)
    elif isinstance(value, partial):
        children = [value.func, value.args, value.keywords]
    elif isinstance(value, FunctionType):
        children = [cell.cell_contents for cell in value.__closure__ or ()]
    elif callable(value) and getattr(value, "__self__", None) is not None:
        children = [value.__self__]
    elif type(value).__module__.startswith("aeromill.") and hasattr(value, "__dict__"):
        children = list(vars(value).values())
    else:
        children = []
    for child in children:
        assert_no_hidden_references(child, seen)


def test_demo_parts_are_reproducible_and_seed_isolated():
    parts = parts_api()
    part = parts.random_part(500000)
    assert part == parts.random_part(500000)
    assert part != parts.random_part(500001)
    assert parts.validate_part(part) == part
    with pytest.raises(FrozenInstanceError):
        part.seed = 42
    for seed in (42, 105094, 200000, 300000, 400059, 499999, 600000, 900000, True):
        with pytest.raises(ValueError, match="seed"):
            parts.random_part(seed)
    samples = [parts.random_part(seed) for seed in range(500000, 500100)]
    assert all(p.profile == ToolProfile() for p in samples)
    assert {len(p.scenario.regions) for p in samples} == {0, 1, 2}
    assert {len(p.scenario.engagement) for p in samples} == {0, 1, 2, 3}
    assert any(p.scenario.impact_positions_mm for p in samples)


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("noise_std", float("nan"), "noise_std"),
        ("noise_std", 0.101, "noise_std"),
        ("chatter_hz", 899.0, "chatter_hz"),
        ("chatter_hz", float("inf"), "chatter_hz"),
        ("initial_phase", [0.0, 0.0], "initial_phase"),
        ("initial_phase", (0.0,), "initial_phase"),
        ("initial_phase", (0.0, "bad"), "initial_phase"),
        ("profile", None, "profile"),
        ("scenario", None, "scenario"),
        ("seed", 42, "seed"),
        ("noise_std", 10**1000, "noise_std"),
        ("profile", ToolProfile(profile_id=np.array(["bad", "id"])), "profile"),
    ],
)
def test_custom_parts_refuse_bad_values(field, value, reason):
    parts = parts_api()
    with pytest.raises(ValueError, match=reason):
        replace(parts.random_part(500000), **{field: value})


@pytest.mark.parametrize(
    "region,reason",
    [
        (Region(80.0, 400.0, (3200.0,), (0.0,)), "width"),
        (Region(80.0, 400.0, (4201.0,), (120.0,)), "center"),
        (Region(80.0, 400.0, (3200.0,), (120.0,), gain=2.0), "gain"),
        (Region(80.0, 400.0, [3200.0], (120.0,)), "centers"),
        (Region(80.0, 400.0, (3200.0,), (120.0, 130.0)), "width"),
        (Region(80.0, 400.0, (3200.0,), (120.0,), zone_class="secret"), "zone_class"),
        (Region(float("nan"), 400.0, (3200.0,), (120.0,)), "start_mm"),
        (Region(80.0, 401.0, (3200.0,), (120.0,)), "end_mm"),
    ],
)
def test_custom_nested_geometry_is_strict(region, reason):
    parts = parts_api()
    with pytest.raises(ValueError, match=reason):
        parts.Part(500000, Scenario("custom", (region,)))
    with pytest.raises(ValueError, match="regions"):
        parts.Part(500000, Scenario("custom", [region]))


def test_unknown_session_hides_truth_and_reveals_detached_terminal_data():
    parts = parts_api()
    part = parts.Part(
        500000, Scenario("custom", (Region(80.0, 400.0, (3200.0,), (120.0,)),))
    )
    run = session.RunSession.from_part(part, "ml_agent")
    assert run.snapshot().zones == []
    with pytest.raises(RuntimeError, match="terminal"):
        run.reveal()
    with pytest.raises(RuntimeError, match="terminal"):
        run.stability_map(1200.0, [3200.0], [80.0])
    live = run.advance(10)
    assert live.zones == [] and live.evaluation is None
    assert all("seed" not in event for event in live.events)
    assert all("seed" not in event for event in json.loads(run.export()[1])["events"])
    assert run._engine.events[0]["seed"] == 500000
    assert set(live.series) == {"t", "score", "rpm", "feed", "x"}
    assert not any(
        k in vars(run._engine.agent) for k in ("scenario", "part", "simulator", "truth")
    )
    assert_no_hidden_references(run._engine.agent)
    assert_no_hidden_references(run._engine.detector)
    assert_no_hidden_references(live)
    final = run.advance(650)
    assert final.terminal and final.evaluation["run_success"]
    truth = run.reveal()
    assert truth["zones"] == part.scenario.regions
    assert truth["engagement"] == part.scenario.engagement
    assert len(truth["time_s"]) == len(truth["envelope"]) == len(truth["truth_states"])
    assert np.max(truth["envelope"]) > 0.75
    assert "chatter" in truth["truth_states"]
    truth["envelope"][:] = -1
    truth["evaluation"]["run_success"] = False
    assert np.min(run.reveal()["envelope"]) >= 0
    assert run.reveal()["evaluation"]["run_success"]
    assert run.stability_map(1200.0, [3200.0], [80.0])["gain_resonance"][0, 0] == 1.0


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"impact_positions_mm": (np.array([120.0, 280.0]),)}, "impact_positions_mm"),
        ({"impact_positions_mm": [120.0, 280.0]}, "impact_positions_mm"),
        ({"engagement": ("bad",)}, "engagement"),
        ({"regions": ("bad",)}, "regions"),
        ({"path_length_mm": True}, "path_length_mm"),
        ({"path_length_mm": float("inf")}, "path_length_mm"),
    ],
)
def test_custom_scenario_malformed_values_have_reasons(changes, reason):
    parts = parts_api()
    with pytest.raises(ValueError, match=reason):
        parts.Part(500000, Scenario("custom", **changes))
