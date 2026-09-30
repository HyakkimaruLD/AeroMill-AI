"""R9: hidden load changes the signal but never the truth or public contract."""

from dataclasses import asdict, fields, replace

import numpy as np
import pytest

from aeromill import scenarios
from aeromill.config import ToolProfile
from aeromill.contracts import AppliedState, Observation, TelemetryChunk
from aeromill.engine import Engine
from aeromill.simulator import Simulator


def test_engagement_contract_and_ramps():
    assert hasattr(scenarios, "EngagementSegment"), "hidden engagement is missing"
    segment = scenarios.EngagementSegment(150.0, 250.0, 2.0)
    s = replace(scenarios.BASE_SCENARIOS["stable"], engagement=(segment,))
    assert [s.engagement_at(x) for x in (0, 59, 150, 160, 170, 230, 240, 250, 300)] == [
        1,
        1,
        1,
        1.5,
        2,
        2,
        1.5,
        1,
        1,
    ]
    for cls in (ToolProfile, Observation, TelemetryChunk):
        assert not {"engagement", "family_id", "split", "truth", "envelope"} & {
            f.name for f in fields(cls)
        }
    assert asdict(ToolProfile()) == {
        "profile_id": "four-tooth-v1",
        "h": 1.0,
        "teeth": 4,
    }
    for args in [(40, 100, 2), (70, 100, 2), (70, 200, 2), (70, 150, 2.1)]:
        with pytest.raises(ValueError):
            scenarios.EngagementSegment(*args)


def test_engagement_changes_only_harmonic_signal():
    assert "stable_engagement" in scenarios.NAMED_SCENARIOS
    a = Simulator(scenarios.NAMED_SCENARIOS["stable_engagement"], seed=411)
    b = Simulator(scenarios.BASE_SCENARIOS["stable"], seed=411)
    changed = False
    for _ in range(150):
        x, y = a.step(0.1, AppliedState()), b.step(0.1, AppliedState())
        assert np.array_equal(x.truth.envelope, y.truth.envelope)
        assert np.array_equal(x.truth.states, y.truth.states)
        if x.telemetry.xyz_mm[-1, 0] <= 150:
            assert np.array_equal(x.telemetry.vibration, y.telemetry.vibration)
        else:
            changed |= not np.array_equal(x.telemetry.vibration, y.telemetry.vibration)
    assert changed


def test_total_rms_false_incident(tmp_path):
    assert hasattr(scenarios, "NAMED_SCENARIOS")
    e = Engine(
        scenarios.NAMED_SCENARIOS["stable_engagement"],
        seed=412,
        log_path=tmp_path / "r.jsonl",
    )
    truth = []
    while not e.done:
        truth.extend(e.tick().truth.envelope)
    assert max(truth) <= 0.25
    assert e.agent.incident > 0 and e.controller.action_count > 0


def test_manifest_initial_conditions():
    a = Simulator(
        scenarios.BASE_SCENARIOS["stable"],
        initial_state=AppliedState(2500, 650),
        initial_phase=(0.3, 0.7),
    )
    step = a.step(0.1, AppliedState(2500, 650))
    assert np.all(step.telemetry.rpm == 2500) and np.all(
        step.telemetry.feed_mm_min == 650
    )
    assert step.telemetry.xyz_mm[-1, 0] == pytest.approx(650 / 60 * (819 / 8192))
