"""Viewer-only geometry and real fault runs have independent oracles."""

from dataclasses import asdict
from math import exp

import numpy as np
import pytest


def viewer():
    from aeromill import viewer

    return viewer


def test_viewer_api_exists():
    import importlib.util

    assert importlib.util.find_spec("aeromill.viewer") is not None


def test_map_matches_formula():
    from aeromill.scenarios import NAMED_SCENARIOS

    scenario = NAMED_SCENARIOS["retry_b"]
    rpm = np.array([2880.0, 3200.0, 3520.0, 3840.0])
    xs = np.array([0.0, 79.9, 80.0, 200.0, 400.0])
    result = viewer().stability_map(scenario, 960.0, rpm, xs)
    expected = np.array(
        [
            [
                0.0
                if x < 80
                else min(
                    1.0, sum(exp(-(((s - c) / 120.0) ** 2)) for c in (3200.0, 3520.0))
                )
                for x in xs
            ]
            for s in rpm
        ]
    )
    assert np.array_equal(result["gain_resonance"], expected)
    assert np.array_equal(result["A_target"], 0.1 + 4 * expected * (960.0 / 1200))
    assert not np.array_equal(expected[0], expected[1])


@pytest.mark.parametrize(
    "name,reason",
    [
        ("nan_telemetry", "DATA_QUALITY"),
        ("missing_telemetry", "DATA_QUALITY"),
        ("out_of_order", "DATA_QUALITY"),
        ("rejected_command", "COMMAND_REJECTED"),
        ("lost_acknowledgment", "ACK_TIMEOUT"),
        ("manual_stop", "MANUAL_STOP"),
        ("startup_chatter", "CALIBRATION_FAILED"),
    ],
)
def test_catalog_failure_runs(name, reason):
    from aeromill.session import RunSession

    row = next(row for row in viewer().catalog() if row["name"] == name)
    assert all(
        row[key] for key in ("title", "hides", "expect", "evaluator_checks", "run")
    )
    session = RunSession(name, 42, "ml_agent")
    snap = session.advance(650)
    assert (snap.state, snap.reason) == ("HOLD", reason)
    assert snap.evaluation["run_success"] is False
    assert snap.target_feed_mm_min == 0.0
    assert snap.feed_mm_min == 0.0


def test_catalog_covers_presets_and_trajectory_matches_events():
    from aeromill.scenarios import NAMED_SCENARIOS
    from aeromill.session import RunSession

    assert set(NAMED_SCENARIOS) <= {r["name"] for r in viewer().catalog()}
    snap = RunSession("stable", 42, "threshold_search").advance(8)
    result = viewer().trajectory(snap)
    ticks = [r for r in snap.events if r["event"] == "tick"]
    assert np.array_equal(result["x_mm"], [r["x_mm"] for r in ticks])
    assert np.array_equal(result["rpm"], [r["rpm"] for r in ticks])
    result["rpm"][:] = -1
    assert (snap.series["rpm"] > 0).all()


def test_agent_callback_cannot_reach_engine(tmp_path):
    from aeromill.engine import Engine
    from aeromill.scenarios import NAMED_SCENARIOS

    e = Engine(NAMED_SCENARIOS["stable"], log_path=tmp_path / "run.jsonl")
    assert getattr(e.agent.log, "__self__", None) is not e
    assert not any(value is e for value in getattr(e.agent.log, "args", ()))
    e.tick()
    assert len(e.events) > 1


def test_terminal_unknown_map_uses_stopped_feed():
    from aeromill.parts import Part
    from aeromill.scenarios import NAMED_SCENARIOS
    from aeromill.session import RunSession

    run = RunSession.from_part(Part(500000, NAMED_SCENARIOS["recover_a"]), "ml_agent")
    run.request_stop()
    stopped = run.advance(1)
    assert stopped.terminal and stopped.feed_mm_min == 0.0
    result = run.stability_map(stopped.feed_mm_min, [3200.0, 3520.0], [0.0, 80.0])
    assert result["gain_resonance"][0, 1] == 1.0
    assert np.array_equal(result["A_target"], np.zeros((2, 2)))
    with pytest.raises(ValueError, match="range"):
        run.stability_map(1.0, [3200.0], [80.0])
