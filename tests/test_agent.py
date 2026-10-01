"""Table boundary tests. Wrong durations, thresholds, transitions or priority must fail."""

from dataclasses import replace
import ast
from pathlib import Path

import numpy as np
import pytest

from aeromill import agent
from aeromill.config import ToolProfile
from aeromill.contracts import TelemetryChunk, Detection, Observation, ControllerEvent


def observation(tick, score=0.0, ratio=1.0, events=(), rpm=3200.0, feed=1200.0, x=None):
    end = tick * 8192 // 10
    start = (tick - 1) * 8192 // 10
    times = np.arange(start + 1, end + 1) / 8192
    n = len(times)
    xyz = np.zeros((n, 3))
    xyz[:, 0] = tick * 2 if x is None else x
    t = TelemetryChunk(
        "r",
        tick - 1,
        start,
        times,
        xyz,
        ("nominal",) * n,
        np.full(n, rpm),
        np.full(n, feed),
        np.ones((n, 3)),
    )
    f = tuple(v for _ in range(3) for v in (ratio, 1.4, 1.5, 216.0, 0.2, 0.1, 0.0)) + (
        rpm,
        feed,
    )
    return Observation(
        t,
        Detection(
            f,
            score,
            residual_rms=tuple(
                np.asarray(ToolProfile().reference_rms) * ratio * np.sqrt(0.1)
            ),
        )
        if tick >= 3
        else None,
        events,
    )


def make_agent(**kwargs):
    return agent.Agent("r", ToolProfile(), **kwargs)


def warm(a):
    for tick in range(1, 23):
        assert a.tick(observation(tick)) is None
        assert a.state == "WARMUP"
    assert a.tick(observation(23)) is None
    assert a.state == "MONITOR"


def trigger(a):
    warm(a)
    assert a.tick(observation(24, 1.0, 2.0)) is None
    assert a.tick(observation(25, 1.0, 2.0)) is None
    cmd = a.tick(observation(26, 1.0, 2.0))
    assert a.state == "APPLY"
    assert (cmd.rpm, cmd.feed_mm_min) == (3520, 960)
    return cmd


def settled(a):
    cmd = trigger(a)
    a.tick(
        observation(27, 1, 2, (ControllerEvent(cmd.command_id, "accepted", 2.7, "r"),))
    )
    assert a.state == "SETTLE"
    a.tick(
        observation(
            28, 0, 1, (ControllerEvent(cmd.command_id, "reached", 2.8, "r"),), 3520, 960
        )
    )
    for k in range(29, 38):
        a.tick(observation(k, rpm=3520, feed=960))
        assert a.state == "SETTLE"
    a.tick(observation(38, rpm=3520, feed=960))
    assert a.state == "VERIFY"
    return cmd


def test_agent_api():
    assert hasattr(agent, "Agent")


def test_warmup_median_reset_and_deadline():
    a = make_agent()
    warm(a)
    assert np.allclose(a.calibration_rms, ToolProfile().reference_rms)
    a = make_agent()
    for k in range(1, 50):
        a.tick(observation(k, 1, 2))
        assert a.state == "WARMUP"
    cmd = a.tick(observation(50, 1, 2))
    assert (a.state, a.reason, cmd.type) == ("HOLD", "CALIBRATION_FAILED", "stop")
    a = make_agent()
    for k in range(1, 24):
        a.tick(observation(k, 1 if k == 15 else 0))
    assert a.state == "WARMUP"
    for k in range(24, 37):
        a.tick(observation(k))
    assert a.state == "MONITOR"


def test_detection_counter_resets():
    a = make_agent()
    warm(a)
    for k, p in [(24, 0.8), (25, 0.8), (26, 0.79), (27, 0.8), (28, 0.8)]:
        assert a.tick(observation(k, p)) is None
    assert a.tick(observation(29, 0.8)).type == "set"


@pytest.mark.parametrize(
    "score,ratio,success",
    [(0.0, 1.0, True), (0.5, 1.0, False), (0.3, 1.51, False), (0.31, 1.0, False)],
)
def test_verify_exact_15_ticks_and_last_ten(score, ratio, success):
    a = make_agent()
    settled(a)
    for k in range(39, 53):
        a.tick(observation(k, score, ratio, rpm=3520, feed=960))
        assert a.state == "VERIFY"
    cmd = a.tick(observation(53, score, ratio, rpm=3520, feed=960))
    assert a.state == ("MONITOR" if success else "APPLY")
    assert (cmd is None) == success
    if not success:
        assert (cmd.rpm, cmd.feed_mm_min) == (2880, 960)
    outcomes = [e for e in a.events if e["event"] == "record_outcome"]
    assert (
        len(outcomes) == 1
        and outcomes[0]["success"] == success
        and outcomes[0]["time_s"] == 5.3
    )


def test_ack_and_reached_deadlines_and_foreign_events():
    a = make_agent()
    cmd = trigger(a)
    a.tick(
        observation(
            27, events=(ControllerEvent(cmd.command_id, "accepted", 2.7, "old"),)
        )
    )
    assert a.state == "APPLY"
    a.tick(observation(28))
    assert (a.state, a.reason) == ("HOLD", "ACK_TIMEOUT")
    a = make_agent()
    cmd = trigger(a)
    a.tick(
        observation(27, events=(ControllerEvent(cmd.command_id, "accepted", 2.7, "r"),))
    )
    for k in range(28, 47):
        a.tick(observation(k))
    assert a.state == "SETTLE"
    a.tick(observation(47))
    assert (a.state, a.reason) == ("HOLD", "SETPOINT_TIMEOUT")


@pytest.mark.parametrize(
    "state", ["WARMUP", "MONITOR", "PLAN", "APPLY", "SETTLE", "VERIFY"]
)
def test_stop_error_override_every_active_state(state):
    for reason in ("MANUAL_STOP", "DATA_QUALITY"):
        a = make_agent()
        a.state = state
        cmd = a.tick(observation(10, x=400), error=reason)
        assert (a.state, a.reason, cmd.type) == ("HOLD", reason, "stop")
        assert (
            a.tick(
                observation(11, events=(ControllerEvent("late", "accepted", 1.1, "r"),))
            )
            is None
        )
        assert a.state == "HOLD"


def test_timeout_and_endpoint_priority():
    a = make_agent()
    a.state = "MONITOR"
    a.tick(observation(650, x=399))
    assert (a.state, a.reason) == ("HOLD", "RUN_TIMEOUT")
    a = make_agent()
    a.state = "MONITOR"
    a.tick(observation(650, x=400))
    assert a.state == "COMPLETED"


def test_no_candidate_filters_before_dispatch():
    a = make_agent()
    warm(a)
    a.candidates = ((3200.0, 1200.0), (0.0, 960.0), (3520.0, 0.0))
    for k in (24, 25, 26):
        cmd = a.tick(observation(k, 1, 2))
    assert (a.state, a.reason, cmd.type) == ("HOLD", "NO_CANDIDATE", "stop")


def test_agent_import_graph_excludes_hidden_modules():
    root = Path(__file__).parents[1] / "aeromill"
    seen = set()

    def visit(name):
        if name in seen:
            return
        seen.add(name)
        assert name not in {"simulator", "scenarios", "engine", "evaluation"}
        tree = ast.parse((root / f"{name}.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not any(
                    a.name in {"TruthFrame", "StepResult", "TruthState"}
                    for a in node.names
                )
                if node.level == 1 and node.module:
                    visit(node.module.split(".")[0])
                elif node.module and node.module.startswith("aeromill."):
                    visit(node.module.split(".")[1])
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("aeromill."):
                        visit(alias.name.split(".")[1])

    visit("agent")


def test_verify_uses_only_last_ten_and_requires_motion():
    a = make_agent()
    settled(a)
    for k in range(39, 54):
        a.tick(observation(k, 1 if k < 44 else 0, rpm=3520, feed=960))
    assert a.state == "MONITOR"
    a = make_agent()
    settled(a)
    for k in range(39, 54):
        a.tick(observation(k, rpm=3520, feed=960, x=76))
    assert a.state == "APPLY"


def test_exact_ack_reached_boundaries_are_accepted():
    a = make_agent()
    cmd = trigger(a)
    a.tick(observation(27))
    a.tick(
        observation(28, events=(ControllerEvent(cmd.command_id, "accepted", 2.8, "r"),))
    )
    for k in range(29, 48):
        a.tick(observation(k))
    a.tick(
        observation(48, events=(ControllerEvent(cmd.command_id, "reached", 4.8, "r"),))
    )
    assert a.state == "SETTLE"
    for k in range(49, 59):
        a.tick(observation(k))
    assert a.state == "VERIFY"


def test_median_calibration_not_last_sample():
    a = make_agent()
    for k in range(1, 24):
        a.tick(observation(k, ratio=1.4 if k == 23 else 1.0))
    assert np.allclose(a.calibration_rms, ToolProfile().reference_rms)


def test_three_attempts_exhaust_and_reset_after_success():
    a = make_agent()
    cmd = trigger(a)
    tick = 26
    pairs = [(cmd.rpm, cmd.feed_mm_min)]
    for attempt in range(3):
        tick += 1
        a.tick(
            observation(
                tick,
                1,
                2,
                (
                    ControllerEvent(cmd.command_id, "accepted", tick / 10, "r"),
                    ControllerEvent(cmd.command_id, "reached", tick / 10, "r"),
                ),
                cmd.rpm,
                cmd.feed_mm_min,
            )
        )
        for _ in range(25):
            tick += 1
            result = a.tick(
                observation(tick, 1, 2, rpm=cmd.rpm, feed=cmd.feed_mm_min, x=tick)
            )
        if attempt < 2:
            cmd = result
            pairs.append((cmd.rpm, cmd.feed_mm_min))
    assert pairs == [(3520, 960), (2880, 960), (3840, 720)]
    assert (a.state, a.reason) == ("HOLD", "ATTEMPTS_EXHAUSTED")
    assert len([e for e in a.events if e["event"] == "record_outcome"]) == 3
    a = make_agent()
    settled(a)
    for k in range(39, 54):
        a.tick(observation(k, rpm=3520, feed=960))
    for k in (54, 55):
        assert a.tick(observation(k, 1, 2, rpm=3520, feed=960)) is None
    cmd = a.tick(observation(56, 1, 2, rpm=3520, feed=960))
    assert (
        a.incident == 2
        and a.attempts == 1
        and (cmd.rpm, cmd.feed_mm_min) == (2880, 960)
    )


def test_command_deadlines_override_same_tick_endpoint():
    a = make_agent()
    cmd = trigger(a)
    a.tick(observation(27))
    a.tick(observation(28, x=400))
    assert (a.state, a.reason) == ("HOLD", "ACK_TIMEOUT")
    a = make_agent()
    cmd = trigger(a)
    a.tick(
        observation(27, events=(ControllerEvent(cmd.command_id, "accepted", 2.7, "r"),))
    )
    for k in range(28, 47):
        a.tick(observation(k))
    a.tick(observation(47, x=400))
    assert (a.state, a.reason) == ("HOLD", "SETPOINT_TIMEOUT")
    a = make_agent()
    for k in range(1, 50):
        a.tick(observation(k, 1, 2))
    a.tick(observation(50, 1, 2, x=400))
    assert (a.state, a.reason) == ("HOLD", "CALIBRATION_FAILED")
