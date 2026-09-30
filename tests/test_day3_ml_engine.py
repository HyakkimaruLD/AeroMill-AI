"""Repeat Day 2 scenario/fault checks using RF on the shared engine."""

from dataclasses import replace
from pathlib import Path
import json

import numpy as np
import pytest

from aeromill.engine import Engine
from aeromill.scenarios import BASE_SCENARIOS, NAMED_SCENARIOS, EngagementSegment
from test_engine import (
    test_scenarios as day2_scenarios,
    test_command_failure as day2_command_failure,
    test_startup_calibration_and_run_timeout as day2_startup,
    test_quality_before_detector as day2_quality,
    test_stop_during_ramp as day2_stop,
    test_stable_profiles_and_impacts as day2_profiles,
    test_model_and_log_errors_stop as day2_errors,
    test_hold_stops_boundary_state_without_rewriting_history as day2_boundary,
)
import test_engine


@pytest.fixture
def ml_engine(monkeypatch):
    def make(*args, **kwargs):
        kwargs["mode"] = "ml_agent"
        return Engine(*args, **kwargs)

    monkeypatch.setattr(test_engine, "engine", make)


@pytest.mark.parametrize(
    "scenario,state,pairs",
    [
        ("stable", "COMPLETED", []),
        ("recover_a", "COMPLETED", [(3520, 960)]),
        ("retry_b", "COMPLETED", [(3520, 960), (2880, 960)]),
        ("recover_c", "COMPLETED", [(3520, 960), (2880, 960), (3840, 720)]),
        ("unrecoverable", "HOLD", [(3520, 960), (2880, 960), (3840, 720)]),
    ],
)
def test_day2_scenarios_with_rf(tmp_path, ml_engine, scenario, state, pairs):
    day2_scenarios(tmp_path, scenario, "ml_agent", state, pairs)


@pytest.mark.parametrize(
    "fault,reason", [("reject", "COMMAND_REJECTED"), ("lost_ack", "ACK_TIMEOUT")]
)
def test_day2_command_faults(tmp_path, ml_engine, fault, reason):
    day2_command_failure(tmp_path, fault, reason)


@pytest.mark.parametrize(
    "fault",
    ["nan", "inf", "gap", "repeat", "time", "shape", "missing", "zero", "nonnumeric"],
)
def test_day2_data_quality(tmp_path, ml_engine, fault):
    day2_quality(tmp_path, fault)


@pytest.mark.parametrize("h,noise", [(2.0, 0.05), (0.8, 0.1)])
def test_day2_profile_checks(tmp_path, ml_engine, h, noise):
    day2_profiles(tmp_path, h, noise)


@pytest.mark.parametrize("check", [day2_startup, day2_stop, day2_errors, day2_boundary])
def test_day2_other_checks(tmp_path, ml_engine, check):
    check(tmp_path)


def test_hidden_engagement_no_commands(tmp_path):
    e = Engine(
        NAMED_SCENARIOS["stable_engagement"],
        mode="ml_agent",
        seed=412,
        log_path=tmp_path / "ml.jsonl",
    ).run()
    assert (
        e.agent.state == "COMPLETED"
        and e.controller.action_count == 0
        and e.agent.incident == 0
    )


def test_recovery_then_engagement_and_repeat(tmp_path):
    s = replace(
        BASE_SCENARIOS["recover_a"], engagement=(EngagementSegment(200, 300, 2.0),)
    )
    e = Engine(s, mode="ml_agent", seed=413, log_path=tmp_path / "recovery.jsonl")
    claims = []
    truth = []
    times = []
    while not e.done:
        step = e.tick()
        truth.extend(step.truth.envelope)
        times.extend(step.telemetry.time_s)
    claims = [r for r in e.agent.events if r["event"] == "recovery_claim"]
    assert len(claims) == 1 and e.controller.action_count == 1 and e.agent.incident == 1
    t = np.asarray(times)
    a = np.asarray(truth)
    mask = (t > claims[0]["time_s"]) & (t <= claims[0]["time_s"] + 3)
    assert mask.sum() >= 24575 and np.all(a[mask] <= 0.25)
    assert e.agent.state == "COMPLETED"
    # Load rises during VERIFY too: score and residual must remain calm.
    s = replace(
        BASE_SCENARIOS["recover_a"], engagement=(EngagementSegment(120, 240, 2.0),)
    )
    e = Engine(
        s, mode="ml_agent", seed=414, log_path=tmp_path / "verify-engagement.jsonl"
    ).run()
    assert e.agent.state == "COMPLETED" and e.controller.action_count == 1
    e = Engine(
        BASE_SCENARIOS["repeat_after_c"],
        mode="ml_agent",
        seed=415,
        log_path=tmp_path / "repeat.jsonl",
    ).run()
    assert e.agent.state == "COMPLETED" and e.agent.incident == 2
    assert [
        (r["rpm"], r["feed_mm_min"])
        for r in e.agent.events
        if r["event"] == "set_cutting_parameters"
    ] == [(3520, 960), (2880, 960), (3840, 720), (3520, 960)]
