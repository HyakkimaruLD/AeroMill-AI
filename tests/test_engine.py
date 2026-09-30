"""Full closed-loop regressions: real signals, actual commands, truth checked only here."""

from dataclasses import replace
import importlib
import json

import numpy as np
import pytest

from aeromill.config import ToolProfile
from aeromill.contracts import Detection
from aeromill.scenarios import BASE_SCENARIOS, Scenario, Region


def engine(*args, **kwargs):
    return importlib.import_module("aeromill.engine").Engine(*args, **kwargs)


def test_engine_exists():
    from aeromill import cli

    assert hasattr(cli, "main")


@pytest.mark.parametrize(
    "scenario,mode,state,pairs",
    [
        ("stable", "threshold_search", "COMPLETED", []),
        ("recover_a", "threshold_search", "COMPLETED", [(3520, 960)]),
        ("retry_b", "threshold_search", "COMPLETED", [(3520, 960), (2880, 960)]),
        (
            "recover_c",
            "threshold_search",
            "COMPLETED",
            [(3520, 960), (2880, 960), (3840, 720)],
        ),
        (
            "unrecoverable",
            "threshold_search",
            "HOLD",
            [(3520, 960), (2880, 960), (3840, 720)],
        ),
        ("recover_a", "baseline", "HOLD", [(3200, 720)]),
        ("recover_a", "no_adaptation", "COMPLETED", []),
    ],
)
def test_scenarios(tmp_path, scenario, mode, state, pairs):
    e = engine(
        BASE_SCENARIOS[scenario], seed=42, mode=mode, log_path=tmp_path / "run.jsonl"
    )
    truth = []
    times = []
    positions = []
    moving = []
    for _ in range(650):
        step = e.tick()
        if step is not None:
            truth.extend(step.truth.envelope)
            times.extend(step.telemetry.time_s)
            positions.extend(step.truth.x_mm)
            moving.extend(step.truth.moving)
        if e.done:
            break
    assert e.done and e.agent.state == state
    commands = [r for r in e.agent.events if r["event"] == "set_cutting_parameters"]
    assert [(r["rpm"], r["feed_mm_min"]) for r in commands] == pairs
    rows = [json.loads(s) for s in (tmp_path / "run.jsonl").read_text().splitlines()]
    assert [r for r in rows if r["event"] == "set_cutting_parameters"] == commands
    assert e.time_s <= 65
    transitions = [r["state"] for r in rows if r["event"] == "transition"]
    expected = ["MONITOR"] + ["PLAN", "APPLY", "SETTLE", "VERIFY"] * len(pairs)
    expected += (
        ["HOLD"]
        if state == "HOLD"
        else (["MONITOR", "COMPLETED"] if pairs else ["COMPLETED"])
    )
    assert transitions == expected
    verify_times = [
        r["time_s"]
        for r in rows
        if r["event"] == "transition" and r["state"] == "VERIFY"
    ]
    outcome_times = [r["time_s"] for r in rows if r["event"] == "record_outcome"]
    assert [
        b - a for a, b in zip(verify_times, outcome_times, strict=True)
    ] == pytest.approx([1.5] * len(pairs))
    claims = [r for r in rows if r["event"] == "recovery_claim"]
    if state == "HOLD":
        assert e.agent.reason == "ATTEMPTS_EXHAUSTED"
        assert e.controller.applied.stopped and e.controller.applied.feed_mm_min == 0
        assert e.simulator._feed == 0
        assert not claims
    elif pairs:
        assert len(claims) == 1
        t = np.asarray(times)
        a = np.asarray(truth)
        x = np.asarray(positions)
        m = np.asarray(moving)
        for claim in claims:
            mask = (t > claim["time_s"]) & (t <= claim["time_s"] + 3)
            assert mask.sum() >= 24575
            assert (
                np.max(a[mask]) <= 0.25
                and m[mask].all()
                and np.all(np.diff(x[mask]) > 0)
            )
    if mode == "no_adaptation":
        assert not claims
        chatter_s = np.count_nonzero(np.asarray(truth) >= 0.75) / 8192
        assert 15 < chatter_s < 16
    print(
        json.dumps(
            dict(
                scenario=scenario,
                mode=mode,
                state=state,
                time_s=e.time_s,
                commands=len(commands),
                claims=[r["time_s"] for r in claims],
                chatter_s=np.count_nonzero(np.asarray(truth) >= 0.75) / 8192,
            )
        )
    )


@pytest.mark.parametrize(
    "fault,reason", [("reject", "COMMAND_REJECTED"), ("lost_ack", "ACK_TIMEOUT")]
)
def test_command_failure(tmp_path, fault, reason):
    e = engine(
        BASE_SCENARIOS["recover_a"], fault=fault, log_path=tmp_path / "run.jsonl"
    )
    e.run()
    assert (e.agent.state, e.agent.reason) == ("HOLD", reason)
    assert e.time_s < 6 and e.controller.applied.stopped


def test_startup_calibration_and_run_timeout(tmp_path):
    for scenario, reason, time in [
        (
            Scenario("startup", (Region(0, 400, (3200,), (120,)),)),
            "CALIBRATION_FAILED",
            5,
        ),
        (Scenario("long", path_length_mm=2000), "RUN_TIMEOUT", 65),
    ]:
        e = engine(scenario, log_path=tmp_path / f"{reason}.jsonl")
        e.run()
        assert (e.agent.state, e.agent.reason, e.time_s) == ("HOLD", reason, time)


@pytest.mark.parametrize(
    "fault",
    ["nan", "inf", "gap", "repeat", "time", "shape", "missing", "zero", "nonnumeric"],
)
def test_quality_before_detector(tmp_path, fault):
    class Spy:
        calls = 0

        def predict(self, window, operating_state):
            self.calls += 1
            raise AssertionError("detector must not run on bad data")

    spy = Spy()
    e = engine(BASE_SCENARIOS["stable"], detector=spy, log_path=tmp_path / "run.jsonl")
    step = e.simulator.step(0.1, e.controller.applied)
    t = step.telemetry
    if fault in ("nan", "inf"):
        t = replace(t, vibration=np.full_like(t.vibration, float(fault)))
    elif fault == "gap":
        t = replace(t, sequence=3)
    elif fault == "repeat":
        t = replace(t, sequence=-1)
    elif fault == "time":
        t = replace(t, time_s=t.time_s[::-1])
    elif fault == "nonnumeric":
        t = replace(t, vibration=np.full(t.vibration.shape, "bad"))
    elif fault == "shape":
        t = replace(t, vibration=t.vibration[:, 0])
    elif fault == "missing":
        t = None
    elif fault == "zero":
        t = replace(t, vibration=np.zeros_like(t.vibration))
    if fault == "zero":
        # Fresh sequence stream, zero variance can only be judged with a full window.
        e = engine(
            BASE_SCENARIOS["stable"], detector=spy, log_path=tmp_path / "zero.jsonl"
        )
        original = e.simulator.step

        def zero_step(*args):
            s = original(*args)
            return replace(
                s,
                telemetry=replace(
                    s.telemetry, vibration=np.zeros_like(s.telemetry.vibration)
                ),
            )

        e.simulator.step = zero_step
        for _ in range(3):
            e.tick()
    else:
        e.simulator.step = lambda *args: replace(step, telemetry=t)
        e.tick()
    assert (e.agent.state, e.agent.reason) == ("HOLD", "DATA_QUALITY")
    assert spy.calls == 0 and e.controller.applied.feed_mm_min == 0


def test_stop_during_ramp(tmp_path):
    e = engine(BASE_SCENARIOS["recover_a"], log_path=tmp_path / "run.jsonl")
    while e.controller.action_count == 0:
        e.tick()
    before = e.simulator._x
    step = e.tick(stop=True)
    assert e.agent.reason == "MANUAL_STOP" and e.controller.applied.stopped
    assert e.simulator._x == before
    if step:
        assert np.all(step.telemetry.feed_mm_min == 0)
    assert e.tick() is None


@pytest.mark.parametrize("h,noise", [(2.0, 0.05), (0.8, 0.1)])
def test_stable_profiles_and_impacts(tmp_path, h, noise):
    scenario = replace(BASE_SCENARIOS["stable"], impact_positions_mm=(100.0, 200.0))
    e = engine(
        scenario,
        profile=ToolProfile(h=h),
        noise_std=noise,
        log_path=tmp_path / "run.jsonl",
    )
    e.run()
    assert e.agent.state == "COMPLETED" and e.controller.action_count == 0


def test_model_and_log_errors_stop(tmp_path):
    class Broken:
        def predict(self, *args):
            raise ValueError("broken model")

    e = engine(
        BASE_SCENARIOS["stable"], detector=Broken(), log_path=tmp_path / "run.jsonl"
    )
    e.run()
    assert (e.agent.reason, e.time_s) == ("MODEL_ERROR", 0.3)
    path = tmp_path / "broken.jsonl"
    path.write_text("not json")
    with pytest.raises(ValueError, match="corrupted log"):
        engine(BASE_SCENARIOS["stable"], log_path=path)
    e = engine(BASE_SCENARIOS["stable"], log_path=tmp_path / "write-error.jsonl")

    def broken_log(row):
        raise OSError("disk failure")

    e.log = broken_log
    e.agent.log = broken_log
    with pytest.raises(OSError):
        e.tick(stop=True)
    assert e.controller.applied.stopped and e.controller.applied.feed_mm_min == 0


def test_hold_stops_boundary_state_without_rewriting_history(tmp_path):
    e = engine(BASE_SCENARIOS["stable"], log_path=tmp_path / "stop-boundary.jsonl")
    history = e.tick()
    feed = history.telemetry.feed_mm_min.copy()
    e.stop("MANUAL_STOP")
    assert e.simulator._feed == 0.0
    assert np.array_equal(history.telemetry.feed_mm_min, feed)
    assert history.truth.moving.all()


def test_engine_selects_calibration_reference_only_in_verify(tmp_path):
    e = engine(BASE_SCENARIOS["stable"], log_path=tmp_path / "calibration.jsonl")
    e.agent.calibration_rms = np.asarray(ToolProfile().reference_rms) * 1.1
    e.agent.state = "VERIFY"
    original = e.simulator.step

    def tone(*args):
        s = original(*args)
        t = s.telemetry
        vibration = np.sin(2 * np.pi * 256 * t.time_s)[:, None] * np.array(
            [1.0, 0.8, 0.6]
        )
        return replace(s, telemetry=replace(t, vibration=vibration))

    e.simulator.step = tone
    for _ in range(3):
        e.tick()
    rows = [json.loads(line) for line in e.log_path.read_text().splitlines()]
    assert rows[-1]["score"] == 0.0
    e.agent.state = "MONITOR"
    e.tick()
    rows = [json.loads(line) for line in e.log_path.read_text().splitlines()]
    assert rows[-1]["score"] == 1.0
