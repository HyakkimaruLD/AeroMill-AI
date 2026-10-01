"""Missing mandatory §4 rows: independent grading and failure priority in ML mode."""

from dataclasses import replace
import json
import numpy as np
import pytest

from aeromill.config import ToolProfile
from aeromill.contracts import AppliedState, Command, ControllerEvent, Detection
from aeromill.detector import RFDetector
from aeromill.engine import Engine
from aeromill.scenarios import BASE_SCENARIOS, NAMED_SCENARIOS, Region, Scenario
from test_agent import make_agent, observation, settled


class CalmDetector:
    def predict(self, window, state):
        from aeromill.features import extract_features, residual_rms

        profile = ToolProfile()
        return Detection(
            extract_features(window, state, profile),
            0.0,
            residual_rms=residual_rms(window, state, profile),
        )


def test_always_calm_real_chatter_is_missed(tmp_path):
    e = Engine(
        BASE_SCENARIOS["recover_a"],
        mode="ml_agent",
        detector=CalmDetector(),
        log_path=tmp_path / "calm.jsonl",
    ).run()
    result = e.evaluate()
    assert e.agent.state == "COMPLETED" and e.agent.incident == 0
    assert (
        result["incidents"] == 1
        and result["missed_incidents"] == 1
        and not result["run_success"]
    )


def test_three_of_sixty_false_windows_trigger_failed_run(tmp_path):
    class ThreeFalse(CalmDetector):
        calls = 0

        def predict(self, window, state):
            d = super().predict(window, state)
            self.calls += 1
            return replace(d, chatter_score=float(28 <= self.calls <= 30))

    detector = ThreeFalse()
    e = Engine(
        BASE_SCENARIOS["stable"],
        mode="ml_agent",
        detector=detector,
        log_path=tmp_path / "false.jsonl",
    )
    for _ in range(62):
        e.tick()
    assert detector.calls == 60
    assert (
        sum(r["event"] == "incident" for r in e.events) == 1
        and e.controller.action_count == 1
    )
    e.run()
    result = e.evaluate()
    assert (
        result["incidents"] == 0
        and result["false_interventions"] == 1
        and not result["run_success"]
    )


def test_repeat_c_truth_confirmations_and_noop_memory(tmp_path):
    # This observation-only RMS detector is correct for the fixed no-engagement geometry;
    # residual energy and real telemetry still drive the ML verification path.
    class CorrectDetector(CalmDetector):
        def predict(self, window, state):
            d = super().predict(window, state)
            return replace(d, chatter_score=float(max(d.features[:21:7]) > 1.5))

    e = Engine(
        BASE_SCENARIOS["repeat_after_c"],
        mode="ml_agent",
        detector=CorrectDetector(),
        log_path=tmp_path / "repeat.jsonl",
    ).run()
    result = e.evaluate()
    assert result["run_success"] and result["incidents"] == 2
    assert result["false_recoveries"] == result["unverified_recoveries"] == 0
    commands = [r for r in e.events if r["event"] == "set_cutting_parameters"]
    assert [(c["rpm"], c["feed_mm_min"]) for c in commands] == [
        (3520, 960),
        (2880, 960),
        (3840, 720),
        (3520, 960),
    ]
    confirmations = [i["confirmed_recovery_s"] for i in result["incident_details"]]
    samples = np.concatenate([c["time_s"] for c in e._evaluator.chunks])
    positions = np.concatenate([c["x_mm"] for c in e._evaluator.chunks])
    xs = [float(positions[np.searchsorted(samples, t)]) for t in confirmations]
    assert xs[0] < 400 and xs[1] < 600
    plans = [r for r in e.events if r["event"] == "planning_record"]
    assert plans[-1]["memory"] == "hit"
    assert plans[-1]["ranked"][0] == {"pair": [3840.0, 720.0], "reason": "current pair"}
    print(json.dumps(dict(confirmations_s=confirmations, confirmation_x_mm=xs)))


def test_recover_a_ten_consecutive_distinct_seeds(tmp_path):
    detector = RFDetector(ToolProfile())
    for seed in range(420, 430):
        e = Engine(
            BASE_SCENARIOS["recover_a"],
            mode="ml_agent",
            seed=seed,
            detector=RFDetector(ToolProfile(), model=detector.model),
            log_path=tmp_path / f"{seed}.jsonl",
        ).run()
        r = e.evaluate()
        assert (
            r["run_success"]
            and r["false_recoveries"] == r["unverified_recoveries"] == 0
        )
        assert r["rms_reduction"] >= 0.4


@pytest.mark.parametrize(
    "state", ["WARMUP", "MONITOR", "PLAN", "APPLY", "SETTLE", "VERIFY"]
)
@pytest.mark.parametrize("reason", ["MANUAL_STOP", "DATA_QUALITY"])
def test_ml_stop_error_every_state_and_late_ack(tmp_path, state, reason):
    e = Engine(
        BASE_SCENARIOS["recover_a"],
        mode="ml_agent",
        detector=CalmDetector(),
        log_path=tmp_path / "stop.jsonl",
    )
    e.tick()
    e.agent.state = state
    cmd = Command(e.run_id, "pending", "set", 3520.0, 960.0, "test", 1)
    e.queued = cmd
    before = e.simulator._x
    if reason == "MANUAL_STOP":
        e.tick(stop=True)
    else:
        e.stop(reason)
    assert e.done and e.agent.state == "HOLD" and e.agent.reason == reason
    assert e.controller.applied.feed_mm_min == 0 and e.controller.pending is None
    assert e.simulator._x == before and e.queued is None
    assert (
        e.controller.acknowledge(
            ControllerEvent(cmd.command_id, "accepted", e.time_s, e.run_id)
        )
        is None
    )
    assert e.controller.apply(cmd).status == "rejected"
    assert e.tick() is None


def test_ml_half_score_fails_at_exact_verify_deadline():
    a = make_agent(mode="ml_agent")
    settled(a)
    for k in range(39, 53):
        a.tick(observation(k, score=0.5, rpm=3520, feed=960))
        assert a.state == "VERIFY"
    a.tick(observation(53, score=0.5, rpm=3520, feed=960))
    assert a.state == "APPLY"
    outcomes = [r for r in a.events if r["event"] == "record_outcome"]
    assert (
        len(outcomes) == 1
        and outcomes[0]["time_s"] == 5.3
        and not outcomes[0]["success"]
    )


def test_three_incidents_after_a_use_absolute_pairs(tmp_path):
    scenario = Scenario(
        "three",
        (
            Region(80, 220, (3200.0,), (120.0,)),
            Region(260, 400, (3520.0,), (120.0,)),
            Region(460, 600, (2880.0,), (120.0,)),
        ),
        path_length_mm=600,
    )
    e = Engine(scenario, mode="ml_agent", log_path=tmp_path / "three.jsonl").run()
    assert e.evaluate()["run_success"] and e.evaluate()["incidents"] == 3
    pairs = [
        (r["rpm"], r["feed_mm_min"])
        for r in e.events
        if r["event"] == "set_cutting_parameters"
    ]
    assert pairs == [(3520.0, 960.0), (2880.0, 960.0), (3520.0, 960.0)]
