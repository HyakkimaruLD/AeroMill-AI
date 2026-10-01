"""Independent expected outcomes: removing any truth/motion/horizon check must fail."""

import importlib
import numpy as np
import pytest


def evaluator():
    m = importlib.import_module("aeromill.evaluation")
    assert hasattr(m, "evaluate_run"), "independent evaluator missing"
    return m


def trace(seconds=10):
    t = np.arange(1, int(seconds * 8192) + 1) / 8192
    return dict(
        time_s=t,
        envelope=np.full(len(t), 0.1),
        x_mm=t * 20,
        moving=np.ones(len(t), dtype=bool),
        feed_mm_min=np.full(len(t), 1200.0),
        vibration=np.ones((len(t), 3)),
    )


def event(name, time):
    return dict(event=name, time_s=time)


def test_hidden_incident_cannot_be_hidden_by_calm_agent():
    m = evaluator()
    data = trace()
    data["envelope"][8192 : 4 * 8192] = 1.0
    result = m.evaluate_run(data, [], state="COMPLETED", path_length_mm=200.0)
    assert result["incidents"] == 1 and result["missed_incidents"] == 1
    assert result["unstable_time_s"] == 3.0
    assert not result["run_success"]


@pytest.mark.parametrize(
    "fault,grade",
    [
        (None, "confirmed_recovery"),
        ("amplitude", "false_recovery"),
        ("moving", "false_recovery"),
        ("feed", "false_recovery"),
        ("position", "false_recovery"),
        ("short", "unverified_recovery"),
        ("gap", "unverified_recovery"),
    ],
)
def test_claim_uses_all_samples_and_full_horizon(fault, grade):
    m = evaluator()
    data = trace(4 if fault != "short" else 3.9)
    if fault == "amplitude":
        data["envelope"][17000] = 0.251
    if fault == "moving":
        data["moving"][17000] = False
    if fault == "feed":
        data["feed_mm_min"][17000] = 0
    if fault == "position":
        data["x_mm"][17000] = data["x_mm"][16999]
    if fault == "gap":
        data = {k: np.delete(v, 17000, axis=0) for k, v in data.items()}
    assert m.grade_claim(1.0, data) == grade


def test_brief_dip_does_not_split_incidents_and_claims_match_truth():
    m = evaluator()
    data = trace(15)
    t = data["time_s"]
    data["envelope"][(t > 1) & (t <= 2) | (t > 3) & (t <= 4) | (t > 9) & (t <= 10)] = (
        1.0
    )
    events = [
        event("incident", 1.4),
        event("recovery_claim", 4.5),
        event("incident", 9.4),
    ]
    result = m.evaluate_run(data, events, state="COMPLETED", path_length_mm=300)
    assert result["incidents"] == 2 and result["missed_incidents"] == 0
    assert result["confirmed_recoveries"] == 1 and not result["run_success"]
    assert result["incident_details"][0]["end_s"] == pytest.approx(7.0)
    assert result["incident_details"][1]["confirmed_recovery_s"] is None
    events.append(event("recovery_claim", 10.5))
    result = m.evaluate_run(data, events, state="COMPLETED", path_length_mm=300)
    assert result["run_success"] and result["detected_within_1s"] == 2


def test_agent_labels_and_scores_cannot_change_grading():
    m = evaluator()
    data = trace()
    data["envelope"][8192 : 3 * 8192] = 1.0
    events = [
        event("incident", 1.3),
        event("set_cutting_parameters", 1.3),
        event("recovery_claim", 3.5),
    ]
    clean = m.evaluate_run(data, events, state="COMPLETED", path_length_mm=200)
    polluted = [
        dict(
            e,
            score=1.0,
            success=False,
            incident_id=999,
            confirmed=True,
            features=["lie"],
        )
        for e in events
    ]
    polluted += [dict(event="record_outcome", time_s=3.5, success=False)]
    assert (
        m.evaluate_run(data, polluted, state="COMPLETED", path_length_mm=200) == clean
    )
    assert clean["run_success"]


def test_three_false_windows_is_not_run_success():
    m = evaluator()
    data = trace()
    events = [
        event("incident", 3.0),
        event("set_cutting_parameters", 3.0),
        event("recovery_claim", 5.0),
    ]
    result = m.evaluate_run(data, events, state="COMPLETED", path_length_mm=200)
    assert 3 / 60 == 0.05
    assert result["false_interventions"] == 1 and result["false_incidents"] == 1
    assert not result["run_success"] and not result["stable_without_intervention"]


def test_stop_and_early_endpoint_never_success():
    m = evaluator()
    data = trace()
    for state, path in [("HOLD", 200), ("COMPLETED", 201)]:
        assert not m.evaluate_run(data, [], state=state, path_length_mm=path)[
            "run_success"
        ]


def test_claim_on_fractional_sample_boundary():
    assert evaluator().grade_claim(0.3, trace(4)) == "confirmed_recovery"


def test_confirmed_claim_without_detection_does_not_credit_incident():
    m = evaluator()
    data = trace()
    data["envelope"][8192 : 3 * 8192] = 1.0
    r = m.evaluate_run(
        data, [event("recovery_claim", 3.5)], state="COMPLETED", path_length_mm=200
    )
    assert r["claims"][0]["grade"] == "confirmed_recovery"
    assert (
        r["missed_incidents"] == 1
        and r["incident_details"][0]["confirmed_recovery_s"] is None
    )
    assert not r["run_success"]


def test_observed_failure_outweighs_short_horizon():
    data = trace(2)
    data["feed_mm_min"][-1] = 0
    assert evaluator().grade_claim(1.0, data) == "false_recovery"


def test_collector_detaches_truth_from_returned_steps():
    from aeromill.simulator import Simulator
    from aeromill.scenarios import BASE_SCENARIOS
    from aeromill.contracts import AppliedState

    m = evaluator()
    collector = m.RunEvaluator()
    step = Simulator(BASE_SCENARIOS["stable"]).step(0.1, AppliedState())
    collector.append(step)
    before = collector.evaluate([], state="HOLD", path_length_mm=400)
    step.truth.envelope[:] = 9
    step.telemetry.vibration[:] = 9
    assert collector.evaluate([], state="HOLD", path_length_mm=400) == before


def test_real_endpoint_tail_is_unverified_not_false():
    from dataclasses import replace
    from aeromill.simulator import Simulator
    from aeromill.scenarios import BASE_SCENARIOS
    from aeromill.contracts import AppliedState

    collector = evaluator().RunEvaluator()
    sim = Simulator(replace(BASE_SCENARIOS["stable"], path_length_mm=39.0))
    for _ in range(20):
        collector.append(sim.step(0.1, AppliedState()))
    result = collector.evaluate(
        [event("recovery_claim", 1.0)], state="COMPLETED", path_length_mm=39.0
    )
    assert result["unverified_recoveries"] == 1 and result["false_recoveries"] == 0
    # Actual chatter before the endpoint still invalidates the claim.
    collector.chunks[12]["envelope"][0] = 1.0
    result = collector.evaluate(
        [event("recovery_claim", 1.0)], state="COMPLETED", path_length_mm=39.0
    )
    assert result["false_recoveries"] == 1
