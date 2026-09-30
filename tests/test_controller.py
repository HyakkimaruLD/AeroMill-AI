"""Break targets: clamp/zero bypass, overlapping dispatch, early reach, replay, unlatch."""

from dataclasses import replace

import numpy as np
import pytest

from aeromill import controller, contracts
from aeromill.scenarios import BASE_SCENARIOS
from aeromill.simulator import Simulator


def command(id="c1", rpm=3520.0, feed=960.0, run="r", kind="set"):
    return contracts.Command(run, id, kind, rpm, feed, "test", 1)


def test_controller_api():
    assert hasattr(controller, "Controller")
    assert hasattr(contracts, "Command")


@pytest.mark.parametrize(
    "rpm,feed",
    [
        (2399, 960),
        (4201, 960),
        (3200, 599),
        (3200, 1201),
        (3200, 0),
        (float("nan"), 960),
        (3200, float("inf")),
    ],
)
def test_reject_not_clamp(rpm, feed):
    c = controller.Controller("r")
    result = c.apply(command(rpm=rpm, feed=feed))
    assert result.status == "rejected" and result.reason
    assert c.applied == contracts.AppliedState()


def test_reach_requires_both_actual_errors_and_one_inflight():
    c = controller.Controller("r")
    assert c.apply(command()).status == "accepted"
    assert c.applied == contracts.AppliedState(3520, 960)
    assert c.apply(command("c2")).status == "rejected"
    t = (
        Simulator(BASE_SCENARIOS["stable"], run_id="r")
        .step(0.1, contracts.AppliedState())
        .telemetry
    )
    for rpm, feed in [(3518.99, 960), (3520, 961.01)]:
        assert (
            c.observe(
                replace(
                    t,
                    rpm=np.full_like(t.rpm, rpm),
                    feed_mm_min=np.full_like(t.feed_mm_min, feed),
                )
            )
            == ()
        )
    near = replace(
        t, rpm=np.full_like(t.rpm, 3519), feed_mm_min=np.full_like(t.feed_mm_min, 961)
    )
    assert c.observe(replace(near, run_id="old")) == ()
    events = c.observe(near)
    assert len(events) == 1 and events[0].status == "reached"
    assert events[0].run_id == "r"
    assert c.observe(near) == ()


def test_duplicate_and_priority_stop_ignore_late_ack():
    c = controller.Controller("r")
    first = c.apply(command())
    assert c.apply(command()) == first
    assert c.action_count == 1
    assert c.apply(command("old", run="old")).status == "ignored"
    c.apply(command("stop", kind="stop"))
    assert c.applied.stopped and c.applied.feed_mm_min == 0
    assert c.acknowledge(contracts.ControllerEvent("c1", "accepted", 0.2, "r")) is None
    c.apply(command())
    assert c.apply(command("new")).status == "rejected"
    assert c.applied.stopped and c.applied.feed_mm_min == 0


def test_fault_injection():
    assert (
        controller.Controller("r", fault="reject").apply(command()).status == "rejected"
    )
    c = controller.Controller("r", fault="lost_ack")
    assert c.apply(command()).status == "pending"
    assert c.applied.rpm == 3520
