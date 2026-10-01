"""Viewer-side explanations and maps. Nothing here belongs in an Observation."""

import numpy as np

from .config import FEED_LIMITS, RPM_LIMITS
from .scenarios import NAMED_SCENARIOS, Region, Scenario
from .simulator import resonance

# Name, controller/telemetry fault, expected safe-stop reason.
FAILURES = {
    "nan_telemetry": ("nan", "DATA_QUALITY"),
    "missing_telemetry": ("missing", "DATA_QUALITY"),
    "out_of_order": ("out_of_order", "DATA_QUALITY"),
    "rejected_command": ("reject", "COMMAND_REJECTED"),
    "lost_acknowledgment": ("lost_ack", "ACK_TIMEOUT"),
    "manual_stop": (None, "MANUAL_STOP"),
    "startup_chatter": (None, "CALIBRATION_FAILED"),
}


def scenario_for(name):
    if name in NAMED_SCENARIOS:
        return NAMED_SCENARIOS[name]
    if name == "startup_chatter":
        return Scenario(name, (Region(0.0, 400.0, (3200.0,), (120.0,)),))
    if name in FAILURES:
        return NAMED_SCENARIOS["recover_a"]
    raise ValueError("unknown scenario")


def stability_map(part_or_scenario, feed, rpm_grid, x_grid):
    """Static cutting target; zero feed uses the simulator's Stop target."""
    scenario = getattr(part_or_scenario, "scenario", part_or_scenario)
    if isinstance(scenario, str):
        scenario = scenario_for(scenario)
    rpm = np.asarray(rpm_grid, dtype=np.float64)
    x = np.asarray(x_grid, dtype=np.float64)
    if (
        not np.isfinite(feed)
        or not (feed == 0 or FEED_LIMITS[0] <= feed <= FEED_LIMITS[1])
        or rpm.ndim != 1
        or x.ndim != 1
        or not np.isfinite(rpm).all()
        or not np.isfinite(x).all()
        or np.any((rpm < RPM_LIMITS[0]) | (rpm > RPM_LIMITS[1]))
        or np.any((x < 0) | (x > scenario.path_length_mm))
    ):
        raise ValueError("map feed/RPM/position outside operating range")
    values = np.zeros((len(rpm), len(x)), dtype=np.float64)
    for j, position in enumerate(x):
        region = scenario.region_at(float(position))
        if region is not None:
            for i, speed in enumerate(rpm):
                values[i, j] = region.gain * resonance(
                    float(speed), region.centers, region.widths
                )
    return dict(
        rpm=rpm.copy(),
        x_mm=x.copy(),
        gain_resonance=values,
        A_target=np.zeros_like(values)
        if feed == 0
        else 0.1 + 4 * values * (feed / 1200),
    )


def trajectory(snapshot):
    return dict(x_mm=snapshot.series["x"].copy(), rpm=snapshot.series["rpm"].copy())


def catalog():
    descriptions = {
        "stable": (
            "Stable cut",
            "No resonance zone.",
            "Finish without corrective commands.",
        ),
        "recover_a": (
            "First correction works",
            "A resonance band at the starting speed.",
            "Try A and verify recovery.",
        ),
        "retry_b": (
            "Try again",
            "Both the starting speed and A resonate.",
            "Reject A after verification, then try B.",
        ),
        "recover_c": (
            "Third candidate",
            "The starting speed, A and B resonate.",
            "Try C after A and B fail.",
        ),
        "unrecoverable": (
            "No safe candidate",
            "A wide resonance band covers all candidates.",
            "Stop with ATTEMPTS_EXHAUSTED.",
        ),
        "repeat_after_c": (
            "A second problem zone",
            "A later zone resonates at C after the first recovery.",
            "Detect the second incident and verify a new correction.",
        ),
        "stable_engagement": (
            "Louder, still stable",
            "Engagement doubles the tooth vibration without chatter.",
            "Finish without a false intervention in ml_agent.",
        ),
        "nan_telemetry": (
            "Invalid sensor sample",
            "One vibration sample becomes NaN at 1 s.",
            "Stop with DATA_QUALITY.",
        ),
        "missing_telemetry": (
            "Sensor dropout",
            "The telemetry chunk at 1 s is missing.",
            "Stop with DATA_QUALITY.",
        ),
        "out_of_order": (
            "Stale sensor packet",
            "The sequence number at 1 s goes backward.",
            "Stop with DATA_QUALITY.",
        ),
        "rejected_command": (
            "Controller refuses",
            "The controller rejects the first correction.",
            "Stop with COMMAND_REJECTED.",
        ),
        "lost_acknowledgment": (
            "No acknowledgment",
            "The first correction has no reached acknowledgment.",
            "Stop with ACK_TIMEOUT.",
        ),
        "manual_stop": (
            "Operator Stop",
            "The operator requests Stop at 1 s.",
            "Stop with MANUAL_STOP.",
        ),
        "startup_chatter": (
            "Unsafe calibration",
            "Chatter starts immediately, before a stable calibration.",
            "Stop with CALIBRATION_FAILED.",
        ),
    }
    return [
        dict(
            name=name,
            title=title,
            hides=hides,
            expect=expect,
            evaluator_checks=(
                "HOLD with " + FAILURES[name][1] + ", zero feed, and no success claim."
                if name in FAILURES
                else "Independent envelope, moving-cut recovery horizon, missed chatter and unnecessary commands."
            ),
            run=dict(scenario=name, seed=42, mode="ml_agent"),
        )
        for name, (title, hides, expect) in descriptions.items()
    ]
