"""Post-run grading from sampled hidden truth; never calls a detector.

Claim grades accept timestamps only. Event timestamps are used separately for
latency/action metrics, never to construct ground-truth incidents.
"""

import numpy as np

from .config import SAMPLE_RATE


def grade_claim(
    time_s: float, data: dict, *, completed_path_mm: float | None = None
) -> str:
    t = data["time_s"]
    lo, hi = np.searchsorted(t, [time_s, time_s + 3], side="right")
    # Fixed sample clock: require every sample in the following three seconds.
    expected = (
        np.arange(int(time_s * SAMPLE_RATE) + 1, int((time_s + 3) * SAMPLE_RATE) + 1)
        / SAMPLE_RATE
    )
    arrival = (
        len(t)
        if completed_path_mm is None
        else int(np.searchsorted(data["x_mm"], completed_path_mm))
    )
    early_endpoint = arrival < len(t) and t[arrival] < time_s + 3
    moving_hi = min(hi, arrival) if early_endpoint else hi
    x = data["x_mm"][max(0, lo - 1) : moving_hi]
    if (
        not np.all(data["envelope"][lo:hi] <= 0.25)
        or not np.all(data["moving"][lo:moving_hi])
        or not np.all(data["feed_mm_min"][lo:moving_hi] > 0)
        or not np.all(np.diff(x) > 0)
    ):
        return "false_recovery"
    if early_endpoint or not np.array_equal(t[lo:hi], expected):
        return "unverified_recovery"
    return "confirmed_recovery"


def truth_incidents(data: dict) -> list[dict]:
    t, a = data["time_s"], data["envelope"]
    result = []
    active = None
    stable_samples = 0
    for i in range(len(t)):
        if active is None and a[i] >= 0.75:
            active = dict(start_s=float(t[i]), end_s=None)
            result.append(active)
        if active is not None:
            stable_samples = stable_samples + 1 if a[i] <= 0.25 else 0
            if stable_samples >= 3 * SAMPLE_RATE:
                active["end_s"] = float(t[i])
                active = None
                stable_samples = 0
    return result


def evaluate_run(
    data: dict, events: list[dict], *, state: str, path_length_mm: float
) -> dict:
    """Truth incidents define the denominator, including those never detected."""
    t = data["time_s"]
    incidents = truth_incidents(data)
    times = lambda name: [float(e["time_s"]) for e in events if e["event"] == name]
    detections = times("incident")
    commands = times("set_cutting_parameters")
    claims = [
        dict(
            time_s=c,
            grade=grade_claim(
                c,
                data,
                completed_path_mm=path_length_mm if state == "COMPLETED" else None,
            ),
        )
        for c in times("recovery_claim")
    ]
    matched = set()
    for index, incident in enumerate(incidents):
        start = incident["start_s"]
        end = (
            incident["end_s"]
            if incident["end_s"] is not None
            else (float(t[-1]) if len(t) else 0.0)
        )
        detected = [(j, d) for j, d in enumerate(detections) if start <= d <= end]
        matched.update(j for j, _ in detected)
        detection = min((d for _, d in detected), default=None)
        next_start = (
            incidents[index + 1]["start_s"]
            if index + 1 < len(incidents)
            else float("inf")
        )
        confirmations = [
            c["time_s"] + 3
            for c in claims
            if start <= c["time_s"] < next_start
            and c["grade"] == "confirmed_recovery"
            and detection is not None
            and detection <= c["time_s"]
        ]
        confirmation = min(confirmations, default=None)
        incident.update(
            detection_s=detection,
            detection_latency_s=None if detection is None else detection - start,
            confirmed_recovery_s=confirmation,
            stabilization_time_s=None if confirmation is None else confirmation - start,
        )
    false_commands = 0
    for c in commands:
        mask = (t > c - 0.25) & (t <= c)
        false_commands += int(mask.any() and np.all(data["envelope"][mask] <= 0.25))
    completed = bool(
        state == "COMPLETED" and len(t) and data["x_mm"][-1] >= path_length_mm
    )
    stable_ok = bool(not incidents and completed and not commands and state != "HOLD")
    false = sum(c["grade"] == "false_recovery" for c in claims)
    unverified = sum(c["grade"] == "unverified_recovery" for c in claims)
    success = bool(
        completed
        and not false
        and not unverified
        and all(
            i["detection_s"] is not None and i["confirmed_recovery_s"] is not None
            for i in incidents
        )
        and (bool(incidents) or stable_ok)
    )
    duration = float(t[-1]) if len(t) else 0.0

    def rms(start, end):
        mask = (t > start) & (t <= end)
        if mask.sum() < SAMPLE_RATE or end > duration:
            return None
        v = data["vibration"][mask]
        return (
            [float(v) for v in np.sqrt(np.mean(v * v, axis=0))]
            if np.isfinite(v).all()
            else None
        )

    before = rms(commands[0] - 1, commands[0]) if commands else None
    confirmed_times = [
        c["time_s"] + 3 for c in claims if c["grade"] == "confirmed_recovery"
    ]
    after = (
        rms(min(confirmed_times), min(confirmed_times) + 1) if confirmed_times else None
    )
    reduction = None if before is None or after is None else 1 - after[0] / before[0]
    false_incidents = len(detections) - len(matched)
    return dict(
        state=state,
        completed=completed,
        run_success=success,
        incidents=len(incidents),
        missed_incidents=sum(i["detection_s"] is None for i in incidents),
        detected_within_1s=sum(
            i["detection_latency_s"] is not None and i["detection_latency_s"] <= 1
            for i in incidents
        ),
        incident_details=incidents,
        claims=claims,
        confirmed_recoveries=sum(c["grade"] == "confirmed_recovery" for c in claims),
        false_recoveries=false,
        unverified_recoveries=unverified,
        unstable_time_s=float(np.count_nonzero(data["envelope"] >= 0.75) / SAMPLE_RATE),
        commands=len(commands),
        false_interventions=false_commands,
        false_incidents=false_incidents,
        false_incidents_per_1000_s=1000 * false_incidents / duration
        if duration
        else 0.0,
        stops=int(state == "HOLD"),
        stable_without_intervention=stable_ok,
        time_s=duration,
        traversal_time_s=duration if completed else None,
        rms_before=before,
        rms_after=after,
        rms_reduction=reduction,
    )


class RunEvaluator:
    """Private side-channel collector; the engine never forwards this to Agent."""

    def __init__(self):
        self.chunks = []
        self.samples = 0

    def append(self, step):
        truth, telemetry = step.truth, step.telemetry
        n = len(truth.envelope)
        # Truth sampling remains valid even when a telemetry fault is injected.
        start = self.samples
        self.samples += n

        def observed(name, shape):
            value = getattr(telemetry, name, None)
            if (
                not isinstance(value, np.ndarray)
                or value.shape != shape
                or value.dtype.kind not in "fiu"
            ):
                return np.full(shape, np.nan)
            return value.copy()

        self.chunks.append(
            dict(
                time_s=np.arange(start + 1, start + n + 1) / SAMPLE_RATE,
                envelope=truth.envelope.copy(),
                x_mm=truth.x_mm.copy(),
                moving=truth.moving.copy(),
                feed_mm_min=observed("feed_mm_min", (n,)),
                vibration=observed("vibration", (n, 3)),
            )
        )

    def evaluate(self, events, *, state, path_length_mm):
        names = ("time_s", "envelope", "x_mm", "moving", "feed_mm_min", "vibration")
        data = {
            k: np.concatenate([c[k] for c in self.chunks])
            if self.chunks
            else np.empty((0, 3) if k == "vibration" else (0,))
            for k in names
        }
        return evaluate_run(data, events, state=state, path_length_mm=path_length_mm)


def evaluate_validation(**kwargs):
    """CLI entry; defer orchestration imports so grading has no detector dependency."""
    from .validation import evaluate_validation as run

    return run(**kwargs)
