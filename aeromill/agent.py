"""One bounded observation-only state machine for all control modes."""

from collections import deque
from typing import Callable

import numpy as np

from .config import CANDIDATES, CANDIDATE_SET_VERSION, ToolProfile
from .contracts import Command, Observation
from .controller import admissible
from .memory import Memory


def set_cutting_parameters(agent: "Agent", rpm: float, feed: float) -> Command:
    command = Command(
        agent.run_id,
        f"{agent.run_id}:{agent.incident}:{agent.attempts}",
        "set",
        rpm,
        feed,
        "sustained chatter",
        agent.incident,
    )
    agent.emit(
        "set_cutting_parameters",
        command_id=command.command_id,
        rpm=rpm,
        feed_mm_min=feed,
        incident_id=agent.incident,
    )
    return command


def feed_hold(agent: "Agent", reason: str) -> Command:
    command = Command(
        agent.run_id, f"{agent.run_id}:stop", "stop", 0.0, 0.0, reason, agent.incident
    )
    agent.emit("feed_hold", command_id=command.command_id, reason=reason)
    return command


def record_outcome(agent: "Agent", success: bool, final_features: tuple) -> None:
    agent.emit(
        "record_outcome",
        command_id=agent.command.command_id,
        success=success,
        initial_features=agent.initial_features,
        final_features=final_features,
    )
    if success:
        agent.emit(
            "recovery_claim",
            command_id=agent.command.command_id,
            incident_id=agent.incident,
        )
        agent.memory.remember(
            agent.command_memory_key,
            (agent.command.rpm, agent.command.feed_mm_min),
            agent.command.command_id,
            agent.now / 10,
        )
        agent.emit("memory_write", pair=[agent.command.rpm, agent.command.feed_mm_min])
    elif agent.reusing_memory:
        agent.memory.forget(agent.command_memory_key)
        agent.emit("memory_delete", pair=[agent.command.rpm, agent.command.feed_mm_min])
    agent.reusing_memory = False


class Agent:
    def __init__(
        self,
        run_id: str,
        profile: ToolProfile,
        *,
        mode: str = "threshold_search",
        path_length_mm: float = 400.0,
        log: Callable[[dict], None] | None = None,
        memory: Memory | None = None,
        model_version: str = "rms-v1",
        candidate_set_version: str = CANDIDATE_SET_VERSION,
    ):
        if mode not in ("no_adaptation", "baseline", "threshold_search", "ml_agent"):
            raise ValueError("unknown mode")
        self.run_id, self.profile, self.mode = run_id, profile, mode
        self.path_length_mm, self.log = path_length_mm, log
        self.state, self.reason = "WARMUP", ""
        self.events: list[dict] = []
        self.now = self.entered = 0
        self.calibration_rms = None
        self.calibration_residual_rms = None
        self.residual_calibration = []
        self.calibration: list[np.ndarray] = []
        self.calibration_start = None
        self.counter = self.incident = self.attempts = 0
        self.tried: set[tuple[float, float]] = set()
        self.candidates = (
            ((3200.0, 720.0),)
            if mode == "baseline"
            else tuple((v.rpm, v.feed_mm_min) for v in CANDIDATES.values())
        )
        self.command: Command | None = None
        self.ack_tick = self.reached_tick = None
        self.verify = deque(maxlen=10)
        self.previous_x = 0.0
        self.initial_features = ()
        self.memory = memory if memory is not None else Memory()
        self.model_version, self.candidate_set_version = (
            model_version,
            candidate_set_version,
        )
        self.zone_class = "nominal"
        self.command_memory_key = None
        self.reusing_memory = False

    def emit(self, event: str, **fields) -> None:
        row = dict(event=event, run_id=self.run_id, time_s=self.now / 10, **fields)
        self.events.append(row)
        if self.log is not None:
            self.log(row)

    def transition(self, state: str) -> None:
        self.emit("transition", previous=self.state, state=state)
        self.state, self.entered = state, self.now

    def hold(self, reason: str) -> Command:
        if self.reusing_memory and self.state in ("APPLY", "SETTLE", "VERIFY"):
            self.memory.forget(self.command_memory_key)
            self.emit("memory_delete", reason=reason)
            self.reusing_memory = False
        self.reason = reason
        self.transition("HOLD")
        return feed_hold(self, reason)

    def plan(self, rpm: float, feed: float) -> Command:
        self.transition("PLAN")
        ranked = []
        seen = set()
        key = (
            self.profile.profile_id,
            self.zone_class,
            self.model_version,
            self.candidate_set_version,
        )
        entry = (
            self.memory.get(key)
            if self.mode in ("ml_agent", "threshold_search")
            else None
        )
        self.emit("memory_hit" if entry else "memory_miss", key=list(key))
        pairs = ((entry.pair,) if entry else ()) + self.candidates
        for pair in pairs:
            reason = (
                "duplicate"
                if pair in seen
                else "already tried"
                if pair in self.tried
                else "outside the limits"
                if not admissible(*pair)
                else "current pair"
                if abs(pair[0] - rpm) <= 1 and abs(pair[1] - feed) <= 1
                else "outside candidate set"
                if pair not in self.candidates
                else "kept"
            )
            ranked.append(dict(pair=list(pair), reason=reason))
            seen.add(pair)
        chosen = next((r["pair"] for r in ranked if r["reason"] == "kept"), None)
        self.emit(
            "planning_record",
            ranked=ranked,
            memory="hit" if entry else "miss",
            excluded=[r for r in ranked if r["reason"] != "kept"],
            chosen=chosen,
        )
        if chosen is None:
            return self.hold("NO_CANDIDATE")
        self.tried.add(tuple(chosen))
        self.command_memory_key = key
        self.reusing_memory = entry is not None and tuple(chosen) == entry.pair
        self.attempts += 1
        self.command = set_cutting_parameters(self, *chosen)
        self.ack_tick = self.reached_tick = None
        self.transition("APPLY")
        return self.command

    def warmup_calm(self, detection, rms):
        if detection is None or detection.chatter_score > 0.3:
            return False
        if self.mode == "ml_agent" and (
            detection.residual_rms is None
            or not np.all(np.asarray(detection.residual_rms) > 0)
        ):
            return False
        return np.all(rms > 0) and np.all(rms / self.profile.reference_rms <= 1.5)

    def calibrate(self, calm, rms, detection):
        residual = detection.residual_rms if detection else None
        if not calm:
            self.calibration.clear()
            self.residual_calibration.clear()
            self.calibration_start = None
            return
        if self.calibration_start is None:
            self.calibration_start = self.now
        self.calibration.append(rms)
        if self.mode == "ml_agent":
            self.residual_calibration.append(residual)
        if self.now - self.calibration_start >= 20:
            self.calibration_rms = np.median(self.calibration, axis=0)
            if self.mode == "ml_agent":
                self.calibration_residual_rms = np.median(
                    self.residual_calibration, axis=0
                )
            self.transition("MONITOR")

    def verification_calm(self, detection, rms, advancing):
        if detection is None or detection.chatter_score > 0.3 or not advancing:
            return False
        if self.mode == "ml_agent":
            return detection.residual_rms is not None and np.all(
                np.asarray(detection.residual_rms) / self.calibration_residual_rms
                <= 2.0
            )
        return np.all(rms / self.calibration_rms <= 1.5)

    def tick(
        self, observation: Observation, *, error: str | None = None
    ) -> Command | None:
        if self.state in ("HOLD", "COMPLETED"):
            return None
        telemetry, detection = observation.telemetry, observation.detection
        self.zone_class = telemetry.zone_class[-1]
        self.now = telemetry.sequence + 1
        if error:
            return self.hold(error)
        events = tuple(
            e
            for e in observation.controller_events
            if e.run_id == self.run_id
            and self.command is not None
            and e.command_id == self.command.command_id
        )
        if any(e.status == "rejected" for e in events):
            return self.hold("COMMAND_REJECTED")
        score = detection.chatter_score if detection else None
        rms = (
            np.asarray(detection.features[:21:7]) * self.profile.reference_rms
            if detection
            else None
        )
        calm = self.warmup_calm(detection, rms)
        # Deadline errors outrank a coincident endpoint; an on-time ACK/reached wins.
        if (
            self.state == "APPLY"
            and self.now - self.entered >= 2
            and not any(e.status == "accepted" for e in events)
        ):
            return self.hold("ACK_TIMEOUT")
        if (
            self.state == "SETTLE"
            and self.reached_tick is None
            and self.now - self.ack_tick >= 20
            and not any(e.status == "reached" for e in events)
        ):
            return self.hold("SETPOINT_TIMEOUT")
        if (
            self.state == "WARMUP"
            and self.now >= 50
            and not (
                calm
                and self.calibration_start is not None
                and self.now - self.calibration_start >= 20
            )
        ):
            return self.hold("CALIBRATION_FAILED")
        x = float(telemetry.xyz_mm[-1, 0])
        advancing = x > self.previous_x and telemetry.feed_mm_min[-1] > 0
        self.previous_x = x
        if x >= self.path_length_mm:
            self.transition("COMPLETED")
            return None
        if self.now >= 650:
            return self.hold("RUN_TIMEOUT")
        rpm, feed = float(telemetry.rpm[-1]), float(telemetry.feed_mm_min[-1])
        if self.state == "WARMUP":
            self.calibrate(calm, rms, detection)
        elif self.state == "MONITOR":
            if self.mode == "no_adaptation":
                return None
            self.counter = self.counter + 1 if score is not None and score >= 0.8 else 0
            if self.counter >= 3:
                self.incident += 1
                self.attempts = 0
                self.tried.clear()
                self.initial_features = detection.features
                self.emit("incident", incident_id=self.incident)
                return self.plan(rpm, feed)
        elif self.state == "APPLY":
            if any(e.status == "accepted" for e in events):
                self.ack_tick = self.now
                self.transition("SETTLE")
        if self.state == "SETTLE":
            if self.reached_tick is None:
                if any(e.status == "reached" for e in events):
                    self.reached_tick = self.now
            if self.reached_tick is not None and self.now - self.reached_tick >= 10:
                self.verify.clear()
                self.transition("VERIFY")
        elif self.state == "VERIFY":
            calm = self.verification_calm(detection, rms, advancing)
            self.verify.append(bool(calm))
            if self.now - self.entered >= 15:
                success = len(self.verify) == 10 and all(self.verify)
                record_outcome(self, success, detection.features if detection else ())
                self.counter = 0
                if success:
                    self.transition("MONITOR")
                elif self.attempts >= (1 if self.mode == "baseline" else 3):
                    return self.hold("ATTEMPTS_EXHAUSTED")
                else:
                    self.initial_features = detection.features if detection else ()
                    return self.plan(rpm, feed)
        return None
