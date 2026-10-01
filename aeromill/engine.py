"""Fixed 100 ms tick loop. Truth stays on the returned side channel, never in Agent."""

import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np

from .agent import Agent
from .config import SAMPLE_RATE, WINDOW_SAMPLES, ToolProfile
from .contracts import AppliedState, Command, ControllerEvent, DataQuality, Observation
from .controller import Controller
from .detector import RMSDetector, RFDetector
from .features import FeatureStream
from .evaluation import RunEvaluator
from .simulator import Simulator


class Engine:
    def __init__(
        self,
        scenario,
        *,
        seed: int = 42,
        mode: str = "threshold_search",
        profile: ToolProfile = ToolProfile(),
        noise_std: float = 0.05,
        detector=None,
        fault: str | None = None,
        log_path=None,
        run_id=None,
        model_dir="artifacts",
        chatter_hz=1200.0,
        initial_phase=(0.0, 0.0),
        memory=None,
    ):
        reference = np.asarray(profile.reference_rms, dtype=float)
        if (
            reference.shape != (3,)
            or not np.isfinite(reference).all()
            or np.any(reference <= 0)
        ):
            raise ValueError("invalid profile reference RMS")
        self.run_id = run_id or uuid4().hex
        self.events = []
        self._evaluator = RunEvaluator()
        self._evaluation = None
        self.simulator = Simulator(
            scenario,
            seed=seed,
            profile=profile,
            noise_std=noise_std,
            run_id=self.run_id,
            chatter_hz=chatter_hz,
            initial_phase=initial_phase,
        )
        self.controller = Controller(self.run_id, fault=fault)
        self.detector = (
            detector
            if detector is not None
            else (
                RFDetector(profile, model_dir=model_dir)
                if mode == "ml_agent"
                else RMSDetector(profile)
            )
        )
        self.stream = FeatureStream(profile)
        self.log_path = (
            Path(log_path) if log_path else Path("runs") / f"{self.run_id}.jsonl"
        )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.log_path.exists():
            try:
                for line in self.log_path.read_text().splitlines():
                    row = json.loads(line)
                    if (
                        not isinstance(row, dict)
                        or not {"event", "run_id", "time_s"} <= row.keys()
                    ):
                        raise ValueError("invalid event")
            except (ValueError, TypeError) as exc:
                raise ValueError(f"corrupted log: {self.log_path}") from exc
        # Open before any motion; inability to create the log prevents startup.
        with self.log_path.open("a"):
            pass
        model_version = "rms-v1"
        if mode == "ml_agent" and detector is None:
            model_version = json.loads((Path(model_dir) / "model.json").read_text())[
                "model_sha256"
            ]
        self.agent = Agent(
            self.run_id,
            profile,
            mode=mode,
            path_length_mm=scenario.path_length_mm,
            log=self.log,
            memory=memory,
            model_version=model_version,
        )
        self.index = 0
        self.queued = None
        self.last_telemetry = None
        self.processing_ms = []
        self.log(
            dict(
                event="run",
                run_id=self.run_id,
                time_s=0.0,
                seed=seed,
                mode=mode,
                scenario=scenario.name,
                scenario_version=scenario.version,
                profile_id=profile.profile_id,
                h=profile.h,
                reference_rms=list(profile.reference_rms),
            )
        )

    @property
    def time_s(self):
        return self.index / 10

    @property
    def done(self):
        return self.agent.state in ("HOLD", "COMPLETED")

    def log(self, row: dict) -> None:
        with self.log_path.open("a") as file:
            file.write(json.dumps(row, allow_nan=False) + "\n")
        self.events.append(row)

    def stop(self, reason: str) -> None:
        self.agent.now = self.index
        self.queued = None
        # Latch before logging: an unavailable log cannot prevent a stop.
        self.controller.apply(
            Command(
                self.run_id,
                f"{self.run_id}:stop",
                "stop",
                0.0,
                0.0,
                reason,
                self.agent.incident,
            )
        )
        self.simulator.stop_at_boundary(self.controller.applied)
        self.agent.hold(reason)

    def valid(self, t) -> bool:
        start = (self.index - 1) * SAMPLE_RATE // 10
        end = self.index * SAMPLE_RATE // 10
        n = end - start
        if (
            t is None
            or t.run_id != self.run_id
            or t.sequence != self.index - 1
            or t.sample_start != start
            or t.quality != DataQuality.VALID
        ):
            return False
        arrays = (
            (t.time_s, (n,)),
            (t.xyz_mm, (n, 3)),
            (t.rpm, (n,)),
            (t.feed_mm_min, (n,)),
            (t.vibration, (n, 3)),
        )
        if any(
            not isinstance(a, np.ndarray)
            or a.shape != shape
            or a.dtype.kind not in "fiu"
            or not np.isfinite(a).all()
            for a, shape in arrays
        ):
            return False
        return len(t.zone_class) == n and np.array_equal(
            t.time_s, np.arange(start + 1, end + 1) / SAMPLE_RATE
        )

    def tick(self, *, stop: bool = False):
        try:
            step = self._tick(stop=stop)
            if step is not None:
                self._evaluator.append(step)
            return step
        except OSError:
            self.agent.log = None
            self.stop("LOG_ERROR")
            raise

    def _tick(self, *, stop: bool = False):
        if self.done:
            return None
        self.index += 1
        if stop:
            self.stop("MANUAL_STOP")
        events = []
        if self.queued is not None:
            result = self.controller.apply(self.queued)
            event = ControllerEvent(
                result.command_id, result.status, self.time_s, self.run_id
            )
            events.append(event)
            self.log(
                dict(
                    event="ack",
                    run_id=self.run_id,
                    time_s=self.time_s,
                    command_id=result.command_id,
                    status=result.status,
                    reason=result.reason,
                )
            )
            self.queued = None
            if result.status == "rejected":
                self.stop("COMMAND_REJECTED")
        step = self.simulator.step(0.1, self.controller.applied)
        if self.done:
            return step
        t = step.telemetry
        started = perf_counter()
        try:
            if not self.valid(t):
                self.stop("DATA_QUALITY")
                return step
            self.last_telemetry = t
            self.stream.append(t)
            if len(self.stream.window) == WINDOW_SAMPLES and np.all(
                np.var(self.stream.window, axis=0) == 0
            ):
                self.stop("DATA_QUALITY")
                return step
            events.extend(self.controller.observe(t))
            for event in events:
                if event.status == "reached":
                    self.log(
                        dict(
                            event="reached",
                            run_id=self.run_id,
                            time_s=self.time_s,
                            command_id=event.command_id,
                        )
                    )
            detection = None
            if len(self.stream.window) == WINDOW_SAMPLES:
                try:
                    if isinstance(self.detector, RMSDetector):
                        self.detector.score_reference_rms = (
                            self.agent.calibration_rms
                            if self.agent.state == "VERIFY"
                            else np.asarray(self.agent.profile.reference_rms)
                        )
                    detection = self.detector.predict(
                        self.stream.window,
                        AppliedState(float(t.rpm[-1]), float(t.feed_mm_min[-1])),
                    )
                    if (
                        detection.quality != DataQuality.VALID
                        or len(detection.features) != 23
                        or not np.isfinite(detection.features).all()
                        or not np.isfinite(detection.chatter_score)
                        or not 0 <= detection.chatter_score <= 1
                    ):
                        raise ValueError("invalid detector output")
                    if self.agent.mode == "ml_agent" and (
                        detection.residual_rms is None
                        or np.asarray(detection.residual_rms).shape != (3,)
                        or not np.isfinite(detection.residual_rms).all()
                        or np.any(np.asarray(detection.residual_rms) < 0)
                    ):
                        raise ValueError("invalid residual RMS")
                except Exception:
                    self.stop("MODEL_ERROR")
                    return step
            command = self.agent.tick(Observation(t, detection, tuple(events)))
            if command is not None:
                if command.type == "stop":
                    self.controller.apply(command)
                    self.simulator.stop_at_boundary(self.controller.applied)
                else:
                    self.queued = command
            self.log(
                dict(
                    event="tick",
                    run_id=self.run_id,
                    time_s=self.time_s,
                    state=self.agent.state,
                    x_mm=float(t.xyz_mm[-1, 0]),
                    rpm=float(t.rpm[-1]),
                    feed_mm_min=float(t.feed_mm_min[-1]),
                    score=detection.chatter_score if detection else None,
                )
            )
        finally:
            self.processing_ms.append((perf_counter() - started) * 1000)
        return step

    def run(self):
        while not self.done:
            self.tick()
        return self

    def evaluate(self):
        if not self.done:
            return None
        if self._evaluation is None:
            self._evaluation = self._evaluator.evaluate(
                self.events,
                state=self.agent.state,
                path_length_mm=self.agent.path_length_mm,
            )
        return self._evaluation
