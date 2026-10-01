"""T4 session API: one engine per run, detached snapshots, terminal-only grading.

Synthetic simulation; not validated on real CNC. No UI imports or rendering.
"""

from copy import deepcopy
import csv
from dataclasses import dataclass, replace
from io import StringIO
import json
from pathlib import Path

import numpy as np
from scipy.signal import welch

from .config import SAMPLE_RATE, WINDOW_SAMPLES
from .engine import Engine
from .parts import Part, validate_part
from .scenarios import NAMED_SCENARIOS
from .viewer import FAILURES, scenario_for, stability_map


@dataclass(frozen=True)
class Snapshot:
    time_s: float
    state: str
    reason: str
    terminal: bool
    x_mm: float
    path_length_mm: float
    rpm: float
    feed_mm_min: float
    target_rpm: float
    target_feed_mm_min: float
    zones: list[tuple[float, float, str]]
    vibration_tail: np.ndarray
    spectrum: tuple[np.ndarray, np.ndarray]
    tooth_hz: float
    series: dict[str, np.ndarray]
    events: list[dict]
    evaluation: dict | None


def list_scenarios() -> list[str]:
    return [*NAMED_SCENARIOS, *FAILURES]


def list_modes() -> list[str]:
    return ["no_adaptation", "baseline", "threshold_search", "ml_agent"]


class RunSession:
    def __init__(self, scenario: str, seed: int, mode: str):
        if scenario not in list_scenarios() or mode not in list_modes():
            raise ValueError("unknown scenario or mode")
        self._scenario, self._seed, self._mode = scenario_for(scenario), seed, mode
        self._fault = FAILURES.get(scenario, (None, None))[0]
        self._manual_stop = scenario == "manual_stop"
        self._engine = None
        self._part = None
        self._truth = []
        self._stop_requested = False
        self._tail = np.empty((0, 3), dtype=np.float64)
        self._series = {key: [] for key in ("t", "score", "rpm", "feed", "x")}
        self._snapshot = Snapshot(
            0.0,
            "WARMUP",
            "",
            False,
            0.0,
            float(self._scenario.path_length_mm),
            3200.0,
            1200.0,
            3200.0,
            1200.0,
            [(r.start_mm, r.end_mm, r.zone_class) for r in self._scenario.regions],
            self._tail,
            (np.empty(0), np.empty(0)),
            3200.0 * 4 / 60,
            {k: np.empty(0) for k in self._series},
            [],
            None,
        )

    @classmethod
    def from_part(cls, part: Part, mode: str):
        validate_part(part)
        session = cls("stable", part.seed, mode)
        session._part = part
        session._scenario = replace(part.scenario, name="unknown", version="demo-v1")
        session._snapshot = replace(
            session._snapshot,
            zones=[],
            path_length_mm=float(part.scenario.path_length_mm),
        )
        return session

    def advance(self, ticks: int) -> Snapshot:
        if type(ticks) is not int or ticks < 0:
            raise ValueError("ticks must be a nonnegative integer")
        if ticks == 0 or self._snapshot.terminal:
            return self.snapshot()
        if self._engine is None:
            options = (
                {}
                if self._part is None
                else dict(
                    profile=self._part.profile,
                    noise_std=self._part.noise_std,
                    chatter_hz=self._part.chatter_hz,
                    initial_phase=self._part.initial_phase,
                )
            )
            self._engine = Engine(
                self._scenario,
                seed=self._seed,
                mode=self._mode,
                model_dir=Path(__file__).parents[1] / "artifacts",
                fault=self._fault,
                **options,
            )
        e = self._engine
        for _ in range(ticks):
            step = e.tick(
                stop=self._stop_requested or (self._manual_stop and e.index == 10)
            )
            self._stop_requested = False
            if step is not None and self._part is not None:
                self._truth.append(
                    (step.truth.envelope.copy(), step.truth.states.copy())
                )
            if step is not None and step.telemetry is not None:
                t = step.telemetry
                self._tail = np.concatenate((self._tail, t.vibration))[-SAMPLE_RATE:]
                tick_event = next(
                    (
                        row
                        for row in reversed(e.events)
                        if row["event"] == "tick" and row["time_s"] == e.time_s
                    ),
                    None,
                )
                score = tick_event["score"] if tick_event else None
                values = (
                    e.time_s,
                    np.nan if score is None else score,
                    float(t.rpm[-1]),
                    0.0 if e.controller.applied.stopped else float(t.feed_mm_min[-1]),
                    float(t.xyz_mm[-1, 0]),
                )
                for key, value in zip(self._series, values, strict=True):
                    self._series[key].append(value)
            if e.done:
                break
        spectrum = (np.empty(0), np.empty(0))
        if len(e.stream.window) == WINDOW_SAMPLES:
            spectrum = welch(
                e.stream.window[:, 0],
                fs=SAMPLE_RATE,
                window="hann",
                nperseg=1024,
                noverlap=512,
                detrend="constant",
            )
        last = {
            k: self._series[k][-1] if self._series[k] else initial
            for k, initial in (("rpm", 3200.0), ("feed", 1200.0), ("x", 0.0))
        }
        if e.controller.applied.stopped:
            last["feed"] = 0.0
        events = (
            e.events
            if self._part is None
            else [
                {key: value for key, value in event.items() if key != "seed"}
                for event in e.events
            ]
        )
        self._snapshot = Snapshot(
            float(e.time_s),
            e.agent.state,
            e.agent.reason,
            bool(e.done),
            float(last["x"]),
            float(e.agent.path_length_mm),
            float(last["rpm"]),
            float(last["feed"]),
            float(e.controller.applied.rpm),
            float(e.controller.applied.feed_mm_min),
            self._snapshot.zones,
            self._tail,
            spectrum,
            float(e.agent.profile.teeth * last["rpm"] / 60),
            {k: np.asarray(v, dtype=np.float64) for k, v in self._series.items()},
            events,
            e.evaluate(),
        )
        return self.snapshot()

    def request_stop(self) -> None:
        if not self._snapshot.terminal:
            self._stop_requested = True

    def snapshot(self) -> Snapshot:
        return deepcopy(self._snapshot)

    def reveal(self) -> dict:
        if not self._snapshot.terminal:
            raise RuntimeError("reveal requires a terminal run")
        if self._part is None:
            raise ValueError("reveal requires an unknown-part session")
        envelope = np.concatenate([chunk[0] for chunk in self._truth])
        states = np.concatenate([chunk[1] for chunk in self._truth])
        return dict(
            zones=self._scenario.regions,
            engagement=self._scenario.engagement,
            time_s=np.arange(1, len(envelope) + 1) / SAMPLE_RATE,
            envelope=envelope,
            truth_states=states,
            evaluation=deepcopy(self._snapshot.evaluation),
        )

    def stability_map(self, feed, rpm_grid, x_grid) -> dict:
        if self._part is not None and not self._snapshot.terminal:
            raise RuntimeError("unknown-part map requires a terminal run")
        return stability_map(self._scenario, feed, rpm_grid, x_grid)

    def export(self) -> tuple[bytes, bytes]:
        s = self._snapshot
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(s.series)
        writer.writerows(zip(*s.series.values(), strict=True))
        return (
            output.getvalue().encode(),
            json.dumps(
                dict(events=s.events, evaluation=s.evaluation), allow_nan=False
            ).encode(),
        )


def compare(scenario: str, seed: int, modes: list[str] | None = None) -> list[dict]:
    rows = []
    for mode in list_modes() if modes is None else modes:
        s = RunSession(scenario, seed, mode)
        result = s.advance(650).evaluation
        rows.append(dict(result, mode=mode))
    baseline = next(
        (r["traversal_time_s"] for r in rows if r["mode"] == "no_adaptation"), None
    )
    for row in rows:
        time = row["traversal_time_s"]
        row["productivity_loss"] = (
            None if not baseline or time is None else time / baseline - 1
        )
    return rows
