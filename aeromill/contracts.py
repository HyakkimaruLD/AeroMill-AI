"""Day 1 data contracts. Pass only telemetry to observers, never StepResult.

Truth is a separate evaluator/training channel. These records describe trusted
in-process data, not a sandbox against hostile Python reflection or mutation.
"""

from dataclasses import dataclass
from enum import StrEnum

from numpy.typing import NDArray
import numpy as np


class DataQuality(StrEnum):
    VALID = "valid"
    INVALID = "invalid"


class TruthState(StrEnum):
    STABLE = "stable"
    TRANSITION = "transition"
    CHATTER = "chatter"


@dataclass(frozen=True, slots=True)
class AppliedState:
    """Accepted controller targets; stop has priority."""

    rpm: float = 3200.0
    feed_mm_min: float = 1200.0
    stopped: bool = False


@dataclass(frozen=True, slots=True)
class TelemetryChunk:
    run_id: str
    sequence: int
    sample_start: int
    time_s: NDArray[np.float64]
    xyz_mm: NDArray[np.float64]
    zone_class: tuple[str, ...]
    rpm: NDArray[np.float64]
    feed_mm_min: NDArray[np.float64]
    vibration: NDArray[np.float64]
    quality: DataQuality = DataQuality.VALID


@dataclass(frozen=True, slots=True)
class Detection:
    features: tuple[float, ...]
    chatter_score: float
    quality: DataQuality = DataQuality.VALID
    residual_rms: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True)
class ControllerEvent:
    command_id: str
    status: str
    time_s: float
    run_id: str = ""


@dataclass(frozen=True, slots=True)
class Observation:
    telemetry: TelemetryChunk
    detection: Detection | None = None
    controller_events: tuple[ControllerEvent, ...] = ()


@dataclass(frozen=True, slots=True)
class TruthFrame:
    envelope: NDArray[np.float64]
    states: NDArray[np.str_]
    x_mm: NDArray[np.float64]
    moving: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class StepResult:
    telemetry: TelemetryChunk
    truth: TruthFrame


@dataclass(frozen=True, slots=True)
class Command:
    run_id: str
    command_id: str
    type: str
    rpm: float
    feed_mm_min: float
    reason: str
    incident_id: int


@dataclass(frozen=True, slots=True)
class CommandResult:
    command_id: str
    status: str
    reason: str = ""
