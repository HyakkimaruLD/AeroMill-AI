"""Frozen v1 numerical constants and public analytic tool profile (spec §2, R7)."""

from dataclasses import dataclass
from math import isfinite, sqrt
from types import MappingProxyType
from typing import Final

from .contracts import AppliedState

SAMPLE_RATE: Final = 8192
WINDOW_SAMPLES: Final = 2048
HOP_S: Final = 0.1
RPM_LIMITS: Final = (2400.0, 4200.0)
FEED_LIMITS: Final = (600.0, 1200.0)
RPM_SLEW: Final = 1000.0
FEED_SLEW: Final = 600.0
AXIS_MULTIPLIERS: Final = (1.0, 0.8, 0.6)
ENVELOPE_TAU_S: Final = 0.35
IMPACT_TAU_S: Final = 0.02
MAX_RUN_S: Final = 65.0
STABLE_MAX: Final = 0.25
CHATTER_MIN: Final = 0.75
FORMULA_VERSION: Final = "v1"
CANDIDATE_SET_VERSION: Final = "v1"
CANDIDATES: Final = MappingProxyType(
    {
        "A": AppliedState(3520.0, 960.0),
        "B": AppliedState(2880.0, 960.0),
        "C": AppliedState(3840.0, 720.0),
    }
)


@dataclass(frozen=True, slots=True)
class ToolProfile:
    profile_id: str = "four-tooth-v1"
    h: float = 1.0
    teeth: int = 4

    def __post_init__(self):
        if not isinstance(self.h, (int, float)) or not isfinite(self.h) or self.h <= 0:
            raise ValueError("h must be finite and positive")
        if type(self.teeth) is not int or self.teeth <= 0:
            raise ValueError("teeth must be a positive integer")
        if any(not isfinite(v) or v <= 0 for v in self.reference_rms):
            raise ValueError("reference RMS must be finite and positive")

    @property
    def reference_rms(self) -> tuple[float, float, float]:
        return tuple(m * self.h * sqrt(0.2) for m in AXIS_MULTIPLIERS)
