"""Hidden scenario parameters; all material regions are indexed by position."""

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class Region:
    start_mm: float
    end_mm: float
    centers: tuple[float, ...]
    widths: tuple[float, ...]
    gain: float = 1.0
    zone_class: str = "material-zone"


@dataclass(frozen=True, slots=True)
class EngagementSegment:
    start_mm: float
    end_mm: float
    level: float

    def __post_init__(self):
        if (
            not all(isfinite(v) for v in (self.start_mm, self.end_mm, self.level))
            or self.start_mm < 60
            or not 40 <= self.end_mm - self.start_mm <= 120
            or not (0.8 <= self.level <= 1.0 or 1.3 <= self.level <= 2.0)
        ):
            raise ValueError("invalid engagement segment")

    def multiplier(self, x_mm: float) -> float:
        ramp = max(
            0.0, min(1.0, (x_mm - self.start_mm) / 20, (self.end_mm - x_mm) / 20)
        )
        return 1.0 + (self.level - 1.0) * ramp


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    regions: tuple[Region, ...] = ()
    path_length_mm: float = 400.0
    impact_positions_mm: tuple[float, ...] = ()
    version: str = "v1"
    engagement: tuple[EngagementSegment, ...] = ()

    def __post_init__(self):
        if len(self.engagement) > 3 or any(
            a.end_mm > b.start_mm for a, b in zip(self.engagement, self.engagement[1:])
        ):
            raise ValueError(
                "engagement requires at most three disjoint ordered segments"
            )

    def engagement_at(self, x_mm: float) -> float:
        for segment in self.engagement:
            if segment.start_mm <= x_mm <= segment.end_mm:
                return segment.multiplier(x_mm)
        return 1.0

    def region_at(self, x_mm: float) -> Region | None:
        for region in self.regions:
            if region.start_mm <= x_mm < region.end_mm or (
                x_mm == region.end_mm == self.path_length_mm
            ):
                return region
        return None


BASE_SCENARIOS = MappingProxyType(
    {
        "stable": Scenario("stable"),
        "recover_a": Scenario("recover_a", (Region(80.0, 400.0, (3200.0,), (120.0,)),)),
        "retry_b": Scenario(
            "retry_b", (Region(80.0, 400.0, (3200.0, 3520.0), (120.0, 120.0)),)
        ),
        "recover_c": Scenario(
            "recover_c",
            (Region(80.0, 400.0, (2880.0, 3200.0, 3520.0), (120.0, 120.0, 120.0)),),
        ),
        "unrecoverable": Scenario(
            "unrecoverable", (Region(80.0, 400.0, (3300.0,), (2000.0,)),)
        ),
        "repeat_after_c": Scenario(
            "repeat_after_c",
            (
                Region(80.0, 320.0, (2880.0, 3200.0, 3520.0), (120.0, 120.0, 120.0)),
                Region(400.0, 600.0, (3840.0,), (120.0,)),
            ),
            path_length_mm=600.0,
        ),
    }
)

NAMED_SCENARIOS = MappingProxyType(
    dict(
        BASE_SCENARIOS,
        stable_engagement=Scenario(
            "stable_engagement", engagement=(EngagementSegment(150.0, 250.0, 2.0),)
        ),
    )
)
