"""Unknown demonstration parts, sampled from data.py's frozen distributions."""

from dataclasses import dataclass
from math import pi
from secrets import randbelow

import numpy as np

from .config import ToolProfile
from .data import varied_scenario
from .scenarios import BASE_SCENARIOS, EngagementSegment, Region, Scenario

DEMO_SEED_MIN = 500000
DEMO_SEED_MAX = 599999


@dataclass(frozen=True, slots=True)
class Part:
    seed: int
    scenario: Scenario
    profile: ToolProfile = ToolProfile()
    noise_std: float = 0.05
    chatter_hz: float = 1200.0
    initial_phase: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self):
        validate_part(self)


def _number(value, low, high, name):
    if type(value) not in (int, float) or not low <= value <= high:
        raise ValueError(f"{name} must be finite and in [{low}, {high}]")


def validate_part(part: Part) -> Part:
    """Refuse unsupported custom inputs; immutable tuples prevent later mutation."""
    if type(part) is not Part:
        raise ValueError("expected a Part")
    if type(part.seed) is not int or not DEMO_SEED_MIN <= part.seed <= DEMO_SEED_MAX:
        raise ValueError("seed must be in the demo range 500000–599999")
    _number(part.noise_std, 0.02, 0.10, "noise_std")
    _number(part.chatter_hz, 900.0, 1800.0, "chatter_hz")
    if type(part.initial_phase) is not tuple or len(part.initial_phase) != 2:
        raise ValueError("initial_phase must be a pair of numbers")
    for value in part.initial_phase:
        _number(value, 0.0, 2 * pi, "initial_phase")
    if (
        type(part.profile) is not ToolProfile
        or type(part.profile.profile_id) is not str
        or part.profile.profile_id != "four-tooth-v1"
        or part.profile.teeth != 4
    ):
        raise ValueError("profile must use the frozen four-tooth tool")
    _number(part.profile.h, 0.8, 2.0, "profile.h")
    s = part.scenario
    if type(s) is not Scenario:
        raise ValueError("scenario must be a Scenario")
    if type(s.name) is not str or type(s.version) is not str:
        raise ValueError("scenario name and version must be strings")
    _number(s.path_length_mm, 400.0, 600.0, "path_length_mm")
    if s.path_length_mm not in (400.0, 600.0):
        raise ValueError("path_length_mm must be 400 or 600")
    if type(s.regions) is not tuple or len(s.regions) > 2:
        raise ValueError("regions must be a tuple of at most two zones")
    for index, region in enumerate(s.regions):
        if type(region) is not Region:
            raise ValueError("regions must contain Region values")
        _number(
            region.start_mm,
            60.0 if index == 0 else 400.0,
            100.0 if index == 0 else 400.0,
            "start_mm",
        )
        _number(
            region.end_mm,
            320.0 if index == 0 else 600.0,
            min(400.0, s.path_length_mm) if index == 0 else s.path_length_mm,
            "end_mm",
        )
        if region.start_mm >= region.end_mm or (
            index and s.regions[index - 1].end_mm > region.start_mm
        ):
            raise ValueError("regions must be ordered and disjoint")
        if type(region.centers) is not tuple or not 1 <= len(region.centers) <= 3:
            raise ValueError("centers must be a tuple with one to three values")
        if type(region.widths) is not tuple or len(region.widths) != len(
            region.centers
        ):
            raise ValueError("widths must be a tuple matching centers")
        for center in region.centers:
            _number(center, 2880.0, 3840.0, "center")
        for width in region.widths:
            _number(width, 100.0, 2000.0, "width")
            if width > 180.0 and width != 2000.0:
                raise ValueError("width must be 100–180 or the frozen broad width 2000")
        _number(region.gain, 1.0, 1.0, "gain")
        if type(region.zone_class) is not str or region.zone_class != "material-zone":
            raise ValueError("zone_class must be material-zone")
    if type(s.impact_positions_mm) is not tuple:
        raise ValueError("impact_positions_mm must be a tuple")
    for position in s.impact_positions_mm:
        _number(position, 120.0, 280.0, "impact_positions_mm")
    if s.impact_positions_mm not in ((), (120.0, 280.0)):
        raise ValueError("impact_positions_mm must be empty or (120, 280)")
    if type(s.engagement) is not tuple or len(s.engagement) > 3:
        raise ValueError("engagement must be a tuple of at most three segments")
    for index, segment in enumerate(s.engagement):
        if type(segment) is not EngagementSegment:
            raise ValueError("engagement must contain EngagementSegment values")
        _number(segment.start_mm, 60.0, s.path_length_mm - 40.0, "engagement start_mm")
        _number(
            segment.end_mm,
            segment.start_mm + 40.0,
            min(segment.start_mm + 120.0, s.path_length_mm),
            "engagement end_mm",
        )
        _number(segment.level, 0.8, 2.0, "engagement level")
        if 1.0 < segment.level < 1.3:
            raise ValueError("engagement level must be .8–1 or 1.3–2")
        if index and s.engagement[index - 1].end_mm > segment.start_mm:
            raise ValueError("engagement segments must be ordered and disjoint")
    return part


def random_part(seed: int | None = None) -> Part:
    if seed is None:
        seed = DEMO_SEED_MIN + randbelow(DEMO_SEED_MAX - DEMO_SEED_MIN + 1)
    if type(seed) is not int or not DEMO_SEED_MIN <= seed <= DEMO_SEED_MAX:
        raise ValueError("seed must be in the demo range 500000–599999")
    rng = np.random.default_rng(seed)
    bases = tuple(BASE_SCENARIOS.values())
    scenario = varied_scenario(
        bases[int(rng.integers(len(bases)))], rng, rng.random() < 0.5
    )
    return Part(
        seed,
        scenario,
        ToolProfile(),
        float(rng.uniform(0.02, 0.10)),
        float(rng.uniform(900.0, 1800.0)),
        tuple(float(value) for value in rng.uniform(0.0, 2 * np.pi, 2)),
    )
