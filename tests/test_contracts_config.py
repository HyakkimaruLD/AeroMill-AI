"""Break targets: contaminated public records, signal-based reference, wrong h scaling."""

from dataclasses import fields, is_dataclass, FrozenInstanceError
from math import sqrt
from typing import get_type_hints

import numpy as np
import pytest

from aeromill import config, contracts


def test_analytic_profile_reference():
    assert hasattr(config, "ToolProfile"), "Day 1 profile is missing"
    for h in (0.8, 1.0, 2.0):
        profile = config.ToolProfile(h=h)
        assert profile.reference_rms == tuple(
            m * h * sqrt(0.2) for m in (1.0, 0.8, 0.6)
        )
        with pytest.raises(FrozenInstanceError):
            profile.h = 3
    for h in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            config.ToolProfile(h=h)


def assert_no_truth(value):
    assert not isinstance(value, contracts.TruthFrame)
    if is_dataclass(value):
        for field in fields(value):
            assert field.name not in {"truth", "envelope", "gain", "centers", "widths"}
            assert_no_truth(getattr(value, field.name))
    elif isinstance(value, (tuple, list)):
        for item in value:
            assert_no_truth(item)
    elif isinstance(value, np.ndarray):
        assert value.dtype != object
        assert value.base is None or isinstance(value.base, np.ndarray)


def test_public_contract_type_graph_excludes_truth():
    assert hasattr(contracts, "Observation"), "Day 1 contracts are missing"
    seen = set()

    def visit(cls):
        if cls in seen:
            return
        seen.add(cls)
        assert cls is not contracts.TruthFrame
        from typing import get_args

        for annotation in get_type_hints(cls).values() if is_dataclass(cls) else ():
            visit(annotation)
        for arg in get_args(cls):
            visit(arg)

    for cls in (contracts.TelemetryChunk, contracts.Observation, config.ToolProfile):
        visit(cls)
        assert "truth" not in {f.name for f in fields(cls)}
