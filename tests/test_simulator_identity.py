"""Bit identity to the immutable Day 1 simulator across commands and impacts."""

from dataclasses import replace
from pathlib import Path
import types

import numpy as np

from aeromill.simulator import Simulator
from aeromill.scenarios import BASE_SCENARIOS
from aeromill.contracts import AppliedState


def test_day1_bit_identity():
    source = (Path(__file__).parent / "evidence" / "day1_simulator.txt").read_text()
    old = types.ModuleType("aeromill.day1_reference")
    old.__package__ = "aeromill"
    exec(compile(source, "day1_reference", "exec"), old.__dict__)
    scenario = replace(BASE_SCENARIOS["recover_c"], impact_positions_mm=(10.0, 80.0))
    a = Simulator(scenario, seed=129)
    b = old.Simulator(scenario, seed=129)
    schedule = (
        [AppliedState()] * 45
        + [AppliedState(3520, 960)] * 10
        + [AppliedState(2880, 960)] * 10
        + [AppliedState(3840, 720)] * 10
        + [AppliedState(stopped=True)] * 2
    )
    for state in schedule:
        x = a.step(0.1, state)
        y = b.step(0.1, state)
        for name in ("time_s", "xyz_mm", "rpm", "feed_mm_min", "vibration"):
            assert np.array_equal(
                getattr(x.telemetry, name), getattr(y.telemetry, name)
            ), name
        assert np.array_equal(x.truth.envelope, y.truth.envelope)
        assert np.array_equal(x.truth.states, y.truth.states)
