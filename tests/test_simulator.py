"""Each assertion protects a physical/data-flow property, not a success flag."""

from dataclasses import replace
from math import exp, pi

import numpy as np
import pytest

from aeromill import simulator, scenarios
from aeromill.config import CANDIDATES, ToolProfile
from aeromill.contracts import AppliedState, Observation
from test_contracts_config import assert_no_truth


def make(name="recover_a", **kwargs):
    assert hasattr(simulator, "Simulator"), "Day 1 simulator is missing"
    return simulator.Simulator(scenarios.BASE_SCENARIOS[name], **kwargs)


def rms(signal):
    return np.sqrt(np.mean(signal**2, axis=0))


def test_replay_chunking_and_sample_clock():
    whole = make(seed=42).step(1.0, AppliedState(3840.0, 720.0))
    repeat = make(seed=42).step(1.0, AppliedState(3840.0, 720.0))
    np.testing.assert_array_equal(whole.telemetry.vibration, repeat.telemetry.vibration)
    sim = make(seed=42)
    parts = [sim.step(0.1, AppliedState(3840.0, 720.0)) for _ in range(10)]
    ends = np.cumsum([len(p.telemetry.time_s) for p in parts])
    np.testing.assert_array_equal(
        ends, [819, 1638, 2457, 3276, 4096, 4915, 5734, 6553, 7372, 8192]
    )
    for field in ("vibration", "rpm", "feed_mm_min", "xyz_mm", "time_s"):
        np.testing.assert_array_equal(
            getattr(whole.telemetry, field),
            np.concatenate([getattr(p.telemetry, field) for p in parts]),
        )
    np.testing.assert_array_equal(
        whole.truth.envelope, np.concatenate([p.truth.envelope for p in parts])
    )
    assert [p.telemetry.sequence for p in parts] == list(range(10))
    assert parts[-1].telemetry.sample_start == 7372
    other = make(seed=43).step(1.0, AppliedState(3840.0, 720.0))
    assert not np.array_equal(whole.telemetry.vibration, other.telemetry.vibration)
    np.testing.assert_array_equal(whole.truth.envelope, other.truth.envelope)


def test_noise_absolute_index_and_axis_independence():
    assert hasattr(simulator, "axis_noise"), "indexed noise is missing"
    for seed in (42, 43, 44):
        full = simulator.axis_noise(seed, 0, 8192)
        split = np.concatenate(
            [simulator.axis_noise(seed, 0, 819), simulator.axis_noise(seed, 819, 7373)]
        )
        np.testing.assert_array_equal(full, split)
        assert abs(full.mean()) < 0.04
        assert 0.96 < full.std() < 1.04
    assert not np.array_equal(
        simulator.axis_noise(42, 0, 100), simulator.axis_noise(43, 0, 100)
    )


def test_changed_command_changes_hidden_state_and_next_signal():
    changed, control = make(seed=42), make(seed=42)
    before = changed.step(6.0, AppliedState())
    control.step(6.0, AppliedState())
    assert before.truth.envelope[-1] >= 0.75
    after = changed.step(3.0, CANDIDATES["A"])
    unchanged = control.step(3.0, AppliedState())
    assert after.truth.envelope[0] != unchanged.truth.envelope[0]
    assert not np.array_equal(
        after.telemetry.vibration[:819], unchanged.telemetry.vibration[:819]
    )
    assert np.max(after.truth.envelope[-8192:]) <= 0.25
    assert np.min(unchanged.truth.envelope[-8192:]) >= 0.75
    before_rms, after_rms = (
        rms(before.telemetry.vibration[-8192:]),
        rms(after.telemetry.vibration[-8192:]),
    )
    assert np.all(after_rms < 0.6 * before_rms)
    print(
        f"CAUSAL A_before={before.truth.envelope[-1]:.6f} A_after={after.truth.envelope[-1]:.6f} RMS_before={before_rms} RMS_after={after_rms}"
    )


def test_stable_throughout():
    result = make("stable").step(10.0, AppliedState())
    assert np.all(result.truth.envelope <= 0.25)
    assert set(result.truth.states) == {"stable"}


@pytest.mark.parametrize("candidate", ["A", "B", "C"])
def test_unrecoverable_stays_chatter(candidate):
    sim = make("unrecoverable")
    sim.step(6.0, AppliedState())
    result = sim.step(3.0, CANDIDATES[candidate])
    assert np.min(result.truth.envelope) >= 0.75
    print(f"UNRECOVERABLE {candidate} min_A={result.truth.envelope.min():.6f}")


def test_stop_freezes_motion_and_decays_envelope():
    sim = make()
    before = sim.step(6.0, AppliedState())
    stopped = sim.step(1.0, AppliedState(stopped=True))
    assert np.all(stopped.telemetry.feed_mm_min == 0)
    assert np.all(stopped.telemetry.xyz_mm[:, 0] == before.telemetry.xyz_mm[-1, 0])
    assert not stopped.truth.moving.any()
    expected = before.truth.envelope[-1] * np.exp(-np.arange(1, 8193) / 8192 / 0.35)
    np.testing.assert_allclose(stopped.truth.envelope, expected, rtol=0, atol=1e-10)


@pytest.mark.parametrize("h,noise", [(2.0, 0.05), (0.8, 0.10)])
def test_stable_rms_ratio_every_axis(h, noise):
    sim = make("stable", profile=ToolProfile(h=h), noise_std=noise, seed=42)
    result = sim.step(3.0, AppliedState())
    ratios = rms(result.telemetry.vibration) / sim.profile.reference_rms
    assert np.all(ratios <= 1.5)
    windows = [
        rms(result.telemetry.vibration[k * 8192 // 10 - 2048 : k * 8192 // 10])
        / sim.profile.reference_rms
        for k in range(3, 31)
    ]
    maxima = np.max(windows, axis=0)
    assert np.all(maxima <= 1.5)
    print(f"REFERENCE h={h} noise={noise} ratios={ratios} window_max={maxima}")


@pytest.mark.parametrize("h", [0.8, 1.0, 2.0])
def test_slew_position_and_continuous_phase(h):
    sim = make("stable", noise_std=0, profile=ToolProfile(h=h))
    result = sim.step(1.0, AppliedState(4200.0, 600.0))
    n = np.arange(1, 8193)
    rpm = 3200 + 1000 * n / 8192
    feed = 1200 - 600 * n / 8192
    np.testing.assert_allclose(result.telemetry.rpm, rpm, rtol=0, atol=1e-10)
    np.testing.assert_allclose(result.telemetry.feed_mm_min, feed, rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        result.truth.x_mm, np.cumsum(feed / 60 / 8192), rtol=0, atol=1e-10
    )
    tooth_phase = np.cumsum(2 * pi * 4 * rpm / 60 / 8192)
    chatter_phase = n * 2 * pi * 1200 / 8192
    expected = h * (
        0.6 * np.sin(tooth_phase) + 0.2 * np.sin(2 * tooth_phase)
    ) + 0.1 * np.sin(chatter_phase)
    np.testing.assert_allclose(
        result.telemetry.vibration[:, 0], expected, rtol=0, atol=1e-10
    )
    np.testing.assert_allclose(
        result.telemetry.vibration[:, 1:],
        expected[:, None] * [0.8, 0.6],
        rtol=0,
        atol=1e-10,
    )


def test_runtime_public_data_cannot_reach_truth():
    result = make().step(0.1, AppliedState())
    assert_no_truth(result.telemetry)
    assert_no_truth(Observation(result.telemetry))
    assert_no_truth(ToolProfile())
    assert not np.shares_memory(result.truth.x_mm, result.telemetry.xyz_mm)


def test_scenarios_geometry_and_resonance():
    assert hasattr(scenarios, "BASE_SCENARIOS"), "six base scenarios missing"
    assert set(scenarios.BASE_SCENARIOS) == {
        "stable",
        "recover_a",
        "retry_b",
        "recover_c",
        "unrecoverable",
        "repeat_after_c",
    }
    for name in ("recover_a", "retry_b", "recover_c", "unrecoverable"):
        scenario = scenarios.BASE_SCENARIOS[name]
        assert scenario.path_length_mm == 400
        assert scenario.region_at(79.99) is None
        assert scenario.region_at(80.0) is not None
    repeat = scenarios.BASE_SCENARIOS["repeat_after_c"]
    assert repeat.path_length_mm == 600
    assert repeat.region_at(320.0) is None
    assert repeat.region_at(399.99) is None
    assert repeat.region_at(400.0).centers == (3840.0,)
    assert repeat.region_at(600.0) is not None
    assert repeat.region_at(80.0).zone_class == repeat.region_at(400.0).zone_class
    assert simulator.resonance(3200.0, (3200.0,), (120.0,)) == 1.0
    assert simulator.resonance(3320.0, (3200.0,), (120.0,)) == pytest.approx(exp(-1))
    assert simulator.resonance(3200.0, (3200.0, 3200.0), (120.0, 120.0)) == 1.0


def test_run_axis_seed_pairs_do_not_reuse_noise():
    clean = make("stable", noise_std=0).step(0.1, AppliedState()).telemetry.vibration
    runs = [
        make("stable", seed=seed).step(0.1, AppliedState()).telemetry.vibration - clean
        for seed in (42, 43)
    ]
    for left in range(3):
        for right in range(3):
            # Quantization on subtraction can differ by 1 ulp after axis scaling.
            assert not np.allclose(
                runs[0][:, left], runs[1][:, right], rtol=0, atol=1e-14
            )


def test_envelope_matches_analytic_target_and_truth_labels():
    result = make().step(6.0, AppliedState())
    # Constant F=1200 reaches x=80 at sample 32768, inclusive right endpoint.
    expected = np.full(49152, 0.1)
    expected[32767:] = 4.1 - 4 * np.exp(-np.arange(1, 16386) / 8192 / 0.35)
    np.testing.assert_allclose(result.truth.envelope, expected, rtol=0, atol=1e-10)
    np.testing.assert_array_equal(result.truth.states == "stable", expected <= 0.25)
    np.testing.assert_array_equal(result.truth.states == "chatter", expected >= 0.75)
    assert "transition" in result.truth.states


def test_feed_alone_changes_envelope_target():
    sim = make()
    sim.step(6.0, AppliedState())
    ramp = sim.step(1.0, AppliedState(3200.0, 600.0))
    settled = sim.step(1.0, AppliedState(3200.0, 600.0))
    # R=1, gain=1, F=600 gives A_target=2.1, independent of production helper.
    expected = 2.1 + (ramp.truth.envelope[-1] - 2.1) * np.exp(
        -np.arange(1, 8193) / 8192 / 0.35
    )
    np.testing.assert_allclose(settled.truth.envelope, expected, rtol=0, atol=1e-10)


@pytest.mark.parametrize(
    "name,candidate,chatter",
    [
        ("retry_b", "A", True),
        ("retry_b", "B", False),
        ("recover_c", "A", True),
        ("recover_c", "B", True),
        ("recover_c", "C", False),
    ],
)
def test_base_region_response(name, candidate, chatter):
    sim = make(name)
    sim.step(6.0, AppliedState())
    after = sim.step(3.0, CANDIDATES[candidate])
    tail = after.truth.envelope[-8192:]
    assert np.all(tail >= 0.75) if chatter else np.all(tail <= 0.25)


def test_impact_is_spatial_decaying_and_not_hidden_chatter():
    base = scenarios.BASE_SCENARIOS["stable"]
    impacted = simulator.Simulator(
        replace(base, impact_positions_mm=(1.0,)), noise_std=0
    )
    result = impacted.step(0.2, AppliedState())
    clean = make("stable", noise_std=0).step(0.2, AppliedState())
    start = np.searchsorted(clean.truth.x_mm, 1.0)
    elapsed = np.arange(len(result.truth.envelope) - start) / 8192
    expected = np.zeros(len(result.truth.envelope))
    expected[start:] = np.exp(-elapsed / 0.02) * np.sin(2 * pi * 1200 * elapsed)
    np.testing.assert_allclose(
        result.telemetry.vibration - clean.telemetry.vibration,
        expected[:, None] * [1.0, 0.8, 0.6],
        rtol=0,
        atol=1e-10,
    )
    np.testing.assert_array_equal(result.truth.envelope, clean.truth.envelope)


@pytest.mark.parametrize(
    "target",
    [
        AppliedState(2399.0, 1200.0),
        AppliedState(4201.0, 1200.0),
        AppliedState(3200.0, 599.0),
        AppliedState(3200.0, 1201.0),
        AppliedState(float("nan"), 960.0),
    ],
)
def test_invalid_target_rejected_without_advancing(target):
    sim = make()
    with pytest.raises(ValueError):
        sim.step(0.1, target)
    result = sim.step(0.1, AppliedState())
    assert result.telemetry.sample_start == 0
    assert result.telemetry.sequence == 0
    np.testing.assert_array_equal(
        result.telemetry.vibration, make().step(0.1, AppliedState()).telemetry.vibration
    )
