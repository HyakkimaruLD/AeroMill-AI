"""Break targets: feature permutation, wrong harmonic mask, window/hop/future leakage."""

import numpy as np
import pytest

from aeromill import features
from aeromill.config import ToolProfile
from aeromill.contracts import AppliedState
from aeromill.scenarios import BASE_SCENARIOS
from aeromill.simulator import Simulator


def test_features_api():
    assert hasattr(features, "extract_features")


@pytest.mark.parametrize("frequency,off", [(256.0, 0.0), (1200.0, 1.0)])
def test_analytic_tones_and_order(frequency, off):
    t = np.arange(2048) / 8192
    window = np.sin(2 * np.pi * frequency * t)[:, None] * np.array([1.0, 0.8, 0.6])
    f = features.extract_features(window, AppliedState(3840, 960), ToolProfile())
    assert features.FEATURE_NAMES == tuple(
        f"{axis}_{name}"
        for axis in "xyz"
        for name in (
            "rms_ratio",
            "crest",
            "kurtosis",
            "dominant_hz",
            "spectral_entropy",
            "off_harmonic",
            "rms_change",
        )
    ) + ("rpm", "feed_mm_min")
    assert len(f) == 23
    for axis in range(3):
        r = f[axis * 7 : axis * 7 + 7]
        assert r[0] == pytest.approx(np.sqrt(2.5))
        assert r[1] == pytest.approx(np.sqrt(2))
        assert r[2] == pytest.approx(1.5)
        assert r[3] == frequency
        assert 0 < r[4] < 1
        assert r[5] == pytest.approx(off, abs=1e-10)
        assert r[6] == 0
    assert f[-2:] == (3840, 960)


def test_causal_window_first_at_third_tick():
    sim = Simulator(BASE_SCENARIOS["stable"])
    stream = features.FeatureStream(ToolProfile())
    chunks = [sim.step(0.1, AppliedState()).telemetry for _ in range(4)]
    assert stream.push(chunks[0]) is None
    assert stream.push(chunks[1]) is None
    third = stream.push(chunks[2])
    expected = features.extract_features(
        np.concatenate([t.vibration for t in chunks[:3]])[-2048:],
        AppliedState(),
        ToolProfile(),
    )
    assert third == expected
    fourth = stream.push(chunks[3])
    assert fourth[6] == pytest.approx(
        (fourth[0] - third[0]) * ToolProfile().reference_rms[0]
    )


def test_rms_detector_verify_uses_calibration_without_changing_features():
    from aeromill.detector import RMSDetector

    d = RMSDetector(ToolProfile())
    t = np.arange(2048) / 8192
    window = np.sin(2 * np.pi * 256 * t)[:, None] * np.array([1.0, 0.8, 0.6])
    normal = d.predict(window, AppliedState(3840, 960))
    assert normal.chatter_score == 1.0
    # calibration ratio 1.1: actual 1.581/reference is 1.437/calibration.
    d.score_reference_rms = np.asarray(ToolProfile().reference_rms) * 1.1
    verified = d.predict(window, AppliedState(3840, 960))
    assert verified.chatter_score == 0.0
    assert verified.features[:6] == normal.features[:6]


def test_welch_matches_three_overlapping_mean_removed_hann_segments():
    # Independent periodogram oracle: a nonstationary tone exposes hop/window errors.
    n = np.arange(2048)
    signal = 7 + (1 + n / 2048) * np.sin(
        2 * np.pi * (256 * n / 8192 + 400 * (n / 8192) ** 2)
    )
    window = signal[:, None] * np.array([1.0, 0.8, 0.6])
    actual = features.extract_features(window, AppliedState(3840, 960), ToolProfile())
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(1024) / 1024)
    power = []
    for start in (0, 512, 1024):
        segment = signal[start : start + 1024]
        p = np.abs(np.fft.rfft((segment - segment.mean()) * hann)) ** 2
        p[1:-1] *= 2
        power.append(p)
    psd = np.mean(power, axis=0)
    weights = psd / psd.sum()
    entropy = -np.sum(
        weights * np.log(np.maximum(weights, np.finfo(float).tiny))
    ) / np.log(513)
    assert actual[3] == np.argmax(psd) * 8
    assert actual[4] == pytest.approx(entropy, abs=1e-12)
