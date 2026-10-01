"""Planning audit and mode-specific residual verification boundaries."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.signal import welch

from aeromill import features
from aeromill.agent import Agent
from aeromill.config import ToolProfile
from aeromill.contracts import AppliedState
from test_agent import observation, settled, warm


def test_planning_record_full_exclusions():
    a = Agent("r", ToolProfile())
    a.candidates = (
        (3200.0, 1200.0),
        (3520.0, 960.0),
        (0.0, 900.0),
        (2880.0, 960.0),
        (2880.0, 960.0),
        (3840.0, 720.0),
    )
    a.tried.add((3520.0, 960.0))
    a.plan(3200.0, 1200.0)
    rows = [e for e in a.events if e["event"] == "planning_record"]
    assert len(rows) == 1, "PLAN must emit its full decision"
    record = rows[0]
    assert [r["reason"] for r in record["ranked"]] == [
        "current pair",
        "already tried",
        "outside the limits",
        "kept",
        "duplicate",
        "kept",
    ]
    assert record["chosen"] == [2880.0, 960.0]
    assert len(record["excluded"]) == 4


def test_residual_against_direct_welch_band_stop():
    assert hasattr(features, "residual_rms"), "exact Welch residual missing"
    t = np.arange(2048) / 8192
    x = (1.2 * np.sin(2 * np.pi * 213.333 * t) + 0.1 * np.sin(2 * np.pi * 1200 * t))[
        :, None
    ] * np.array([1, 0.8, 0.6])
    hz, psd = welch(
        x,
        fs=8192,
        window="hann",
        nperseg=1024,
        noverlap=512,
        detrend="constant",
        axis=0,
    )
    stop = np.any(abs(hz[:, None] - np.arange(1, 7) * 3200 * 4 / 60) <= 16, axis=1)
    direct = np.sqrt(np.sum(np.where(stop[:, None], 0, psd), axis=0) * (hz[1] - hz[0]))
    assert np.allclose(
        features.residual_rms(x, AppliedState(), ToolProfile()), direct, rtol=1e-12
    )


def test_ml_calibration_and_verify_residual_is_not_total():
    a = Agent("r", ToolProfile(), mode="ml_agent")
    settled(a)
    assert np.all(a.calibration_residual_rms > 0)
    for k in range(39, 54):
        obs = observation(k, ratio=2.0, rpm=3520, feed=960)
        obs = replace(
            obs,
            detection=replace(
                obs.detection, residual_rms=tuple(a.calibration_residual_rms)
            ),
        )
        a.tick(obs)
    assert a.state == "MONITOR"
    assert any(e["event"] == "recovery_claim" for e in a.events)


@pytest.mark.parametrize(
    "factor,score,success", [(2.2, 0.3, True), (2.201, 0.3, False), (1.0, 0.301, False)]
)
def test_ml_verify_strict_residual_and_score(factor, score, success):
    a = Agent("r", ToolProfile(), mode="ml_agent")
    settled(a)
    for k in range(39, 54):
        obs = observation(k, score, rpm=3520, feed=960)
        obs = replace(
            obs,
            detection=replace(
                obs.detection, residual_rms=tuple(a.calibration_residual_rms * factor)
            ),
        )
        a.tick(obs)
        if k < 53:
            assert a.state == "VERIFY"
    assert (a.state == "MONITOR") == success


def test_missing_residual_cannot_finish_calibration():
    a = Agent("r", ToolProfile(), mode="ml_agent")
    for k in range(1, 51):
        obs = observation(k)
        if obs.detection:
            obs = replace(obs, detection=replace(obs.detection, residual_rms=None))
        a.tick(obs)
    assert (a.state, a.reason) == ("HOLD", "CALIBRATION_FAILED")


@pytest.mark.parametrize("x", [100.0, 400.0])
def test_invalid_residual_at_exact_calibration_deadline(x):
    a = Agent("r", ToolProfile(), mode="ml_agent")
    for k in range(1, 50):
        a.tick(observation(k, score=1 if k < 30 else 0))
    obs = observation(50, x=x)
    obs = replace(obs, detection=replace(obs.detection, residual_rms=(0.0, 0.0, 0.0)))
    a.tick(obs)
    assert (a.state, a.reason) == ("HOLD", "CALIBRATION_FAILED")
