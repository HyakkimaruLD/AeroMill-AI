"""Causal, ordered v1 features; only received telemetry enters the window."""

import numpy as np
from scipy.signal import welch

from .config import SAMPLE_RATE, WINDOW_SAMPLES, ToolProfile
from .contracts import AppliedState, TelemetryChunk

AXIS_FEATURES = (
    "rms_ratio",
    "crest",
    "kurtosis",
    "dominant_hz",
    "spectral_entropy",
    "off_harmonic",
    "rms_change",
)
FEATURE_NAMES = tuple(f"{axis}_{name}" for axis in "xyz" for name in AXIS_FEATURES) + (
    "rpm",
    "feed_mm_min",
)


def extract_features(
    window: np.ndarray,
    operating_state: AppliedState,
    profile: ToolProfile,
    previous_rms: np.ndarray | None = None,
) -> tuple[float, ...]:
    x = np.asarray(window, dtype=np.float64)
    if x.shape != (WINDOW_SAMPLES, 3) or not np.isfinite(x).all():
        raise ValueError("expected a finite 2048 by 3 window")
    rms = np.sqrt(np.mean(x * x, axis=0))
    centered = x - np.mean(x, axis=0)
    variance = np.mean(centered**2, axis=0)
    # SciPy 1.18.1 installed _spectral_py.welch: periodic Hann, mean PSD;
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html
    hz, psd = welch(
        x,
        fs=SAMPLE_RATE,
        window="hann",
        nperseg=1024,
        noverlap=512,
        detrend="constant",
        axis=0,
    )
    total = np.sum(psd, axis=0)
    probability = psd / np.maximum(total, np.finfo(float).tiny)
    entropy = -np.sum(
        probability * np.log(np.maximum(probability, np.finfo(float).tiny)), axis=0
    ) / np.log(len(hz))
    harmonics = np.arange(1, 7) * profile.teeth * operating_state.rpm / 60
    bands = np.any(np.abs(hz[:, None] - harmonics) <= 16, axis=1)
    off = np.sum(psd[~bands], axis=0) / np.maximum(total, np.finfo(float).tiny)
    rows = np.column_stack(
        (
            rms / np.asarray(profile.reference_rms),
            np.max(np.abs(x), axis=0) / np.maximum(rms, np.finfo(float).tiny),
            np.mean(centered**4, axis=0)
            / np.maximum(variance**2, np.finfo(float).tiny),
            hz[np.argmax(psd, axis=0)],
            entropy,
            off,
            np.zeros(3) if previous_rms is None else rms - previous_rms,
        )
    )
    return tuple(float(v) for v in rows.flat) + (
        float(operating_state.rpm),
        float(operating_state.feed_mm_min),
    )


class FeatureStream:
    def __init__(self, profile: ToolProfile):
        self.profile = profile
        self.window = np.empty((0, 3), dtype=np.float64)
        self.previous_rms = None

    def append(self, telemetry: TelemetryChunk) -> None:
        self.window = np.concatenate((self.window, telemetry.vibration))[
            -WINDOW_SAMPLES:
        ]

    def extract(self, operating_state: AppliedState) -> tuple[float, ...] | None:
        if len(self.window) < WINDOW_SAMPLES:
            return None
        result = extract_features(
            self.window, operating_state, self.profile, self.previous_rms
        )
        self.previous_rms = np.asarray(result[:21:7]) * self.profile.reference_rms
        return result

    def push(self, telemetry: TelemetryChunk) -> tuple[float, ...] | None:
        self.append(telemetry)
        return self.extract(
            AppliedState(float(telemetry.rpm[-1]), float(telemetry.feed_mm_min[-1]))
        )


def residual_rms(
    window: np.ndarray, operating_state: AppliedState, profile: ToolProfile
) -> tuple[float, ...]:
    """Exact band-stop energy from the same Welch definition as off_harmonic.

    Total time-domain RMS times sqrt(off_harmonic) is only approximate because
    windowed/detrended PSD energy is not exactly time-domain mean-square.
    """
    hz, psd = welch(
        window,
        fs=SAMPLE_RATE,
        window="hann",
        nperseg=1024,
        noverlap=512,
        detrend="constant",
        axis=0,
    )
    harmonics = np.arange(1, 7) * profile.teeth * operating_state.rpm / 60
    bands = np.any(np.abs(hz[:, None] - harmonics) <= 16, axis=1)
    return tuple(np.sqrt(np.sum(psd[~bands], axis=0) * (hz[1] - hz[0])))
