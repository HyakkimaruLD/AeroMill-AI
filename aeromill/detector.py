"""RMS and frozen RF detectors share the predict(window, state) boundary."""

import numpy as np

from .config import ToolProfile
from .contracts import AppliedState, Detection
from .features import extract_features


class RMSDetector:
    def __init__(self, profile: ToolProfile):
        self.profile = profile
        self.previous_rms = None
        self.score_reference_rms = np.asarray(profile.reference_rms)

    def predict(self, window: np.ndarray, operating_state: AppliedState) -> Detection:
        features = extract_features(
            window, operating_state, self.profile, self.previous_rms
        )
        ratios = np.asarray(features[:21:7])
        self.previous_rms = ratios * self.profile.reference_rms
        return Detection(
            features, float(np.any(self.previous_rms / self.score_reference_rms > 1.5))
        )


class RFDetector:
    """Observation-only model score, with residual energy for online verification."""

    def __init__(self, profile: ToolProfile, *, model=None, model_dir="artifacts"):
        if model is None:
            from .training import load_model

            model = load_model(model_dir)
        self.model, self.profile = model, profile
        self.previous_rms = None

    def predict(self, window: np.ndarray, operating_state: AppliedState) -> Detection:
        from .features import residual_rms

        features = extract_features(
            window, operating_state, self.profile, self.previous_rms
        )
        self.previous_rms = np.asarray(features[:21:7]) * self.profile.reference_rms
        score = float(
            self.model.predict_proba(np.asarray(features).reshape(1, -1))[0, 1]
        )
        return Detection(
            features,
            score,
            residual_rms=residual_rms(window, operating_state, self.profile),
        )
