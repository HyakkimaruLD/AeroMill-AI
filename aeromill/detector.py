"""RMS and frozen RF detectors share the predict(window, state) boundary."""

import numpy as np

from .config import ToolProfile
from .contracts import AppliedState, Detection
from .features import extract_features


def single_window_proba(model, features: np.ndarray) -> np.ndarray:
    """The frozen binary forest, summed in sklearn's original tree order."""
    if not hasattr(model, "estimators_"):
        return model.predict_proba(features)
    # sklearn 1.9 _forest.py:946-962 and tree/_classes.py:1048-1053.
    # Validate/cast once; tree_.predict is the same leaf lookup without joblib.
    values = model._validate_X_predict(features)
    probabilities = np.zeros((len(values), model.n_classes_), dtype=np.float64)
    for tree in model.estimators_:
        probabilities += tree.tree_.predict(values)[:, : model.n_classes_]
    probabilities /= len(model.estimators_)
    return probabilities


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
            single_window_proba(self.model, np.asarray(features).reshape(1, -1))[0, 1]
        )
        return Detection(
            features,
            score,
            residual_rms=residual_rms(window, operating_state, self.profile),
        )
