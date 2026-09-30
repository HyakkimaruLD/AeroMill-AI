"""RF artifact integrity and the exact online feature boundary."""

from pathlib import Path

import numpy as np
import pytest

from aeromill.config import ToolProfile
from aeromill.contracts import AppliedState
from aeromill.features import extract_features


def test_rf_predict_boundary():
    from aeromill import detector

    assert hasattr(detector, "RFDetector")

    class Model:
        def predict_proba(self, x):
            self.seen = x.copy()
            return np.array([[0.2, 0.8]])

    model = Model()
    d = detector.RFDetector(ToolProfile(), model=model)
    t = np.arange(2048) / 8192
    x = np.sin(2 * np.pi * 213 * t)[:, None] * np.array([1, 0.8, 0.6])
    found = d.predict(x, AppliedState())
    assert found.chatter_score == 0.8 and len(found.residual_rms) == 3
    assert np.array_equal(
        model.seen[0], extract_features(x, AppliedState(), ToolProfile())
    )


def test_corrupt_model_prevents_startup(tmp_path):
    import json
    import pickle
    import sklearn
    from sklearn.ensemble import RandomForestClassifier
    from aeromill.data import VERSIONS, file_hash
    from aeromill.features import FEATURE_NAMES
    from aeromill.training import load_model

    model = RandomForestClassifier(n_estimators=2, random_state=42).fit(
        np.zeros((4, 23)), [0, 1, 0, 1]
    )
    (tmp_path / "model.pkl").write_bytes(pickle.dumps(model))
    metadata = dict(
        feature_order=list(FEATURE_NAMES),
        versions=VERSIONS,
        sklearn=sklearn.__version__,
        classes=[0, 1],
        model_sha256=file_hash(tmp_path / "model.pkl"),
    )
    (tmp_path / "model.json").write_text(json.dumps(metadata))
    assert load_model(tmp_path).n_features_in_ == 23
    metadata["model_sha256"] = "0" * 64
    (tmp_path / "model.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="model"):
        load_model(tmp_path)
