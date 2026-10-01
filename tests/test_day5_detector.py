"""The fast path must keep sklearn's accumulation order and input conversion."""

import numpy as np

from aeromill import detector
from aeromill.training import load_model


def test_single_window_matches_sklearn():
    assert callable(getattr(detector, "single_window_proba", None))
    model = load_model()
    rng = np.random.default_rng(42)
    rows = rng.uniform(-2, 20, size=(100, 23))
    rows = np.vstack((rows, np.zeros((1, 23)), np.ones((1, 23))))
    expected = model.predict_proba(rows)
    actual = np.concatenate(
        [detector.single_window_proba(model, row[None, :]) for row in rows]
    )
    assert np.array_equal(expected, actual)
