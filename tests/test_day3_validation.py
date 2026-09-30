"""The temporary Day3 evidence script must never run withheld entries."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def script():
    path = Path(__file__).parent / "evidence" / "day3_validation.py"
    assert path.exists(), "validation evidence runner missing"
    spec = importlib.util.spec_from_file_location("day3_validation", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_validation_refuses_withheld_before_engine():
    m = script()
    for split in ("train", "test", "final", "reserve"):
        with pytest.raises(ValueError, match="validation"):
            m.run_one(({"split": split}, "ml_agent"))


def test_confirmation_full_horizon_and_motion():
    m = script()
    t = np.arange(1, 32769) / 8192
    a = np.full(t.shape, 0.1)
    moving = np.ones(t.shape, dtype=bool)
    x = t * 20
    assert m.confirm_claim(1.0, t, a, x, moving) == "confirmed"
    assert m.confirm_claim(2.0, t, a, x, moving) == "unverified"
    a[17000] = 1.0
    assert m.confirm_claim(1.0, t, a, x, moving) == "false"
    a[:] = 0.1
    moving[17000] = False
    assert m.confirm_claim(1.0, t, a, x, moving) == "false"


def test_family_and_split_not_encoded_in_agent_run_id(monkeypatch):
    from aeromill.data import make_manifest

    m = script()
    entry = make_manifest()["end_to_end"]["validation"][0]

    class StoppedBeforeMotion(Exception):
        pass

    def engine_tripwire(*args, **kwargs):
        assert "run_id" not in kwargs, "engine must assign an opaque run ID"
        raise StoppedBeforeMotion

    monkeypatch.setattr(m, "Engine", engine_tripwire)
    with pytest.raises(StoppedBeforeMotion):
        m.run_one((entry, "ml_agent"))
