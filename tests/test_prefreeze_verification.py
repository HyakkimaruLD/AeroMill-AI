"""A stable second correction must not be retried after a harmonic-band shift."""

from aeromill.data import make_manifest
from aeromill.validation import run_entry


def test_stable_validation_candidate_survives_residual_band_shift(tmp_path):
    entry = next(
        e for e in make_manifest()["end_to_end"]["validation"] if e["seed"] == 200005
    )
    row = run_entry(entry, "ml_agent", tmp_path)
    assert row["false_interventions"] == 0
    assert row["commands"] == 2
    assert row["run_success"]
    assert row["false_recoveries"] == row["unverified_recoveries"] == 0
