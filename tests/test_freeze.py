"""A changed frozen file must block both withheld entry points before work."""

import json
from pathlib import Path

import pytest


def test_tamper_blocks_withheld_evaluation(tmp_path, monkeypatch, capsys):
    from aeromill import cli
    from aeromill.data import file_hash

    monkeypatch.chdir(tmp_path)
    (tmp_path / "artifacts").mkdir()
    source = tmp_path / "aeromill" / "agent.py"
    source.parent.mkdir()
    source.write_text("original\n")
    from aeromill.freeze import frozen_files, verify_freeze

    (tmp_path / "data").mkdir()
    for name in (
        "artifacts/model.pkl",
        "artifacts/model.json",
        "data/manifest.json",
        "pyproject.toml",
        "uv.lock",
    ):
        (tmp_path / name).write_text("fixture\n")
    marker = {
        "sha256": {
            p.relative_to(tmp_path).as_posix(): file_hash(p)
            for p in frozen_files(tmp_path)
        }
    }
    (tmp_path / "artifacts/FROZEN.json").write_text(json.dumps(marker))
    assert verify_freeze() == file_hash("artifacts/FROZEN.json")
    source.write_text("changed\n")
    for split in ("final", "test"):
        monkeypatch.setattr("sys.argv", ["aeromill", "evaluate", "--split", split])
        with pytest.raises(SystemExit):
            cli.main()
        assert "frozen file" in capsys.readouterr().err
        # A stale marker cannot authorize even an attempted evaluation.
        assert not Path(f"artifacts/{split.upper()}_STARTED.json").exists()


def test_single_use_receipt_survives_failure(tmp_path, monkeypatch):
    from aeromill import freeze

    monkeypatch.chdir(tmp_path)
    Path("artifacts").mkdir()
    monkeypatch.setattr(freeze, "verify_freeze", lambda: "fixture-digest")
    freeze.start_evaluation("final", "artifacts/output")
    with pytest.raises(ValueError, match="already attempted"):
        freeze.start_evaluation("final", "artifacts/different-output")


def test_final_crash_row_is_saved_and_stops_group(tmp_path, monkeypatch):
    from aeromill import validation

    calls = []

    def crash(entry, mode, output, memory):
        calls.append(entry)
        return dict(
            mode=mode, family_id="fixture", error="injected crash", run_success=False
        )

    monkeypatch.setattr(validation, "run_entry", crash)
    with pytest.raises(RuntimeError, match="final run crashed"):
        validation.run_group(
            (
                [{"split": "final", "family_id": "fixture", "seed": 42}] * 2,
                "ml_agent",
                tmp_path,
                False,
            )
        )
    assert len(calls) == 1
    rows = (tmp_path / "raw-runs.jsonl").read_text().splitlines()
    assert len(rows) == 1 and json.loads(rows[0])["error"] == "injected crash"


def test_final_grading_crash_keeps_attempt_and_error(tmp_path, monkeypatch):
    from aeromill import validation

    def broken(*args):
        raise RuntimeError("grading failed")

    monkeypatch.setattr(validation, "run_entry", broken)
    entry = dict(split="final", family_id="fixture", seed=42)
    with pytest.raises(RuntimeError, match="grading failed"):
        validation.run_group(([entry, entry], "ml_agent", tmp_path, False))
    attempts = (tmp_path / "run-attempts.jsonl").read_text().splitlines()
    rows = (tmp_path / "raw-runs.jsonl").read_text().splitlines()
    assert len(attempts) == len(rows) == 1
    assert json.loads(rows[0])["error"] == "RuntimeError: grading failed"


def test_direct_withheld_helpers_require_active_evaluation(monkeypatch):
    from aeromill import data, freeze, training, validation

    monkeypatch.setattr(freeze, "verify_freeze", lambda: "fixture-digest")
    monkeypatch.setattr(freeze, "_ACTIVE", set())
    for action in (
        lambda: data.generate_episode({"split": "test"}),
        lambda: training.window_metrics([0], [0.0], split="test"),
        lambda: validation.run_entry({"split": "final"}, "ml_agent", "."),
    ):
        with pytest.raises(ValueError, match="single-use"):
            action()


@pytest.mark.parametrize("crash", [False, True])
def test_evaluation_authorization_expires(tmp_path, monkeypatch, crash):
    from aeromill import freeze

    monkeypatch.chdir(tmp_path)
    Path("artifacts").mkdir()
    monkeypatch.setattr(freeze, "verify_freeze", lambda: "fixture-digest")
    monkeypatch.setattr(freeze, "_ACTIVE", set())
    try:
        with freeze.evaluation_session("test", "artifacts/output"):
            freeze.require_active("test")
            if crash:
                raise RuntimeError("injected failure")
    except RuntimeError:
        pass
    with pytest.raises(ValueError, match="single-use"):
        freeze.require_active("test")
