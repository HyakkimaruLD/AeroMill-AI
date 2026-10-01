"""Only validation may construct an engine; failures stay in the denominator."""

import importlib
from unittest.mock import patch
import pytest


def test_validation_runner_refuses_withheld_before_engine():
    m = importlib.import_module("aeromill.evaluation")
    from aeromill import validation

    assert hasattr(m, "evaluate_validation")
    for split in ("test", "final", "reserve", "train"):
        with patch.object(
            validation, "Engine", side_effect=AssertionError("must not start")
        ):
            with pytest.raises(ValueError, match="validation"):
                m.evaluate_validation(split=split)


def test_cli_final_guard_and_validation_route(monkeypatch, capsys):
    from aeromill import cli, evaluation

    with patch(
        "sys.argv", ["aeromill", "evaluate", "--split", "final", "--mode", "all"]
    ):
        with pytest.raises(SystemExit):
            cli.main()
    assert "freeze marker" in capsys.readouterr().err
    assert hasattr(evaluation, "evaluate_validation")
    with patch.object(
        evaluation, "evaluate_validation", return_value={"ok": True}
    ) as runner:
        with patch(
            "sys.argv",
            ["aeromill", "evaluate", "--split", "validation", "--mode", "all"],
        ):
            cli.main()
        assert runner.call_args.kwargs["split"] == "validation"


def test_warm_uses_same_memory_after_explicit_priming(tmp_path, monkeypatch):
    from aeromill import validation

    key = ("p", "zone", "rms-v1", "v1")

    def run(entry, mode, output, memory):
        reused = memory.get(key) is not None
        memory.remember(key, (2880.0, 960.0), "v", 4.0)
        return dict(mode=mode, family_id="f", run_success=True, reused=reused)

    monkeypatch.setattr(validation, "run_entry", run)
    rows = validation.run_group(([{}], "threshold_search", tmp_path, True))
    assert rows[0]["reused"] and not rows[0]["priming_run"]["reused"]
    assert rows[0]["priming_run"]["mode"] == "threshold_search_priming"


def test_failed_run_stays_in_results(tmp_path, monkeypatch):
    from aeromill import validation
    from aeromill.data import make_manifest

    entry = make_manifest()["end_to_end"]["validation"][0]

    def broken(*args, **kwargs):
        raise ValueError("injected startup failure")

    monkeypatch.setattr(validation, "Engine", broken)
    row = validation.run_entry(entry, "ml_agent", tmp_path)
    assert row["family_id"] == entry["family_id"] and not row["run_success"]
    assert row["stops"] == 1 and row["error"] == "ValueError: injected startup failure"
    row["productivity_loss"] = None
    summary = validation.summarize([row])["ml_agent"]
    assert summary["runs"] == 1 and summary["stops"] == 1 and summary["completed"] == 0
    assert summary["false_incidents_per_1000_s"] is None
    result = validation.evaluate_validation(mode="ml_agent", output=tmp_path)
    assert len(result["runs"]) == 60 and all(
        not r["run_success"] for r in result["runs"]
    )
    assert (tmp_path / "results.json").is_file()
    assert len((tmp_path / "runs.csv").read_text().splitlines()) == 61


def test_deterministic_regrade_matches_actual_command_run(tmp_path):
    import importlib.util
    from pathlib import Path
    from aeromill.data import make_manifest
    from aeromill.validation import run_entry

    path = Path(__file__).parent / "evidence" / "day4_regrade.py"
    spec = importlib.util.spec_from_file_location("day4_regrade", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    entry = make_manifest()["end_to_end"]["validation"][0]
    row = run_entry(entry, "threshold_search", tmp_path)
    assert row["commands"] > 0
    result = module.regrade(
        entry, row, tmp_path / f"{entry['family_id']}-threshold_search.jsonl"
    )
    assert all(row[k] == value for k, value in result.items())
    import json

    log = tmp_path / f"{entry['family_id']}-threshold_search.jsonl"
    events = [json.loads(line) for line in log.read_text().splitlines()]
    tick = next(e for e in events if e["event"] == "tick")
    for broken in ([e for e in events if e is not tick], events + [tick]):
        path = tmp_path / "broken.jsonl"
        path.write_text("\n".join(json.dumps(e) for e in broken))
        with pytest.raises(AssertionError, match="tick"):
            module.regrade(entry, row, path)
