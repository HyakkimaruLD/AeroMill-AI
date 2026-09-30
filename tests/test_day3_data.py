"""Split provenance, scheduled generation, and withheld-set tripwires."""

from collections import Counter
from dataclasses import fields
import importlib
import json

import numpy as np
import pytest


def data_module():
    from aeromill import data

    return data


def test_manifest_family_seed_separation():
    d = data_module()
    m = d.make_manifest()
    assert m == d.make_manifest()
    episodes = m["episodes"]
    assert len(episodes) == 300
    assert Counter(e["split"] for e in episodes) == {
        "train": 180,
        "validation": 60,
        "test": 60,
    }
    by_family = {}
    for e in episodes:
        by_family.setdefault(e["family_id"], []).append(e)
    assert len(by_family) == 60
    for group in by_family.values():
        assert len(group) == 5 and len({e["split"] for e in group}) == 1
        assert len({json.dumps(e["scenario"], sort_keys=True) for e in group}) == 1
        assert [s["time_s"] for s in group[0]["schedule"]] == [4, 8, 12]
    for scenario in d.BASE_SCENARIOS:
        groups = [g for g in by_family.values() if g[0]["base_scenario"] == scenario]
        assert sum(bool(g[0]["scenario"]["engagement"]) for g in groups) == 5
        assert Counter(g[0]["split"] for g in groups) == {
            "train": 6,
            "validation": 2,
            "test": 2,
        }
    all_entries = episodes + sum(m["end_to_end"].values(), [])
    seeds = [e["seed"] for e in all_entries]
    assert len(seeds) == len(set(seeds)) and max(seeds) < 900000
    families = [
        {e["family_id"] for e in episodes if e["split"] == s}
        for s in ("train", "validation", "test")
    ]
    for split, entries in m["end_to_end"].items():
        assert len(entries) == 60
        assert Counter(e["kind"] for e in entries) == {"stable": 30, "controllable": 30}
        for kind in ("stable", "controllable"):
            assert (
                sum(
                    bool(e["scenario"]["engagement"])
                    for e in entries
                    if e["kind"] == kind
                )
                >= 15
            )
        families.append({e["family_id"] for e in entries})
    for i, a in enumerate(families):
        for b in families[i + 1 :]:
            assert not a & b
    d.validate_manifest(m)
    m["episodes"][1]["seed"] = m["episodes"][0]["seed"]
    with pytest.raises(ValueError, match="seed"):
        d.validate_manifest(m)


def test_withheld_generation_and_metrics_refused(monkeypatch):
    d = data_module()
    m = d.make_manifest()

    def tripwire(*args, **kwargs):
        raise AssertionError("guard bypassed; simulator must never be constructed")

    monkeypatch.setattr(d, "Simulator", tripwire)
    for e in [
        next(e for e in m["episodes"] if e["split"] == "test"),
        m["end_to_end"]["final"][0],
        m["end_to_end"]["reserve"][0],
    ]:
        with pytest.raises(ValueError, match="withheld"):
            d.generate_episode(e)
    from aeromill import training

    for split in ("train", "test", "final", "reserve"):
        with pytest.raises(ValueError, match="validation"):
            training.window_metrics(np.array([0]), np.array([0.0]), split=split)


def test_binary_window_requires_all_samples():
    d = data_module()
    assert d.window_label(np.array(["stable"] * 2048)) == 0
    assert d.window_label(np.array(["chatter"] * 2048)) == 1
    assert d.window_label(np.array(["stable", "chatter"])) == -1
    assert d.window_label(np.array(["transition"] * 2048)) == -1


def test_final_cli_freeze_gate(capsys, monkeypatch):
    from aeromill.cli import main

    monkeypatch.setattr("sys.argv", ["aeromill", "evaluate", "--split", "final"])
    with pytest.raises(SystemExit):
        main()
    assert "freeze" in capsys.readouterr().err.lower()


def test_build_manifest_contains_artifact_hashes(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    d = data_module()
    generated = []

    def fake_episode(entry):
        assert (tmp_path / "manifest.json").is_file(), "manifest must precede signals"
        assert entry["split"] in ("train", "validation")
        generated.append(entry["seed"])
        return dict(
            features=np.zeros((4, 23)),
            label=np.array([0, 1, 0, 1]),
            regime=np.array(["before", "before", "after", "after"]),
            engagement=np.zeros(4, dtype=bool),
            family=np.full(4, entry["family_id"]),
            seed=np.full(4, entry["seed"]),
            split=np.full(4, entry["split"]),
        )

    monkeypatch.setattr(d, "generate_episode", fake_episode)
    monkeypatch.setattr(d, "ProcessPoolExecutor", ThreadPoolExecutor)
    d.build(tmp_path, workers=1)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert "artifact_hashes" in manifest, "artifact digests must be in the manifest"
    assert manifest["artifact_hashes"] == {
        s: d.file_hash(tmp_path / f"{s}.npz") for s in ("train", "validation")
    }
    assert len(generated) == 240
    assert manifest["config_sha256"] == d.config_hash(manifest)
    d.build(
        tmp_path, workers=1
    )  # Metadata enrichment must not prevent exact regeneration.
