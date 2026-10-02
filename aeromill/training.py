"""Fixed RF fit on train; validation-only metrics and checked local artifacts."""

import json
from pathlib import Path
import pickle
from time import perf_counter

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier

from .data import VERSIONS, file_hash, write_json
from .features import FEATURE_NAMES


def window_metrics(labels, scores, *, split, threshold=0.8):
    if split == "test":
        from .freeze import require_active

        require_active("test")
    elif split != "validation":
        raise ValueError("metrics require validation split")
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    positive = labels == 1
    negative = labels == 0
    predicted = scores >= threshold
    return dict(
        chatter_windows=int(positive.sum()),
        stable_windows=int(negative.sum()),
        recall=float(predicted[positive].mean()) if positive.any() else None,
        fpr=float(predicted[negative].mean()) if negative.any() else None,
    )


def validation_report(data, scores, *, split="validation"):
    if not np.all(data["split"] == split):
        raise ValueError("metrics require validation split")
    result = {
        "threshold": 0.8,
        "overall": window_metrics(data["label"], scores, split=split),
    }
    for column in ("regime", "engagement", "family"):
        result[column] = {
            str(value): window_metrics(
                data["label"][data[column] == value],
                scores[data[column] == value],
                split=split,
            )
            for value in np.unique(data[column])
        }
    return result


def train(directory=Path("data"), output=Path("artifacts")):
    started = perf_counter()
    directory = Path(directory)
    output = Path(output)
    build = json.loads((directory / "build.json").read_text())
    if file_hash(directory / "manifest.json") != build["manifest_sha256"]:
        raise ValueError("manifest hash mismatch")
    datasets = {}
    for split in ("train", "validation"):
        path = directory / f"{split}.npz"
        if file_hash(path) != build["artifact_hashes"][split]:
            raise ValueError("dataset hash mismatch")
        with np.load(path, allow_pickle=False) as f:
            datasets[split] = {k: f[k] for k in f.files}
        if not np.all(datasets[split]["split"] == split):
            raise ValueError("dataset split mismatch")
    data = datasets["train"]
    mask = data["label"] >= 0
    # Installed sklearn _forest.py documents balanced weights over training y.
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )
    model.fit(data["features"][mask], data["label"][mask])
    if list(model.classes_) != [0, 1]:
        raise ValueError("both binary classes required")
    validation = datasets["validation"]
    scores = model.predict_proba(validation["features"])[:, 1]
    report = validation_report(validation, scores)
    # Comparator uses the identical profile-normalized feature windows.
    report["total_rms"] = validation_report(
        validation,
        (np.max(validation["features"][:, :21:7], axis=1) > 1.5).astype(float),
    )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "model.pkl").open("wb") as f:
        pickle.dump(model, f, protocol=5)
    metadata = dict(
        versions=VERSIONS,
        feature_order=FEATURE_NAMES,
        sklearn=sklearn.__version__,
        model_sha256=file_hash(output / "model.pkl"),
        manifest_sha256=build["manifest_sha256"],
        data_hashes=build["artifact_hashes"],
        training_seconds=perf_counter() - started,
        parameters=model.get_params(),
        classes=[0, 1],
    )
    write_json(output / "model.json", metadata)
    write_json(output / "validation_metrics.json", report)
    print(
        json.dumps(
            dict(
                validation=report["overall"],
                total_rms=report["total_rms"]["overall"],
                training_seconds=metadata["training_seconds"],
            )
        ),
        flush=True,
    )
    return report


def load_model(directory=Path("artifacts")):
    directory = Path(directory)
    try:
        meta = json.loads((directory / "model.json").read_text())
        if (
            meta["feature_order"] != list(FEATURE_NAMES)
            or meta["versions"] != VERSIONS
            or meta["sklearn"] != sklearn.__version__
            or meta["classes"] != [0, 1]
            or meta["model_sha256"] != file_hash(directory / "model.pkl")
        ):
            raise ValueError("incompatible model metadata or hash")
        # Only locally generated, hash-checked artifacts; pickle is not an untrusted interchange format.
        with (directory / "model.pkl").open("rb") as f:
            model = pickle.load(f)
        if model.n_features_in_ != len(FEATURE_NAMES) or list(model.classes_) != [0, 1]:
            raise ValueError("invalid model feature/class contract")
        return model
    except Exception as exc:
        raise ValueError(f"cannot load model: {exc}") from exc


def evaluate_test(output="artifacts/final"):
    """Generate the fixed test episodes once, without fitting or changing labels."""
    from .data import generate_episode
    from .freeze import evaluation_session, utc_now, verify_freeze

    output = Path(output)
    with evaluation_session("test", output) as receipt:
        manifest = json.loads(Path("data/manifest.json").read_text())
        entries = [e for e in manifest["episodes"] if e["split"] == "test"]
        model = load_model()
        parts = []
        (output / "test-episodes").mkdir()
        with (output / "test-episodes.jsonl").open("x") as journal:
            for entry in entries:
                with (output / "test-attempts.jsonl").open("a") as attempts:
                    attempts.write(
                        json.dumps(
                            dict(
                                family_id=entry["family_id"],
                                seed=entry["seed"],
                                started_utc=utc_now(),
                            )
                        )
                        + "\n"
                    )
                part = generate_episode(entry)
                part["score"] = model.predict_proba(part["features"])[:, 1]
                np.savez_compressed(
                    output / "test-episodes" / f"{entry['seed']}.npz", **part
                )
                parts.append(part)
                journal.write(
                    json.dumps(
                        dict(
                            family_id=entry["family_id"],
                            seed=entry["seed"],
                            windows=len(part["label"]),
                        )
                    )
                    + "\n"
                )
                journal.flush()
                print(
                    f"test {len(parts)}/{len(entries)} seed={entry['seed']}", flush=True
                )
        data = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
        np.savez_compressed(output / "test-windows.npz", **data)
        report = validation_report(data, data["score"], split="test")
        report["transitional_windows"] = int(np.count_nonzero(data["label"] < 0))
        report["provenance"] = dict(
            receipt, finished_utc=utc_now(), freeze_sha256_after=verify_freeze()
        )
        write_json(output / "window-metrics.json", report)
        print(json.dumps(report, indent=2), flush=True)
        return report
