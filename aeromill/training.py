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
    if split != "validation":
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


def validation_report(data, scores):
    if not np.all(data["split"] == "validation"):
        raise ValueError("metrics require validation split")
    result = {
        "threshold": 0.8,
        "overall": window_metrics(data["label"], scores, split="validation"),
    }
    for column in ("regime", "engagement", "family"):
        result[column] = {
            str(value): window_metrics(
                data["label"][data[column] == value],
                scores[data[column] == value],
                split="validation",
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
