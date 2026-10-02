"""File integrity and single-use receipts for withheld evaluation."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path

from .data import file_hash

_ACTIVE = set()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def frozen_files(root=Path(".")):
    root = Path(root)
    sources = [
        p
        for p in (root / "aeromill").rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    ]
    return sorted(
        sources
        + [
            root / p
            for p in (
                "artifacts/model.pkl",
                "artifacts/model.json",
                "data/manifest.json",
                "pyproject.toml",
                "uv.lock",
            )
        ]
    )


def verify_freeze(root=Path(".")):
    root = Path(root)
    try:
        marker = root / "artifacts/FROZEN.json"
        hashes = json.loads(marker.read_text())["sha256"]
        current = {
            p.relative_to(root).as_posix(): file_hash(p) for p in frozen_files(root)
        }
        if current != hashes:
            raise ValueError("frozen file hash or scope mismatch")
        return file_hash(marker)
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("freeze marker or frozen file missing/invalid") from exc


def start_evaluation(split, output):
    marker_hash = verify_freeze()
    receipt = dict(split=split, started_utc=utc_now(), freeze_sha256=marker_hash)
    try:
        with Path(f"artifacts/{split.upper()}_STARTED.json").open("x") as f:
            json.dump(receipt, f, indent=2)
            f.write("\n")
    except FileExistsError as exc:
        raise ValueError(f"{split} evaluation already attempted; no reruns") from exc
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"{split}-receipt.json").open("x") as f:
        json.dump(receipt, f, indent=2)
        f.write("\n")
    return receipt


def require_active(split):
    verify_freeze()
    if split not in _ACTIVE:
        raise ValueError(f"{split} requires the single-use evaluation entry point")


@contextmanager
def evaluation_session(split, output):
    receipt = start_evaluation(split, output)
    _ACTIVE.add(split)
    try:
        yield receipt
    finally:
        _ACTIVE.discard(split)
