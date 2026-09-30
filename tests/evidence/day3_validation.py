"""Day3 validation evidence only; deliberately not the deferred evaluator module."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from aeromill.config import ToolProfile
from aeromill.data import file_hash, scenario_from_dict, write_json
from aeromill.engine import Engine


def confirm_claim(time_s, times, envelope, position, moving):
    mask = (times > time_s) & (times <= time_s + 3)
    if not len(times) or times[-1] < time_s + 3 or mask.sum() < 24575:
        return "unverified"
    if not (
        np.all(envelope[mask] <= 0.25)
        and moving[mask].all()
        and np.all(np.diff(position[mask]) > 0)
    ):
        return "false"
    return "confirmed"


def run_one(job):
    entry, mode = job
    if entry["split"] != "validation":
        raise ValueError("only validation runs are allowed")
    run_name = f"{entry['family_id']}-{mode}"
    engine = Engine(
        scenario_from_dict(entry["scenario"]),
        seed=entry["seed"],
        mode=mode,
        profile=ToolProfile(**entry["profile"]),
        noise_std=entry["noise_std"],
        chatter_hz=entry["chatter_hz"],
        initial_phase=entry["initial_phase"],
        model_dir=PROJECT / "artifacts",
        log_path=PROJECT / "artifacts" / "day3-validation" / f"{run_name}.jsonl",
    )
    chunks = []
    while not engine.done:
        step = engine.tick()
        chunks.append(
            (
                step.telemetry.time_s,
                step.truth.envelope,
                step.truth.x_mm,
                step.truth.moving,
            )
        )
    times, envelope, position, moving = [
        np.concatenate([c[i] for c in chunks]) for i in range(4)
    ]
    claims = [e for e in engine.agent.events if e["event"] == "recovery_claim"]
    confirmations = Counter(
        confirm_claim(c["time_s"], times, envelope, position, moving) for c in claims
    )
    commands = [
        e for e in engine.agent.events if e["event"] == "set_cutting_parameters"
    ]
    false_commands = 0
    for command in commands:
        # Match the causal detector window; hidden truth is used only in this proof script.
        mask = (times > command["time_s"] - 0.25) & (times <= command["time_s"])
        false_commands += bool(mask.any() and np.all(envelope[mask] <= 0.25))
    return dict(
        family_id=entry["family_id"],
        seed=entry["seed"],
        kind=entry["kind"],
        engagement=bool(entry["scenario"]["engagement"]),
        mode=mode,
        state=engine.agent.state,
        reason=engine.agent.reason,
        commands=len(commands),
        incidents=engine.agent.incident,
        completed=int(engine.agent.state == "COMPLETED"),
        holds=int(engine.agent.state == "HOLD"),
        confirmed_recoveries=confirmations["confirmed"],
        false_recoveries=confirmations["false"],
        unverified_recoveries=confirmations["unverified"],
        false_interventions=false_commands,
        time_s=engine.time_s,
        processing_p95_ms=float(np.percentile(engine.processing_ms, 95)),
    )


def main():
    manifest_path = PROJECT / "data" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    jobs = [
        (e, mode)
        for mode in ("threshold_search", "ml_agent")
        for e in manifest["end_to_end"]["validation"]
    ]
    started = perf_counter()
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = []
        for row in pool.map(run_one, jobs):
            rows.append(row)
            if len(rows) % 10 == 0:
                print(f"validation runs: {len(rows)}/{len(jobs)}", flush=True)
    keys = (
        "completed",
        "holds",
        "commands",
        "incidents",
        "confirmed_recoveries",
        "false_recoveries",
        "unverified_recoveries",
        "false_interventions",
    )
    totals = {}
    for mode in ("threshold_search", "ml_agent"):
        totals[mode] = {}
        for kind in ("all", "stable", "controllable"):
            selected = [
                r
                for r in rows
                if r["mode"] == mode and (kind == "all" or r["kind"] == kind)
            ]
            totals[mode][kind] = {k: sum(r[k] for r in selected) for k in keys}
    result = dict(
        manifest_sha256=file_hash(manifest_path),
        model_sha256=file_hash(PROJECT / "artifacts" / "model.pkl"),
        split="validation",
        seconds=perf_counter() - started,
        totals=totals,
        runs=rows,
        limitation="Counts are not run_success: the full incident-matching evaluator is deferred.",
    )
    write_json(ROOT / "day3-validation.json", result)
    print(json.dumps(totals, indent=2), flush=True)


if __name__ == "__main__":
    main()
