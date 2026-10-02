"""Validation-only experiment orchestration, kept outside truth grading."""

from contextlib import nullcontext
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .config import ToolProfile
from .data import file_hash, scenario_from_dict, write_json
from .engine import Engine
from .evaluation import RunEvaluator
from .memory import Memory

MODES = ("no_adaptation", "baseline", "threshold_search", "ml_agent")


def run_entry(entry, mode, output, memory=None):
    if entry["split"] == "final":
        from .freeze import require_active

        require_active("final")
    elif entry["split"] != "validation":
        raise ValueError("only validation is enabled")
    label = mode + ("_warm" if memory is not None else "")
    engine = None
    error = None
    try:
        engine = Engine(
            scenario_from_dict(entry["scenario"]),
            seed=entry["seed"],
            mode=mode,
            profile=ToolProfile(**entry["profile"]),
            noise_std=entry["noise_std"],
            chatter_hz=entry["chatter_hz"],
            initial_phase=entry["initial_phase"],
            memory=memory,
            log_path=Path(output) / f"{entry['family_id']}-{label}.jsonl",
        )
        engine.run()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if engine is not None:
            engine.agent.log = None
            engine.stop("RUN_ERROR")
    metrics = (
        engine.evaluate()
        if engine is not None
        else RunEvaluator().evaluate(
            [], state="HOLD", path_length_mm=entry["scenario"]["path_length_mm"]
        )
    )
    timings = engine.processing_ms if engine is not None else []
    return dict(
        metrics,
        family_id=entry["family_id"],
        seed=entry["seed"],
        kind=entry["kind"],
        scenario=entry["scenario"]["name"],
        engagement=bool(entry["scenario"]["engagement"]),
        mode=label,
        reason=engine.agent.reason if engine is not None else "STARTUP_ERROR",
        error=error,
        run_id=engine.run_id if engine is not None else None,
        memory_hits=sum(e["event"] == "memory_hit" for e in engine.events)
        if engine is not None
        else 0,
        processing_p95_ms=float(np.percentile(timings, 95)) if timings else None,
        processing_p99_ms=float(np.percentile(timings, 99)) if timings else None,
    )


def run_group(job):
    entries, mode, output, warm = job
    rows = []
    for entry in entries:
        memory = Memory() if warm else None
        primer = None
        if warm:
            primer = run_entry(entry, mode, Path(output) / "priming", memory)
            primer["mode"] = mode + "_priming"
        if entry.get("split") == "final":
            from .freeze import utc_now

            with (Path(output) / "run-attempts.jsonl").open("a") as journal:
                journal.write(
                    json.dumps(
                        dict(
                            family_id=entry["family_id"],
                            seed=entry["seed"],
                            mode=mode,
                            started_utc=utc_now(),
                        )
                    )
                    + "\n"
                )
                journal.flush()
        try:
            row = run_entry(entry, mode, output, memory)
        except Exception as exc:
            if entry.get("split") == "final":
                with (Path(output) / "raw-runs.jsonl").open("a") as journal:
                    journal.write(
                        json.dumps(
                            dict(
                                family_id=entry["family_id"],
                                seed=entry["seed"],
                                mode=mode,
                                error=f"{type(exc).__name__}: {exc}",
                                run_success=False,
                            )
                        )
                        + "\n"
                    )
            raise
        if primer is not None:
            row["priming_run"] = primer
        rows.append(row)
        if entry.get("split") == "final":
            with (Path(output) / "raw-runs.jsonl").open("a") as journal:
                journal.write(json.dumps(row, allow_nan=False) + "\n")
                journal.flush()
            if row["error"] is not None:
                raise RuntimeError(
                    f"final run crashed: {row['family_id']} {mode}: {row['error']}"
                )
        print(
            f"{row['mode']} {len(rows)}/{len(entries)} {row['family_id']} success={row['run_success']}",
            flush=True,
        )
    return rows


def summarize(rows):
    totals = {}
    for mode in dict.fromkeys(r["mode"] for r in rows):
        selected = [r for r in rows if r["mode"] == mode]
        stable = [r for r in selected if r["kind"] == "stable"]
        controlled = [r for r in selected if r["kind"] == "controllable"]
        count = sum(r["incidents"] for r in controlled)
        recover = [
            r["rms_reduction"]
            for r in controlled
            if r["scenario"] == "recover_a" and r["rms_reduction"] is not None
        ]
        productivity = [
            r["productivity_loss"]
            for r in selected
            if r["productivity_loss"] is not None
        ]
        observed_s = sum(r["time_s"] for r in selected)
        totals[mode] = dict(
            runs=len(selected),
            controllable_runs=len(controlled),
            stable_runs=len(stable),
            controllable_successes=sum(r["run_success"] for r in controlled),
            stable_without_intervention=sum(
                r["stable_without_intervention"] for r in stable
            ),
            incidents=count,
            missed_incidents=sum(r["missed_incidents"] for r in controlled),
            detection_within_1s_fraction=sum(
                r["detected_within_1s"] for r in controlled
            )
            / count
            if count
            else None,
            false_recoveries=sum(r["false_recoveries"] for r in selected),
            unverified_recoveries=sum(r["unverified_recoveries"] for r in selected),
            recover_a_rms_reduction_min=min(recover) if recover else None,
            recover_a_rms_measured_runs=len(recover),
            processing_p95_ms_max=max(
                (
                    r["processing_p95_ms"]
                    for r in selected
                    if r["processing_p95_ms"] is not None
                ),
                default=None,
            ),
            false_incidents_per_1000_s=1000
            * sum(r["false_incidents"] for r in selected)
            / observed_s
            if observed_s
            else None,
            mean_productivity_loss_completed=float(np.mean(productivity))
            if productivity
            else None,
            completed=sum(r["completed"] for r in selected),
            stops=sum(r["stops"] for r in selected),
        )
    return totals


def evaluate_validation(
    *,
    split="validation",
    mode="all",
    warm_memory=False,
    output=None,
    workers=1,
):
    if output is None:
        output = "artifacts/final" if split == "final" else "artifacts/day4-validation"
    session = nullcontext(None)
    if split == "final":
        from .freeze import evaluation_session

        if mode != "all" or warm_memory or workers != 1:
            raise ValueError("final requires all four cold modes and one worker")
        session = evaluation_session(split, output)
    elif split != "validation":
        raise ValueError("only validation is enabled; withheld sets remain disabled")
    with session as receipt:
        modes = list(MODES) if mode == "all" else [mode]
        if any(m not in MODES for m in modes):
            raise ValueError("unknown mode")
        if warm_memory and modes != ["threshold_search"]:
            raise ValueError("warm-memory experiment requires threshold_search alone")
        manifest_path = Path("data/manifest.json")
        manifest = json.loads(manifest_path.read_text())
        entries = manifest["end_to_end"][split]
        if len(entries) != 60 or any(e["split"] != split for e in entries):
            raise ValueError("invalid validation manifest")
        output = Path(output)
        if (
            warm_memory
            and output.resolve() == Path("artifacts/day4-validation").resolve()
        ):
            output = output / "warm"
        output.mkdir(parents=True, exist_ok=True)
        cold = None
        if warm_memory:
            cold = json.loads(
                Path("artifacts/day4-validation/results.json").read_text()
            )
            if (
                cold["manifest_sha256"] != file_hash(manifest_path)
                or cold["model_sha256"] != file_hash("artifacts/model.pkl")
                or cold["warm_memory"]
            ):
                raise ValueError(
                    "warm experiment requires matching cold validation provenance"
                )
        started = perf_counter()
        jobs = [(entries, m, output, warm_memory) for m in modes]
        if workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
                groups = list(pool.map(run_group, jobs))
        else:
            groups = [run_group(j) for j in jobs]
        rows = [row for group in groups for row in group]
        baselines = {
            r["family_id"]: r["traversal_time_s"]
            for r in rows
            if r["mode"] == "no_adaptation"
        }
        if cold is not None:
            baselines = {
                r["family_id"]: r["traversal_time_s"]
                for r in cold["runs"]
                if r["mode"] == "no_adaptation"
            }
        for row in rows:
            baseline = baselines.get(row["family_id"])
            time = row["traversal_time_s"]
            row["productivity_loss"] = (
                None if not baseline or time is None else time / baseline - 1
            )
        result = dict(
            split=split,
            warm_memory=warm_memory,
            seconds=perf_counter() - started,
            manifest_sha256=file_hash(manifest_path),
            model_sha256=file_hash("artifacts/model.pkl"),
            runs=rows,
            summary=summarize(rows),
            memory_protocol="explicit priming run then warm run of identical entry; shared Memory only"
            if warm_memory
            else "cold per run",
            limitations=[
                "Synthetic simulation; not validated on real CNC.",
                "Baseline remains resonant by construction; only threshold_search controls for search policy.",
                "RF advantage is versus total RMS only, not a spectral-rule comparator.",
            ],
        )
        if receipt is not None:
            from .freeze import utc_now, verify_freeze

            result["provenance"] = dict(
                receipt, finished_utc=utc_now(), freeze_sha256_after=verify_freeze()
            )
        write_json(output / "results.json", result)
        with (output / "runs.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(
                {
                    k: json.dumps(v) if isinstance(v, (dict, list)) else v
                    for k, v in r.items()
                }
                for r in rows
            )
        print(json.dumps(result["summary"], indent=2), flush=True)
        return result
