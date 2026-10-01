"""Regrade recorded validation commands using deterministic simulator replay.

Checks every recorded tick's public trajectory. No detector is invoked, and no
recorded score/outcome is used as truth. Refuses ambiguous multi-run logs.
"""

import csv
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from aeromill.config import ToolProfile
from aeromill.contracts import AppliedState
from aeromill.data import file_hash, scenario_from_dict, write_json
from aeromill.evaluation import RunEvaluator
from aeromill.simulator import Simulator
from aeromill.validation import summarize


def regrade(entry, row, log):
    if entry["split"] != "validation":
        raise ValueError("validation only")
    events = [json.loads(line) for line in Path(log).read_text().splitlines()]
    assert len({e["run_id"] for e in events}) == 1, "ambiguous appended log"
    header = events[0]
    assert header["seed"] == entry["seed"] and header["mode"] == row["mode"]
    commands = {
        e["command_id"]: e for e in events if e["event"] == "set_cutting_parameters"
    }
    acks = {
        round(e["time_s"] * 10): commands[e["command_id"]]
        for e in events
        if e["event"] == "ack" and e["status"] == "accepted"
    }
    ticks = {round(e["time_s"] * 10): e for e in events if e["event"] == "tick"}
    stops = {round(e["time_s"] * 10) for e in events if e["event"] == "feed_hold"}
    count = round(row["time_s"] * 10)
    assert len(ticks) == sum(e["event"] == "tick" for e in events) == count and set(
        ticks
    ) == set(range(1, count + 1)), "complete unique tick records required"
    assert len(acks) == sum(
        e["event"] == "ack" and e["status"] == "accepted" for e in events
    ), "duplicate ACK tick"
    assert (
        len(commands)
        == sum(e["event"] == "set_cutting_parameters" for e in events)
        == row["commands"]
    ), "command count mismatch"
    terminal = [
        e
        for e in events
        if e["event"] == "transition" and e["state"] in ("HOLD", "COMPLETED")
    ]
    assert (
        len(terminal) == 1
        and terminal[0]["state"] == row["state"]
        and round(terminal[0]["time_s"] * 10) == count
    ), "terminal mismatch"
    # This evidence runner is restricted to normal cold validation, without faults.
    assert all(
        e["reason"]
        in ("ATTEMPTS_EXHAUSTED", "NO_CANDIDATE", "CALIBRATION_FAILED", "RUN_TIMEOUT")
        for e in events
        if e["event"] == "feed_hold"
    ), "unsupported priority/fault Stop"
    sim = Simulator(
        scenario_from_dict(entry["scenario"]),
        seed=entry["seed"],
        profile=ToolProfile(**entry["profile"]),
        noise_std=entry["noise_std"],
        chatter_hz=entry["chatter_hz"],
        initial_phase=entry["initial_phase"],
    )
    collector = RunEvaluator()
    target = AppliedState()
    for tick in range(1, count + 1):
        if tick in acks:
            command = acks[tick]
            target = AppliedState(command["rpm"], command["feed_mm_min"])
        step = sim.step(0.1, target)
        collector.append(step)
        record = ticks[tick]
        t = step.telemetry
        assert np.allclose(
            [t.xyz_mm[-1, 0], t.rpm[-1], t.feed_mm_min[-1]],
            [record["x_mm"], record["rpm"], record["feed_mm_min"]],
            rtol=0,
            atol=1e-9,
        )
        if tick in stops:
            target = AppliedState(target.rpm, 0.0, True)
            sim.stop_at_boundary(target)
    return collector.evaluate(
        events, state=row["state"], path_length_mm=entry["scenario"]["path_length_mm"]
    )


def main():
    directory = PROJECT / "artifacts" / "day4-validation"
    result = json.loads((directory / "results.json").read_text())
    manifest = json.loads((PROJECT / "data" / "manifest.json").read_text())
    assert result["manifest_sha256"] == file_hash(PROJECT / "data" / "manifest.json")
    assert result["model_sha256"] == file_hash(PROJECT / "artifacts" / "model.pkl")
    entries = {e["family_id"]: e for e in manifest["end_to_end"]["validation"]}
    changes = []
    for row in result["runs"]:
        old = dict(row)
        row.update(
            regrade(
                entries[row["family_id"]],
                row,
                directory / f"{row['family_id']}-{row['mode']}.jsonl",
            )
        )
        changed = {k: [old[k], v] for k, v in row.items() if old[k] != v}
        if changed:
            changes.append(
                dict(family_id=row["family_id"], mode=row["mode"], changes=changed)
            )
        print(row["family_id"], row["mode"], row["run_success"], flush=True)
    result["summary"] = summarize(result["runs"])
    result["evaluation_sha256"] = file_hash(PROJECT / "aeromill" / "evaluation.py")
    result["regrade_changes"] = changes
    result["regrade_protocol"] = (
        "deterministic command replay, every recorded public tick checked to 1e-9; no detector"
    )
    write_json(directory / "results.json", result)
    with (directory / "runs.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(result["runs"][0]))
        writer.writeheader()
        writer.writerows(
            {
                k: json.dumps(v) if isinstance(v, (list, dict)) else v
                for k, v in r.items()
            }
            for r in result["runs"]
        )
    print(
        json.dumps(
            dict(changed_runs=len(changes), summary=result["summary"]), indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
