"""Offline simulation, dataset build, training, and a guarded evaluation boundary."""

import argparse
from pathlib import Path

from .engine import Engine
from .scenarios import NAMED_SCENARIOS


def main():
    parser = argparse.ArgumentParser(
        description="Synthetic simulation; not validated on real CNC"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--scenario", choices=tuple(NAMED_SCENARIOS), default="recover_a")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument(
        "--mode",
        choices=("no_adaptation", "baseline", "threshold_search", "ml_agent"),
        default="threshold_search",
    )
    run.add_argument("--model-dir", type=Path, default=Path("artifacts"))
    data = sub.add_parser("data").add_subparsers(dest="data_command", required=True)
    build = data.add_parser("build")
    build.add_argument("--workers", type=int, default=4)
    sub.add_parser("train")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument(
        "--split", choices=("validation", "final", "test", "reserve"), required=True
    )
    evaluate.add_argument("--mode", default="all")
    args = parser.parse_args()
    if args.command == "data":
        from .data import build

        build(workers=args.workers)
    elif args.command == "train":
        from .training import train

        train()
    elif args.command == "evaluate":
        if args.split == "final" and not Path("artifacts/FROZEN.json").is_file():
            parser.error(
                "final evaluation requires a freeze marker; none is created by Day 3"
            )
        parser.error(
            "evaluator deferred; Day 3 validation evidence: tests/evidence/day3_validation.py; withheld sets remain disabled"
        )
    else:
        e = Engine(
            NAMED_SCENARIOS[args.scenario],
            seed=args.seed,
            mode=args.mode,
            model_dir=args.model_dir,
        ).run()
        print(
            f"{e.agent.state} reason={e.agent.reason or 'endpoint'} t={e.time_s:.1f}s commands={e.controller.action_count} log={e.log_path}"
        )
