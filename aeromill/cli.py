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
    evaluate.add_argument("--warm-memory", action="store_true")
    evaluate.add_argument("--output")
    evaluate.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "data":
        from .data import build

        build(workers=args.workers)
    elif args.command == "train":
        from .training import train

        train()
    elif args.command == "evaluate":
        if args.output is None:
            args.output = (
                "artifacts/final"
                if args.split in ("final", "test")
                else "artifacts/day4-validation"
            )
        from .evaluation import evaluate_validation

        try:
            if args.split in ("final", "test"):
                from .freeze import verify_freeze

                verify_freeze()
            if args.split == "test":
                from .training import evaluate_test

                evaluate_test(output=args.output)
                return
            evaluate_validation(
                split=args.split,
                mode=args.mode,
                warm_memory=args.warm_memory,
                output=args.output,
                workers=args.workers,
            )
        except ValueError as exc:
            parser.error(str(exc))
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
