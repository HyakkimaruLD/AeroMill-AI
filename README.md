# AeroMill-Adaptive AI

A local agent detects simulated milling chatter, changes spindle speed and feed,
verifies the result, and retries or stops. **Synthetic simulation; not validated
on real CNC.** All control deadlines use simulation time in 100 ms ticks.

## Run locally

Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run streamlit run app.py
uv run python -m aeromill run --scenario recover_a --seed 42 --mode ml_agent
uv run python -m pytest -q
```

The shipped `artifacts/model.pkl` and `model.json` contain the frozen
RF and its hash/version contract. Only load trusted pickle artifacts.

Modes are `no_adaptation`, `baseline`, `threshold_search`, and `ml_agent`.
The CLI writes `runs/<run_id>.jsonl`. The feed-only baseline remains resonant by
construction; compare ML with threshold_search to isolate the detector's effect.
RF scores are model scores, not probabilities of an accident.

## Session and viewer APIs

`RunSession(scenario, seed, mode)` owns one engine. `advance(ticks)` advances it;
`snapshot()` returns detached observations; `request_stop()` latches Stop on
the next tick. `export()` returns CSV and JSON bytes. Evaluations appear only
at termination, and online recovery claims remain provisional until then.

```python
from aeromill.parts import Part, random_part, validate_part
from aeromill.session import RunSession
from aeromill.viewer import catalog, trajectory

part = random_part()  # Optional reproducible demo seed: 500000–599999.
session = RunSession.from_part(part, "ml_agent")
live = session.advance(10)
# live has no hidden zones, parameters, truth, or reconstructible seed.
finished = session.advance(650)
truth = session.reveal()  # Raises before HOLD or COMPLETED.
heatmap = session.stability_map(1200., [2880., 3200., 3520., 3840.], [80., 200.])
path = trajectory(finished)
```

Custom `Part` values use `Scenario`, `Region`, and `EngagementSegment` records.
Construction and `validate_part(part)` refuse nonfinite or unsupported values
with a field-specific reason. Bounds come from the frozen dataset generator;
gain is fixed at 1, widths are 100–180 or the broad-band preset's 2000.
Random demos use the fixed known tool profile so public profile fields cannot
identify their hidden seed. Unknown runs use unchanged initial RPM/feed and
frozen detectors/thresholds.
They are demonstrations, not acceptance metrics.

`catalog()` describes all presets, including NaN/missing/out-of-order telemetry,
command rejection, missing acknowledgment, manual Stop and startup chatter.
Each row's `run` dictionary can be passed to `RunSession(**row["run"])`.
`stability_map(part_or_scenario, feed, rpm_grid, x_grid)` in `aeromill.viewer`
returns gain × resonance and static A_target (zero at stopped feed), with shape
`(len(rpm_grid), len(x_grid))`. It is viewer-only. Use the session method for
Unknown parts so the terminal gate is enforced. Maps are synthetic simulator
explanations, not physical stability-lobe predictions.

## Evidence and limits

The tests include command safety, causal features, detector behavior, independent
truth grading, fault handling, Unknown-part isolation, and viewer formulas.
Reference scripts used by some tests live in `tests/evidence/`.

The latest four-mode validation is in
`artifacts/prefreeze-validation/acceptance.md`. ML verification uses a residual
RMS ratio limit of 2.2, selected on validation. Two false retries remain in the
40-part demo diagnostic; those parts were not used for tuning.

Cold validation results are in `artifacts/day4-validation/`; warm results use
its `warm/` directory and retain a separate priming run. Memory is cold by default;
reused corrections must pass verification again. The evaluator never treats
RF scores as truth. The frozen source, model and manifest are identified by
`artifacts/FROZEN.json`. Final evidence is retained in `artifacts/final/`:
`SUMMARY.md`, `acceptance.md`, `failed-runs.md`, per-run JSONL logs,
`runs.csv`, `results.json`, test-window arrays, commands and execution receipts.
Reserve execution remains disabled.

The single evaluation uses these commands from the project root, with an
external writable cache supplied through `UV_CACHE_DIR`:

```bash
uv run --offline --no-sync python -m aeromill evaluate --split test --mode all --workers 1 --output artifacts/final
uv run --offline --no-sync python -m aeromill evaluate --split final --mode all --workers 1 --output artifacts/final
```

These are provenance commands, not instructions to repeat the experiment.
The committed start receipts refuse another attempt. Both entry points verify
frozen file hashes; a changed, added or missing source file blocks execution.
A crash ends the attempt, and its receipt and available raw records remain.
Do not rebuild or retrain the frozen artifacts to reinterpret these results.

The offline dataset/training commands remain available for reproducing training:
`python -m aeromill data build` and `python -m aeromill train`. Smoke seeds
900000–900001 and demo seeds 500000–599999 are separate from all manifest splits.
The shipped model was not retrained for Unknown parts or the faster RF path.
