# AeroMill-Adaptive AI

A local agent that detects simulated milling chatter, changes spindle speed and
feed, checks the result, and then either keeps cutting or stops safely.

**Synthetic simulation; not validated on real CNC.**

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Run

```bash
uv run python -m aeromill run --scenario recover_a --seed 42 --mode ml_agent
uv run python -m pytest -q
```

Modes: `no_adaptation`, `baseline`, `threshold_search`, `ml_agent`. Each run
writes `runs/<run_id>.jsonl` and prints the final state, reason, simulated time
and number of commands. `recover_a` with seed 42 finishes at 23.9 s with one
command.

## Rebuild the dataset and the model

```bash
uv run python -m aeromill data build
uv run python -m aeromill train
```

`data build` splits 300 episodes by family (180/60/60) and generates only the
train and validation parts. The shipped `artifacts/model.pkl` is the trained
RandomForest; `artifacts/model.json` holds its feature order, versions and
hashes. Only load pickle files you trust.

## What is inside

- `simulator.py`, `scenarios.py`: phenomenological milling model with resonance
  zones, hidden engagement changes, impacts and noise.
- `controller.py`: the only command entry; limits, acknowledgments, stop latch.
- `features.py`, `detector.py`: 23 causal window features and the chatter detectors.
- `agent.py`, `engine.py`: one state machine for every mode, with a planning
  record at each decision.
- `data.py`, `training.py`: dataset manifest and RF training.

All deadlines use 100 ms simulation ticks. The feed-only baseline stays inside
the resonance zone by construction, so the ML agent is compared with
`threshold_search`, which uses the same candidates. Detector scores are model
scores, not probabilities of failure.
