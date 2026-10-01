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
uv run streamlit run app.py
uv run python -m aeromill run --scenario recover_a --seed 42 --mode ml_agent
uv run python -m aeromill evaluate --split validation --mode all
uv run python -m pytest -q
```

The dashboard shows the agent loop, live vibration, spectrum, detector score,
the agent's plan and decision log, and the evaluator's verdict when a run ends.

Modes: `no_adaptation`, `baseline`, `threshold_search`, `ml_agent`. The feed-only
baseline stays inside the resonance zone by construction, so the ML agent is
compared with `threshold_search`, which uses the same candidates.

## Evaluation and memory

The evaluator grades every run from the simulator's hidden state and never uses
the detector score as truth. Validation results are in `artifacts/day4-validation/`.
Memory is cold by default, and a remembered correction has to pass verification
again before it counts. Test, final and reserve runs stay disabled until the
model is frozen.

## Rebuild the dataset and the model

```bash
uv run python -m aeromill data build
uv run python -m aeromill train
```

The shipped `artifacts/model.pkl` is the trained RandomForest; `model.json` holds
its feature order, versions and hashes. Only load pickle files you trust.
