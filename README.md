# Driftmoor Spider Agent Experiments

Research into perception, prediction and action selection for Driftmoor's
giant spider, using structured sensory observations and the game's actual
mechanics. No video generation or visual foundation model is required.

## Repositories

- `~/Python/Driftmoor`: normal game checkout on main.
- `~/Python/Driftmoor-spider-sim`: isolated game worktree on
  `codex/spider-world-model`, containing sensing, body geometry and encounter tools.
- This repository: training, evaluation, diagnostics and result summaries.

## Setup

Use a Python environment with this project's requirements and the experimental
Driftmoor checkout's requirements installed:

```bash
python3 -m pip install -r requirements.txt -r ../Driftmoor-spider-sim/requirements.txt
```

## Run

```bash
python3 experiments/run_cavern_baseline.py \
  --driftmoor-root ../Driftmoor-spider-sim \
  --mode encounters --seeds 5 --output /tmp/cavern-encounters.jsonl

python3 -m pytest -q experiments/test_world_model.py experiments/test_world_model_v2.py
```

See [experiment instructions](experiments/README.md) for training, validation,
fresh-test evaluation and the limits of each experiment.
Small measured results are tracked in `experiments/results/`.
Generated datasets and trained checkpoints belong outside Git.

## Current status

V2 learns immediate and two-step attack opportunities from filtered sensory
history and candidate actions. Its two-step planner improves on v1 but still
trails the chamber-gated experimental rule baseline. This is not yet the
production Driftmoor spider controller.

The old video integration, downloaded visual-model weights, notebook and
standalone mock simulator have been removed. Their tracked source remains
recoverable in Git history. The directory name is retained for existing paths.
