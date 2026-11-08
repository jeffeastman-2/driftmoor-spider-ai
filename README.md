# Driftmoor Spider Agent Experiments

Open this directory in VS Code. It contains the encounter environments,
controllers, rule teachers, neural networks, training, diagnostics, and tests.
Driftmoor supplies the game mechanics and authored world as an external
dependency; you do not need to open a second VS Code project.

## Code map

| Location | Purpose |
| --- | --- |
| `spider_sim/dual_encounter.py` | Two-actor sensory fixture, demonstrations and neural rollouts |
| `spider_sim/encounter_teacher.py` | Observation-only teachers, distinct retreat/flee behaviors |
| `spider_sim/encounter_evaluation.py` | Matched encounter scoring and behavior-specific metrics |
| `spider_sim/spider_experiment.py` | Scene adapter, observations, movement and combat execution |
| `spider_sim/spider_controller.py` | Earlier observation-driven rules and candidate scoring |
| `spider_sim/spider_learning.py` | Earlier scripted episodes and collection |
| `spider_sim/spider_diagnostics.py`, `spider_v2.py` | Counterfactual diagnostics and two-step experiments |
| `experiments/recurrent_agents.py` | Numeric encoder, GRU, output heads and checkpoint loading |
| `experiments/run_recurrent_agents.py` | Training and evaluation |
| `tests/`, `experiments/test_*.py` | Encounter and neural tests |

## Setup

Keep the experimental Driftmoor worktree at `../Driftmoor-spider-sim` on
`codex/spider-world-model`. It contains the game-side sensing, body geometry,
movement and combat mechanics. The normal `../Driftmoor` checkout stays separate.

Use one Python environment with both sets of requirements:

```bash
python3 -m pip install -r requirements.txt -r ../Driftmoor-spider-sim/requirements.txt
python3 -m pytest -q tests experiments
```

Test discovery and direct experiment imports use the sibling game worktree by
default. Set `DRIFTMOOR_ROOT` to use another compatible game checkout. Command
line runners retain their explicit `--driftmoor-root` option. A process uses
one game checkout; changing it after game modules load requires a new process.

VS Code settings include the sibling game's modules for reference resolution
and both test directories. Choose the Python interpreter with the dependencies
installed. The Run and Debug panel includes baseline and neural-training
launch configurations. Training outputs default to a temporary directory.

## Run

```bash
python3 experiments/run_cavern_baseline.py \
  --driftmoor-root ../Driftmoor-spider-sim \
  --mode encounters --seeds 5 --output /tmp/cavern-encounters.jsonl

python3 -m experiments.run_recurrent_agents \
  --driftmoor-root ../Driftmoor-spider-sim \
  --output /tmp/driftmoor-recurrent-agents
```

See [neural agent training](experiments/RECURRENT_AGENTS.md) and the
[earlier experiment instructions](experiments/README.md).
Small measured reports are tracked in `experiments/results/`; generated
datasets and trained checkpoints belong outside Git.

The networks currently imitate private-observation rule teachers and learn
memory and next-observation predictions. They are an experimental baseline,
not the production spider controller or a complete JEPA planner. No images,
LLMs, model downloads, or V-JEPA integration are required.

## Refactor boundary

The six experimental modules and five encounter test files formerly under
Driftmoor's `engine/` and `tests/` now live here. Use `spider_sim.*` imports.
Driftmoor retains generic sensing, body geometry and combat extensions and their
tests. Both repositories' experimental branches must be updated together.
Existing checkpoints and result formats are unchanged.

Refactor verification: 67 tests pass here and 43 generic body/senses tests pass
in Driftmoor. All 32 saved test demonstrations and all 16 trained-GRU rollout
totals match the pre-refactor artifacts exactly. Baseline, v1 training, neural
training, diagnostics, and v2 evaluation entry points were smoke-tested.

## Improved teacher experiment

See [teacher evaluation](experiments/TEACHER_EXPERIMENT.md) for the new reach/facing teacher, matched MLP/GRU comparisons, results, and review guide.

## Retreat and flee

The current teacher distinguishes maintaining focus while backing away from turning away to flee. See [behavior, training, and results](experiments/RETREAT_AND_FLEE.md). The network still has eight motor outputs. All current changes remain uncommitted pending review.

## Corrected turn-and-move timing

Retreat and flee now turn and move in one actor slot. See [current action semantics, results, and 5e considerations](experiments/TURN_AND_MOVE.md). Earlier turn-slot survival reports are historical.
