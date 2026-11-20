# Learned lifecycle goals in actual engine encounters

Steps 3 and 4 are now exercised: retrain on the response teacher's recordings,
then let the network choose goals while the real engine runs. This experiment
is **not installed in the viewer** and does not replace the teacher by default.

## Retraining

Used `lifecycle-response-v1`, retaining its original whole-seed train/validation/
test partitions. The architecture is unchanged: 99 observation features ->
128 ReLU -> 64 ReLU -> GRU(64) -> 64 ReLU -> nine goal logits. The memory-free
comparison uses the same encoder/actor without the GRU. Each has one fixed
training initialization, 30 epochs, training-only frequency weights, and
validation macro-F1 checkpoint selection.

| Recorded held-out decisions | No GRU | GRU |
|---|---:|---:|
| Teacher-goal agreement | 91.67% | 95.08% |
| Macro F1 | 0.791 | 0.860 |

These results are on a different corpus from the first experiment; the small
accuracy difference from the previous 94.3% is not a controlled improvement
claim. Validation remains small, including very few courtship decisions.

## What controls an actual turn

```text
Private senses + own sex/cooldown
    -> learned lifecycle encoder and GRU
    -> argmax of nine learned goal logits
    -> sensory target/motor executor
    -> existing learned pursuit motor / engine command
    -> actual engine movement, damage, food, mating, hatching and death
```

`LearnedLifecyclePolicy.choose()` never calls `FeedingPolicy.choose()`. There is
no teacher goal substitution, no safety goal override and no learned-goal mask.
Targets and locomotion remain procedural, as in the teacher. Missing targets
produce a wait command and are counted; targets beyond engine reach can still
cause rejected commands. The selected goal is retained in the log even when
execution fails.

The executor advances pursuit memory once per turn and keeps short target,
meal and attacker bookkeeping. The teacher's hunger/health priorities and
mate-seeking latch are not used to select the network's goals. The existing
engine still enforces feasibility and all lifecycle/resource rules.

Each spider owns its own lifecycle GRU state. Newborns share frozen model
weights but receive a new controller, fresh lifecycle memory and fresh pursuit
memory. Own cooldown/sex are supplied as proprioception at decision time, not
as another actor's hidden state. Other creatures retain their existing scripted
controllers. The old one-step predictor is not part of this goal-selection test.

## Eight new matched seeds

Preselected seeds: 151, 163, 179, 191, 211, 223, 239, 251. None appears in the
imitation corpus. Each controller gets 1,000 six-second turns per seed, the same
starting world and fixed pursuit weights. Changed decisions alter subsequent
encounters and random-event consumption, so individual fights are not identical.

| Outcome across eight runs | Response teacher | Learned selector |
|---|---:|---:|
| Populations alive at turn 1,000 | 1 | 0 |
| Egg sacs | 70 | 2 |
| Food consumed | 242.50 | 54.53 |
| Recorded spider decisions | 27,629 | 5,860 |

Teacher seed 211 maintains 37 spiders at the end (peak 56). Its learned pair
fails to reproduce and dies out. Conversely, learned seed 251 lays two sacs
and reaches six spiders, whereas its teacher pair never reproduces; this learned
population also dies out. All student spider policies, including offspring,
are confirmed to remain learned controllers.

Food and decision totals are affected by population/lifetime differences;
they are not per-actor efficiency measures. Eight seeds are a descriptive
comparison, not a statistical architecture ranking.

## Why the copying score did not transfer

The learned selector sometimes confuses approaching a target with interacting
with it. Repeated engine rejections leave it in situations the teacher almost
never demonstrates. Its GRU can then reinforce the wrong activity.

In diagnostic seed 97, the first goal disagreement occurs on turn 20: the
teacher chooses `approach_meal`, but the student chooses `wrap` and receives
`food_not_reachable`. The trajectories diverge well before the teacher's
successful courtship on turn 85. The student makes no courtship attempt in
this diagnostic run; this is not evidence that the network saw the identical
invitation and rejected it.

In prospective seed 211 the student makes 190 `food_not_reachable` attempts;
in seed 239 it makes 89, plus 18 unreachable courtship attempts. These failures
are not corrected by the teacher during student runs. Explicit refusal categories
such as `food_not_reachable` are in raw observations but are not distinct flags
in the current 99-feature encoder; that information is lost before the GRU.
The encoder does include food/partner distance and previous action, so this is
both a boundary-learning and recovery-data problem, not a lack of distance data.

The learned controller is **not ready to replace the teacher**. High agreement
on the teacher's trajectories is insufficient for independent operation.

## Next learning experiment

Keep this failed closed-loop benchmark as a checkpoint. Then:

1. Encode actual refusal categories and reach-relative target distances clearly,
   without silently changing the selected goal or forcing a feasibility mask.
2. Collect teacher corrections on states the student actually visits, especially
   failed wrapping/courtship and recovery. The engine should continue executing
   student decisions during collection; the teacher supplies training labels.
3. Retrain on original demonstrations plus these recovery states. Reserve new
   seeds for the next closed-loop comparison. Check event completion and survival,
   not only agreement on recorded turns.

This is still imitation improvement. Learning to outperform a flawed teacher
requires alternative engine rollouts and consequence/cost learning later.

## Validation and files

Tests verify streamed GRU inference equals full-sequence inference, fresh newborn
memory, absence of teacher calls/goal substitution, engine courtship execution,
and motor command parity when supplied the same goals. The full suite also
covers unchanged teacher, lifecycle, feeding and viewer replay behavior.

Outside Git under `/Users/jeff/Documents/ChatGPT/White Fang/experiments/`:

- `lifecycle-response-imitation-v1`: both retrained checkpoints and offline metrics.
- `lifecycle-closed-loop-v1`: prospective teacher/student decision logs, populations,
  metrics and population comparison plot.
- `lifecycle-closed-loop-diagnostic-v1`: known seeds 42 and 97, explicitly diagnostic
  because they overlap the original corpus. They are not added to prospective totals.

[Compact results](results/learned-lifecycle-v1.json).

```sh
python3 -m experiments.train_lifecycle_model "$ARTIFACT_ROOT/lifecycle-response-v1" --output "$ARTIFACT_ROOT/lifecycle-response-imitation-v1" --epochs 30
python3 -m experiments.evaluate_learned_lifecycle --output "$ARTIFACT_ROOT/lifecycle-closed-loop-v1" --pursuit "$PURSUIT_WEIGHTS" --model "$ARTIFACT_ROOT/lifecycle-response-imitation-v1/gru.pt" --turns 1000 --workers 3
python3 -m experiments.evaluate_learned_lifecycle --output "$ARTIFACT_ROOT/lifecycle-closed-loop-diagnostic-v1" --pursuit "$PURSUIT_WEIGHTS" --model "$ARTIFACT_ROOT/lifecycle-response-imitation-v1/gru.pt" --turns 1000 --workers 3 --seeds 42 97 --purpose diagnostic
MPLCONFIGDIR=/tmp/driftmoor-mpl python3 -m experiments.report_learned_lifecycle "$ARTIFACT_ROOT" --summary experiments/results/learned-lifecycle-v1.json
```
