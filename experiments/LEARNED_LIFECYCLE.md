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

## Recovery learning: revised experiment

This follow-up preserves existing 99-input checkpoints and adds a versioned
145-input encoder. The original 99 values remain the prefix. New values encode
20 engine-result statuses, missing/other status, whether the last attack defeated
its target, four signed distances from the six-foot interaction boundary, ten
measurements of an identified and currently sensed attacker, and the actor's own
previous selected intent (nine values). The two MLPs and 64-value GRU retain their
previous sizes; only the first input matrix grows.

Previous intent means the actor's actual prior choice, even if the engine rejected
its command. In training it is reconstructed causally from the previous decision
in the same actor's episode; corrective records use `executed_goal`, not the
previous teacher recommendation. At runtime it comes from the actor itself.
Current teacher labels, proposed advisor commands, future results, global IDs and
opponent state are not encoder inputs. Legacy observations and checkpoints keep
using the 99-value path.

Fresh demonstrations use the current response teacher, damage interruption in
mating phases, one-HP recovery after feeding or below 10% hunger, and the 30%
mating cutoff. The original whole-seed train/validation/test split is retained.
The first correction corpus contains only the eight training seeds, run for
500 turns with the old trained network as the actual controller. An independent
advisor sees the same private senses and this actor's actual prior activity and
motor target bookkeeping. Its hunger-derived search latch follows that private
history. The advisor does not inspect GRU state, opponent internals or future
outcomes, and its commands never reach the engine.

The correction corpus contains 10,393 decisions and 3,364 teacher disagreements,
including 611 `wrap -> approach_meal` corrections and 166 `court -> approach_mate`
corrections. The student actually received 790 `food_not_reachable` and 238
`mate_not_reachable` results. These are visited failure/recovery states rather
than synthetic claims about hypothetical outcomes. Not every disagreement is a
proven error; the teacher remains an authored and fallible reference.

Two 145-input models use the same initialization, 30-epoch budget, validation
macro-F1 selection, and GRU architecture. One uses current teacher demonstrations
alone; the other adds recovery sequences with fourfold sampling. Repeated
sequences remain the same correlated examples and are not counted as independent
experience. Test episodes are deferred until checkpoints are selected.

The prospective comparison reserves seeds 263, 277, 293, 307, 331, 347, 359 and
373 for 1,000-turn runs. Four conditions use identical current engine rules:
teacher, old 99-input prior, new encoder trained on demonstrations, and the same
encoder trained with corrections. This old-prior control matters because healing
and the wider mating window can themselves improve an unchanged network.
Every learned condition uses its own unmasked argmax goals; no advisor or teacher
goal fallback is present during evaluation. Newborns retain shared frozen weights
and independent fresh memories. No live deployment is part of this experiment.

### Completed fresh-seed results

All 32 episodes completed: eight unused seeds, four frozen controllers, 1,000
turns each, with identical healing, mating and defensive-interruption rules.
The old network is included under these same current rules, so the comparison
does not attribute rule changes to learning.

| Measure | Teacher | Old network | New inputs + demonstrations | With corrections |
|---|---:|---:|---:|---:|
| Runs ending with more than two spiders | 4/8 | 1/8 | 3/8 | 3/8 |
| Total living spiders at turn 1,000 | 146 | 19 | 124 | 108 |
| Mean population per seed over final 200 turns | 23.21 | 2.96 | 15.36 | 17.38 |
| Egg sacs | 350 | 51 | 190 | 225 |
| Feeding/mating/web refusals per 1,000 creature decisions | 0.14 | 37.51 | 18.50 | 4.80 |
| Longest consecutive same-target refusal streak | 1 | 114 | 79 | 29 |
| Waits per 1,000 creature decisions | 1.98 | 15.93 | 10.78 | 6.48 |
| Longest hungry wait with visible food and no new damage | 12 | 38 | 68 | 40 |

The new models maintain groups in seeds 277, 307 and 347, where the old network
fails. The old network maintains a group in seed 293, where the new models fail
to establish one; the corrected model retains only a lone survivor there. Seeds
263, 331 and 373 fail for all four controllers. Seed 359 leaves a single teacher
survivor and no learned survivors. These regressions are retained in the results.

Adding corrections reduces lifecycle refusals by approximately 74% relative to
the demonstrations-only condition and reduces waiting, rather than merely
replacing rejected commands with idleness overall. The mean of the eight per-seed
refusal rates also falls (125.66 to 5.03 per 1,000 decisions), so the aggregate
change is not solely dilution by more decisions in long-lived populations.
However, both new conditions establish the same three groups, and the corrected
model does not match teacher robustness. Final population alone is phase-sensitive:
the corrected model has fewer spiders on turn 1,000 but a higher final-200-turn
mean and more egg sacs than the demonstrations-only model. None of these measures
establishes long-term equilibrium or a statistically robust ranking from eight
seeds and one initialization.

Frozen copying accuracy on current held-out teacher episodes is 94.05% for the
old network, 98.47% for new inputs plus demonstrations, and 97.71% with corrections.
Corresponding macro F1 values are 0.826, 0.943 and 0.916. Thus correction training
improves the measured rejection/recovery behavior despite slightly worse copying
accuracy. No fresh-seed outcome or copying-test result was used to change the
frozen models. Prior-versus-new comparisons change both encoder and teacher
corpus; demonstrations-versus-corrections also changes examples, sampling weights
and optimizer steps per epoch. This is exploratory evidence, not a step-matched
causal ablation of every component.

### Remaining target-context mismatch

A post-hoc audit identified a concrete interface gap. In fresh seed 263 the
new-inputs model makes 72 failed wraps while a meal is approximately five feet
away: its motor controller retains a different meal 29.53 feet away and directs
the wrap at that one. The encoder describes the nearest two meals but does not
explicitly bind either slot to the motor's retained target. The corrected model
reduces that case to three isolated attempts, but the missing binding remains.
Across the fresh runs, the same near-food/far-retained-target conflict accounts
for 508 refusals in the demonstrations model and 130 in the corrected model;
130 and 40 respectively target a meal outside the two encoded meal slots.
These are real distance errors, not floating-point cases infinitesimally above
six feet (none were found among these food refusals).

The next interface change should expose the actual currently selected motor
target's private measurements to the actor, and have the executor use that same
selection. It should not override the actor's goal with teacher rules or hide
failures behind a feasibility mask. Own defensive-memory age is another candidate
for making recovery states explicit. These are follow-up designs, not changes
made during this frozen comparison. Further training should include trajectories
from the new controller, not just the old prior's failures, and reserve another
fresh seed set for assessment.

### Verification and artifacts

All 281 tests pass. Coverage includes versioned checkpoint loading, streamed GRU
parity, causal previous-intent construction, advisor noninterference, an actual
engine test proving student commands are executed instead of advisor commands,
and safe reuse of completed evaluations. Each learned run confirms that every
spider, including offspring, retains a learned controller and fresh individual
memory. There is no teacher goal fallback, no online weight update, and no new
cost/world-model planner in these runs.

Detailed recording initially used gzip level 9; the evaluation was resumed with
level 1 and then six parallel workers. Completed cases were reused after manifest
checks; incomplete folders were retained as `.partial-*` and replayed. A control
replay verified identical decoded decision records and simulation metrics across
the compression change. Worker count and compression change recording/runtime,
not decisions or engine rules. No live viewer was restarted or modified.

[Compact metrics and per-seed results](results/recovery-learning-v2.json).
The comparison plot and raw observations remain outside Git in
`/Users/jeff/Documents/ChatGPT/White Fang/experiments/recovery-closed-loop-v2`.
Demonstrations are in `recovery-teacher-corpus-v2` (224,959 decisions), corrections
in `student-recovery-corpus-v2`, and model weights in `recovery-encoder-demo-v2`
and `recovery-corrected-v2`, under the same artifact root.

Reproduce the fitting/evaluation (use new output folders, or `--resume` only for
an existing matching evaluation):

```sh
python3 -m experiments.collect_lifecycle_experience --output "$ARTIFACT_ROOT/recovery-teacher-corpus-v2" --spider-model "$PURSUIT_WEIGHTS" --turns 1000 --workers 3 --current-rules
python3 -m experiments.collect_recovery_experience --output "$ARTIFACT_ROOT/student-recovery-corpus-v2" --pursuit "$PURSUIT_WEIGHTS" --model "$ARTIFACT_ROOT/lifecycle-response-imitation-v1/gru.pt" --turns 500 --workers 2
python3 -m experiments.train_lifecycle_model "$ARTIFACT_ROOT/recovery-teacher-corpus-v2" --output "$ARTIFACT_ROOT/recovery-encoder-demo-v2" --feature-version v2 --gru-only --epochs 30 --defer-test
python3 -m experiments.train_lifecycle_model "$ARTIFACT_ROOT/recovery-teacher-corpus-v2" --output "$ARTIFACT_ROOT/recovery-corrected-v2" --feature-version v2 --gru-only --epochs 30 --defer-test --recovery-corpus "$ARTIFACT_ROOT/student-recovery-corpus-v2" --recovery-repeat 4
python3 -m experiments.evaluate_learned_lifecycle --output "$ARTIFACT_ROOT/recovery-closed-loop-v2" --pursuit "$PURSUIT_WEIGHTS" --models "prior=$ARTIFACT_ROOT/lifecycle-response-imitation-v1/gru.pt" "demos=$ARTIFACT_ROOT/recovery-encoder-demo-v2/gru.pt" "recovery=$ARTIFACT_ROOT/recovery-corrected-v2/gru.pt" --current-rules --turns 1000 --workers 6 --seeds 263 277 293 307 331 347 359 373
MPLCONFIGDIR=/tmp/driftmoor-mpl python3 -m experiments.report_recovery_learning "$ARTIFACT_ROOT" --summary experiments/results/recovery-learning-v2.json
```

## Selected-meal binding v3

The meal target is now selected **before** lifecycle inference by one pure helper,
`select_meal(observation, remembered_track)`. It retains the preferred meal only
when that private track is currently perceived, otherwise selecting the nearest
currently perceived meal or no meal. The actor receives that selected observation,
and `execute()` receives the same snapshot. It does not look up another meal
between inference and execution. Explicit absence remains absence. This preserves
the existing meal preference; it does not silently switch to nearer food or
supply an unseen remembered position.

A new, versioned 152-input encoder appends seven values to the existing 145:
selected-meal presence, range, relative bearing sine/cosine, wrapping state,
remaining food and distance from the six-foot interaction boundary. Private track
IDs are used to bind the target but are not numerical model inputs. The first
MLP input matrix grows; the 128/64 encoder, GRU64 and 64/9 goal head otherwise
keep their shapes. Versions 99, 145 and 152 remain loadable.

New recordings capture the selected-meal observation before a policy acts. For
older complete per-actor recordings, the loader reconstructs meal memory causally
from past actual activities and observations. Corrective recordings use the
student's actual activity, not the advisor's recommendation. The current teacher
answer or command is not used to choose the input target; recorded wrap/feed
commands are checked afterward as an integrity assertion. A disagreement raises
an error instead of replacing the input with the answer. Explicit recorded
snapshots are also checked against the reconstruction.

The target-bound model uses the same teacher demonstrations, correction corpus,
30-epoch recipe, seed 2026, validation macro-F1 selection and fourfold correction
sampling as v2. Changing input shape also changes the initialization stream, so
one run is not a fully controlled multi-initialization architecture study.
Checkpoints are frozen before assessment. No feasibility mask, automatic approach
rule or teacher goal fallback is added: a model that chooses `wrap` while far away
still issues `wrap` and can be rejected by the engine.

Regression tests include a retained meal outside the nearest two slots, opposite
movement directions for near and retained meals, changing memory during inference,
a frozen missing selection, private/causal encoding, and all three checkpoint
versions. A real legacy-v2 replay produced identical commands, probabilities,
results and final metrics after the refactor. The known seed-263 failure frame
now makes the trained v3 model choose `approach_meal` for the retained carcass at
29.53 ft, despite another meal at 5 ft. This probe uses a recorded old trajectory;
it is not an executed counterfactual or fresh-seed outcome.

The prospective assessment reserves six unused seeds: 383, 397, 419, 433, 449 and
461. It compares the teacher, the previous corrected 145-input model, and the
152-input target-bound model for 1,000 turns each under the same current rules.
Only meal binding is changed here; mate/attacker target binding and learned target
choice remain separate concerns. The running viewer and existing defaults are
unchanged, and no online weight updates occur during evaluation.

### Target-binding v3 assessment

All 286 tests pass. The known-failure probe also holds the pre-step GRU state
and all other inputs fixed while swapping only the seven selected-meal fields:
the trained model chooses `approach_meal` for the 29.53-foot retained carcass,
and `wrap` for the five-foot meal. These are input probes, not executed engine
counterfactuals or calibrated success probabilities.

The full eighteen-run, six-seed comparison gives:

| Measure | Teacher | Previous corrected v2 | Target-bound v3 |
|---|---:|---:|---:|
| Runs ending above two living spiders | 3/6 | 4/6 | 3/6 |
| Total living spiders at turn 1,000 | 101 | 172 | 102 |
| Mean population per seed, final 200 turns | 17.25 | 24.82 | 21.09 |
| Egg sacs | 253 | 314 | 210 |
| Food refusals | 0 | 152 | 72 |
| Food refusals per 1,000 creature decisions | 0 | 1.26 | 0.76 |
| Mean of per-seed food-refusal rates | 0 | 0.89 | 1.09 |
| Rejected far-retained-food attempts while nearer food is reachable | 0 | 128 | 0 |
| Waits per 1,000 creature decisions | 3.05 | 7.18 | 9.63 |
| Longest repeated same-target refusal streak | 1 | 34 | 13 |

The specific near-food/far-retained-target failure pattern disappears in this
batch. All 11,089 v3 wrapping/feeding commands agree with their pre-inference
selected-meal snapshots. Previous and teacher logs also include snapshots for
audit; only v3 encodes them, so snapshot/command equality alone is not evidence
that an old model saw its selected target.

The replacement weights are **not a general performance improvement**. Aggregate
food-refusal rate is lower, but the mean per-seed rate rises, waiting increases,
and fewer runs establish lasting groups. V3 maintains populations in seeds 383,
433 and 461; it gains seed 461 compared with v2 but loses seeds 419 and 449
(the latter retains only one survivor). V2 maintains groups in 383, 419, 433 and
449. With six seeds and one initialization these are exploratory outcomes, not
proof of a general ranking. Both models achieve essentially the same held-out
copying accuracy (v3 97.69%, previous v2 97.71%; v3 macro F1 0.919).

The interface fix and versioned data path are worth preserving. Keep the new
checkpoint experimental rather than replacing the previous weights. Correct
selected-target information removes one ambiguity; it does not guarantee a
correct goal, improve the authored meal preference, or solve remaining goal,
mating and defensive-memory decisions. Future fitting can preserve prior learned
weights while adapting the added inputs and should use another fresh seed set;
no such tuning was performed after viewing these assessment results.

[Compact target-binding results](results/target-binding-v3.json).
Raw trials and the comparison plot are under
`/Users/jeff/Documents/ChatGPT/White Fang/experiments/target-binding-closed-loop-v3`;
weights are in `target-bound-model-v3`. Existing checkpoints and the live viewer
remain unchanged. Raw datasets and weights are outside Git.

```sh
python3 -m experiments.train_lifecycle_model "$ARTIFACT_ROOT/recovery-teacher-corpus-v2" --output "$ARTIFACT_ROOT/target-bound-model-v3" --feature-version v3 --epochs 30 --gru-only --defer-test --recovery-corpus "$ARTIFACT_ROOT/student-recovery-corpus-v2" --recovery-repeat 4
python3 -m experiments.evaluate_learned_lifecycle --output "$ARTIFACT_ROOT/target-binding-closed-loop-v3" --pursuit "$PURSUIT_WEIGHTS" --models "previous=$ARTIFACT_ROOT/recovery-corrected-v2/gru.pt" "bound=$ARTIFACT_ROOT/target-bound-model-v3/gru.pt" --current-rules --turns 1000 --workers 6 --seeds 383 397 419 433 449 461
MPLCONFIGDIR=/tmp/driftmoor-mpl python3 -m experiments.report_target_binding "$ARTIFACT_ROOT" --summary experiments/results/target-binding-v3.json
```
