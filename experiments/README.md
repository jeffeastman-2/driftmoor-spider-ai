# Echoing Caverns spider experiment

Driftmoor runs from the isolated checkout on `codex/spider-world-model`.
The original main checkout and the old toy simulation are not used for changes.

## Run

From ~/Python/driftmoor-spider-ai, with Driftmoor dependencies available:

```bash
python3 experiments/run_cavern_baseline.py \
  --driftmoor-root ../Driftmoor-spider-sim \
  --mode encounters --seeds 5 --output /tmp/cavern-encounters.jsonl
```

Modes:
- baseline: seven prescribed placements using existing session perception/activation.
- resolved: eight resolved movement/wait transitions.
- encounters: seven episodes per seed; five seeds produce 260 transitions.

The launcher isolates Driftmoor imports in a subprocess. No LLM requests or
model downloads occur. Output is JSONL.

## Mechanics

The actual scene loader merges PLACE residents with scene players.
Movement uses existing speed clamping, home restriction, boundary and
impassable checks. Each sequential action slot is six seconds with a combat
movement budget; this is not the full narrated session or initiative loop.
The secret passage is assumed open and other party members stay at their
initial positions. Geometry is still 2.5D.

Stored z is floor-relative. Scene-local elevation for distance calculations
is floor_z_at(x,y) + z. Structured sensory ranges use 3D separation with
existing horizontal occlusion and nonvisual range rules. Contacts include
tracking ID, modality, rounded range, horizontal bearing and vertical angle.
Unobserved coordinates/health are not supplied to the observer.

Body geometry has width, depth and height. Local +x is forward/depth, +y is
left/width, +z is up. The anchor is the center of the local support face
(normally between the feet). Rotation is Rz(heading) Ry(pitch) Rx(roll);
0 heading points along +x, 90 along +y. Angles are right-handed degrees.
Serialized orientation_roll_pitch_heading_deg is ordered roll, pitch, heading.
Interpret the pose intrinsically: establish heading about scene vertical,
pitch about the headed lateral axis, then roll about the pitched forward
axis. Positive pitch points the nose downward with these right-handed axes.
Angles describe absolute pose, not incremental world-axis rotations.
Euler heading/roll become coupled at pitch +/-90; a future continuous
orientation controller may use quaternions internally.

Gil's IDENTITY.md supplies provisional dimensions: width 2 ft, depth 1 ft,
height 5.5 ft. These are editable experiment values, not inferred racial
measurements. The spider uses an explicitly labeled 4 x 4 x 2 ft proxy.
Profiles and provenance are recorded. Ceiling pose starts at roll=180.

Bite and spinneret have distinct local origins and opposite directions.
They transform with body pose. A bite aims at the target oriented box's
closest point and checks a 60-degree half-angle cone and weapon reach.
This conservative aim-point rule is not an exact cone/box intersection solver.
Spinneret geometry is defined; web production/damage is not implemented.

Stationary attacks use the real CombatResolver rolls and HP damage, with
auto-approach disabled. Range uses fang-to-target-volume separation, not
anchor-to-anchor distance. The existing 0.5-ft melee tolerance is retained.
Per-experiment RNG is temporarily installed and restored after combat.
Defeated targets leave the active set. An orient action consumes a slot
without translating the anchor.

A target may be reachable from above but outside the current bite arc.
Seven scripted scenarios include retreat before entry, waiting, attacking,
offsetting the target, a grounded control, turning away, and aiming downward.
The grounded control is an initialization fixture, not an executed descent.

## Limitations and training boundary

No learned policy/predictor is implemented. No drop/climb dynamics, attachment
validation, body collisions, full 3D occlusion, poison saves, web restraint,
spells, ammunition consumption or neurochemical effects are applied.
Orientation can be set without simulating how a supported body achieves it.
Planar movement budget is retained on slopes; rounding may add about 0.071 ft.

Legacy full-session callers still use point/planar distances unless they
explicitly supply height/body-contact data. Scene activation and spider
behavior prompts remain unchanged.

Records separate observer_before/after, action, outcome, baseline diagnostics,
and privileged ground_truth. Combat geometry/debug fields include exact aim
points and must not be given to a policy as sensory input. Player commands
are exogenous interventions, not the spider's actions. Do not train on player
destinations, episode labels, seeds or privileged diagnostics. Split by whole
encounter; repeated dice seeds alone do not provide new sensory trajectories.
This small fixture verifies integration, not learning effectiveness.

## Tests

From ~/Python/Driftmoor-spider-sim:

```bash
python3 -m pytest -q tests/test_body_geometry.py tests/test_spider_experiment.py \
  tests/test_activation.py tests/test_senses.py tests/test_perception_items.py \
  tests/test_geometry_loader.py tests/test_combat_resolver.py
```

Tests cover geometry transforms, body-surface reach, direction, elevations,
actual combat damage, visibility, activation, movement/collision, stable
records, deterministic encounters and RNG isolation.

## Observation-driven control and first learned predictor

Run the complete experiment from the spider project:

```bash
python3 experiments/run_world_model.py \
  --driftmoor-root ../Driftmoor-spider-sim \
  --output-dir /tmp/driftmoor-world-model-v1
```

Defaults: 160 training episodes, 30 validation episodes, 40 test episodes;
14 decision slots per episode; 140 training epochs. Episode seeds are split
0+, 10000+, 20000+, respectively. Whole encounters stay in one split.
The validation loss selects the checkpoint; test results do not select or
tune the controller. The first result is preserved in
results/world-model-v1.json.

Files in output-dir: train/validation/test JSONL, predictor.pt, report.json.
Keep generated data and checkpoints outside Git. Training uses installed CPU
PyTorch, one thread and seed 7; there are no model downloads or API charges.

The controller receives current/previous contacts and its own pose, position,
body dimensions, HP, and bite reach. Candidate destinations are computed from
coarse sensed direction/range and own position, not exact player coordinates.
Its geometric target proxy is an explicit humanoid estimate. Own body and
heading are proprioceptive inputs. No activation state, episode family, target
HP, resolver geometry, player commands or ground-truth positions enter the
numeric predictor. Vision is still omnidirectional; no new camera simulation
is claimed.

Actions: wait, face a contact, move up to eight feet toward it, or bite.
The engine rejects invalid attacks regardless of the controller's prediction.
A successfully executed experimental attack reveals the spider.
The rule controller uses estimated surface reach and facing.
The zone comparator runs that SAME rule controller only after chamber entry;
it is not a recreation of the production creature's LLM choices.
The learned controller scores candidate actions using predicted next contact
range/presence and probability that a bite is admitted, with explicit effort
costs. This is one-step scoring, not multi-step planning.

The neural model is a small two-hidden-layer MLP. Inputs concatenate two
17-value observations and a 10-value action encoding. Outputs predict the
next 17-value observation plus an attack-admission logit. Training uses
observation MSE and binary cross entropy. The next observation includes the
scripted player's subsequent response, whose command is never an input.
This is a structured-state dynamics experiment, NOT JEPA, V-JEPA2,
a learned visual encoder, or a full LeCun-style autonomous architecture.

First run: 2,240 training, 420 validation, 560 test transitions. Prediction
metrics on held-out recorded transitions:
- Visible range MAE: neural 9.54 ft, persistence 11.43 ft, constant 15.59 ft.
- Admission precision 0.833, recall 0.938, with only 16 positive test labels.

Closed-loop comparison on the same 40 held-out seeds:
- Zone-gated rules: 38 bite commands, 34 admitted, 22 hits, 153 damage.
- Observation rules: 24 bite commands, 19 admitted, 11 hits, 87 damage.
- Learned scoring: 34 bite commands, 4 admitted, 2 hits, 21 damage.

This is a NEGATIVE control result: offline prediction improves, but the
learned action chooser is not an improvement. It made no repositioning moves
and issued many rejected bites. Distribution shift from exploratory training
and the short-sighted scoring objective are hypotheses to investigate, not
established explanations. Do not replace the production spider with it.

The opponent follows seeded approach, retreat, pause, sidestep and feint
scripts and does not attack the spider. There are no distractions from
additional actors, no injury tradeoff, and no claim of generalization to new
maps or behavior families. Dice seeds alone do not count as scene diversity.

Additional tests:
```bash
# In Driftmoor-spider-sim
python3 -m pytest -q tests/test_spider_controller.py

# In driftmoor-spider-ai
python3 -m pytest -q experiments/test_world_model.py
```

Next research choices: inspect validation-only closed-loop failures, collect
better-covered action sequences, and test longer-horizon planning on a NEW
held-out evaluation split. Preserve this first result; do not tune against it
and continue calling it an untouched test.

## Why v1 misperforms: validation-only diagnostics

The v1 network, scoring function and test results were frozen. Diagnostics
use validation seeds 10000–10029 only (420 decision states). Every candidate
action is resolved on a deep copy of the engine state and its RNG; selected
fork outcomes are checked against the live episode. Exact future outcomes
are privileged diagnostic information, never controller inputs.

```bash
python3 experiments/diagnose_world_model.py \
  --driftmoor-root ../Driftmoor-spider-sim \
  --run-dir /tmp/driftmoor-world-model-v1 \
  --output-dir /tmp/driftmoor-world-model-v1-diagnostics
```

Full candidate traces go to diagnostics.json; aggregate results to summary.json.
The tracked aggregate is results/world-model-v1-diagnostics.json.

Confirmed findings:
- Chosen actions: 381 waits, 29 bites, 10 turns, no moves.
- Only 5 of those bites were admitted. Nineteen failed the bite arc; five
  failed reach. Mean predicted admission was 66.5%, actual admission 17.2%.
- One rejected bite (seed 10025, turn 10) had predicted admission 96.6%.
  Twenty-one chosen bites had probability >=0.5, so the lower decision
  threshold alone does not explain the failure.
- Admission Brier error was 0.011 on recorded validation attack actions but
  0.390 on attacks selected by the controller. Selection exposes errors
  that an average recorded-data score conceals.
- Replacing predictions with exact engine outcomes under the SAME one-step
  objective still chooses wait in 415 states and bite in five. Thus perfect
  immediate predictions alone cannot fix this objective on these states.
- A direct second-step probe found 38 states where turn-then-bite is admitted
  but wait-then-bite is not. The one-step score penalizes the enabling turn
  without valuing the subsequent bite. These are states, not 38 independent
  episodes. The probe follows the same scripted opponent response.
- Training had 30 admitted attacks among 305 bite commands, across 2,240
  transitions. Same-action nearest-training-input distance averaged 1.32
  for chosen closed-loop bites versus 0.40 for recorded validation bites.
  These distances use the fixed normalized feature coordinates; they are
  coverage indicators, not a causal proof or a calibrated uncertainty measure.

Conclusions: both action-conditioned prediction errors and a myopic
objective contribute. The controller exploits optimistic predictions on
poorly covered states, while its objective fails to reward preparation.
The same-state rule comparator would turn in 255 states, illustrating the
different incentive, not proving all those turns are optimal.

Recommended next experiment: add targeted training examples spanning close
targets with varied roll/pitch/heading (valid bites and matched hard
negatives), and evaluate two-step planning that credits a prepared bite.
Choose changes using validation only and reserve NEW test episodes for v2.
Do not merely raise the probability threshold, add more epochs, or replace
the current model with a larger network without measuring these failures.

## V2: covered training and two-step preparation

V1 code, checkpoint and results remain frozen. V2 adds:
- Every candidate action is labeled at sampled training states, not only
  the action chosen by the behavior policy.
- Matched close-target cases vary roll, pitch and heading at identical
  initial target positions, with scene variants kept in the same split.
- A second learned head predicts admission of a follow-up bite after the
  candidate action and the scripted player response.
- Planning compares sequences [candidate, bite-or-wait], discounting the
  optional second bite reward by 0.8. It replans after real observations.

This is DIRECT two-step outcome prediction, not autoregressive latent
rollout, an unrestricted action search, JEPA, V-JEPA2 or an LLM. It permits
a bounded test of preparation value without claiming the broader architecture.

Run fit/validation first, inspect its report, then invoke the fresh test:

```bash
python3 experiments/run_world_model_v2.py --stage fit \
  --driftmoor-root ../Driftmoor-spider-sim \
  --v1-checkpoint /tmp/driftmoor-world-model-v1/predictor.pt \
  --output-dir /tmp/driftmoor-world-model-v2

python3 experiments/run_world_model_v2.py --stage test \
  --driftmoor-root ../Driftmoor-spider-sim \
  --v1-checkpoint /tmp/driftmoor-world-model-v1/predictor.pt \
  --output-dir /tmp/driftmoor-world-model-v2
```

Fit refuses to overwrite a checkpoint, and test refuses to overwrite an
existing test report. Use a new experiment directory for a changed version.
Weights and datasets remain outside Git.

Protocol:
- Training encounters 1000–1159; matched pose groups 40000–40199.
- Validation encounters 11000–11029; pose groups 50000–50039.
- Fresh test encounters 30000–30039; pose groups 60000–60039.
- All candidate branches and pose variants share their scene's group ID.
- Two hidden layers of 96 units, 160 epochs, Adam 0.002, training seed 17.
- Validation loss selects epoch 102. No objective/hyperparameter changes
  were made after seeing this validation result or the fresh test.
- Training had 8,854 transitions, 139 immediate and 774 follow-up positives.
- Comparing v2_one_step and v2_two_step holds the trained model/data fixed,
  isolating lookahead scoring. Comparing v1 with v2_one_step combines data
  coverage and the extra training head; it does NOT isolate data alone.

Validation valid bites / commands:
- Zone-gated rules: 20/28.
- Sensory rules: 17/22.
- V1: 1/12.
- V2 one-step: 5/6.
- V2 two-step: 16/20.

Fresh test, 40 whole encounters:

| Controller | Bite commands | Valid bites | Rejected | Turns | Damage |
|---|---:|---:|---:|---:|---:|
| Zone-gated rules | 43 | 42 | 1 | 75 | 179 |
| Sensory rules | 20 | 17 | 3 | 220 | 61 |
| V1 | 29 | 6 | 23 | 11 | 44 |
| V2 one-step | 13 | 13 | 0 | 7 | 51 |
| V2 two-step | 25 | 24 | 1 | 34 | 73 |

V2 two-step improves strongly over v1 and finds more valid opportunities
than the same v2 model scored one step ahead. It still loses to the
chamber-gated rule comparator. That comparator is the experimental rule
controller behind the original trigger, not production Driftmoor LLM AI.

Fresh offline prediction precision/recall at 0.5:
- Immediate admission: 0.590 / 0.920 (25 positive labels).
- Follow-up admission: 0.655 / 0.767 (146 positive labels).
These are over all candidate test actions, not only selected policy actions.
The decrease from validation precision shows prediction errors remain;
24/25 selected valid bites must not be read as universal model accuracy.

A paired episode bootstrap (10,000 samples, seed 23) estimates extra valid
bites per episode for v2 two-step: +0.45 vs v1 (95% interval 0.10–0.85),
+0.275 vs v2 one-step (0.025–0.575), and -0.45 vs zone rules (-0.75–-0.175).
These describe these scripted encounters, not real gameplay effectiveness.

Reports: results/world-model-v2-validation.json and
results/world-model-v2-test.json. Full episodes, datasets and checkpoints
live in the run directory. The validation/test checkpoint SHA256 matches.

Limits remain: non-attacking scripted player, same cavern and behavior
families, coarse sensory observations, no support/attachment dynamics,
no injury tradeoff or full 3D collision. Matched poses are explicit fixtures,
not simulated climbing. Next investigate the ambush baseline's remaining
advantage and longer planning without tuning on this now-used test split.

Additional tests:
```bash
# Driftmoor experimental checkout
python3 -m pytest -q tests/test_spider_v2.py

# Spider project
python3 -m pytest -q experiments/test_world_model.py experiments/test_world_model_v2.py
```
