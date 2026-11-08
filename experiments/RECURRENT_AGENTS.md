# Two private recurrent encounter agents

This page describes the initial v1 experiment. For the current teacher and
body-center sensor schema, see [WITHDRAWAL_5E.md](WITHDRAWAL_5E.md).

This experiment trains **separate spider and Gilthaness networks**, without an
LLM, images, V-JEPA, or a shared transcript. It is supervised imitation, not
reinforcement learning or a completed JEPA architecture.

## Run and train

From /Users/jeff/Python/driftmoor-spider-ai:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m experiments.run_recurrent_agents \
  --driftmoor-root /Users/jeff/Python/Driftmoor-spider-sim \
  --output "/Users/jeff/Documents/ChatGPT/White Fang/experiments/recurrent-v1"
```

Install this repository's requirements and Driftmoor's existing requirements
in the same Python environment. No network access or model download is used.
The default run collects 160 training, 24 validation, and 32 test episodes,
then trains each network for 160 epochs. Checkpoints are selected on validation
loss; test episodes and 16 rollout episodes use separate seeds. All sequences
from an encounter remain in the same split. Output includes full demonstration
sequences, four small checkpoints, private observation rollout traces, and a
report. Keep these artifacts outside Git.

The encounter and teacher implementation is now in `spider_sim/dual_encounter.py`
in this project. Its game dependency is selected through `spider_sim/runtime.py`.

## What is learned

Each actor receives 41 numbers: sensory availability and measurement masks,
anonymous contact range/bearing/elevation and visually estimated dimensions,
web evidence, its own health, body dimensions, floor-relative height,
orientation and position, prior action, and hit/damage/pain feedback.
No target identity, species, hidden coordinates, hidden health, or other
actor's private memory is passed to the network.

A 41-to-32 MLP encodes each observation. A 48-unit GRU updates private memory.
Three heads predict:
* one of eight actions (wait, advance, retreat, left/right turn, up/down pitch,
  attack);
* whether contact was recently observed and its last observed range, bearing,
  and elevation;
* the next private observation and admitted-attack/damage feedback, conditioned
  on the executed action.

The next observation is at that actor's next turn, after the opponent responds.
Dice outcomes are stochastic: the prediction is a regression estimate.
Separate memoryless MLP networks provide an ablation; they still receive the
same prior-action and feedback inputs. Sessions reset recurrent state between
encounters. Model weights are independent for the two roles.

Rule teachers use only the respective private observations and at most six
turns of stale contact memory. Fifteen percent random action exploration
broadens training transitions. Policy labels remain the teacher's suggested
action; prediction targets follow the action actually executed.
Loss combines action cross-entropy, contact-memory supervision, and
action-conditioned next-observation/feedback regression. Padding is masked.

## Initial results

See results/recurrent-v1.json. On held-out demonstration episodes:
* Spider action agreement: MLP 80.4%, GRU 81.2%.
* Gil action agreement: MLP 87.5%, GRU 89.4%.
* Remembered range error during missing contact: spider 9.9 to 6.1 feet;
  Gil 6.5 to 4.5 feet (61 and 72 recall frames respectively).

In 16 fresh GRU-versus-GRU encounters the spider had 8 admitted attacks out
of 44 requests and dealt 10 damage; Gil had 6 of 30 and dealt 11 damage.
These are real engine combat results, but demonstrate poor attack selection.
Agreement with a rule teacher is not strategic competence. No repeated-seed
confidence interval or claim of generalization beyond this fixture is made.

## Explicit experimental shortcuts

The new dual runner bypasses the original chamber activation gate. It adds a
rectangular connected web strip, an opaque entrance curtain, ceiling
concealment broken by movement/attack, and periodic sensor outages. These
are authored training fixtures, not inferred physical web mechanics.
Spider vision is panoramic; Gil has a 130-degree view cone. Web evidence is
a coarse location cue, not proof of a spider. Initial health, horizontal
positions, spider height, and heading vary. Gil draws his existing longsword;
his spells and full wizard strategy are not represented. Spider movement
stays horizontal, with no attachment/climbing physics. Occlusion remains
2.5D. Backward movement is the current crude retreat primitive.

The teacher attacks on a coarse 15-foot contact estimate; the engine's actual
weapon reach, body volume, visibility, facing, and dice remain authoritative.
This intentionally simple teacher is a major cause of rejected attacks.
Networks currently choose directly from their policy head. Their prediction
heads are trained but **are not used for planning**. Belief targets record
past measurements, not omniscient current opponent state or a calibrated
probability distribution.

## Next training stage

Improve the observation-only teacher's reach and facing decisions, then
collect learner-visited states and relabel them (dataset aggregation).
Only after reliable action feasibility should we compare a short-horizon
planner using the learned predictor against this policy, or add reward-based
self-play. Useful objectives would balance injury, survival, resource use,
and the actor's encounter goal. The engine remains the final action and
damage authority. Training the encoder/predictor jointly toward future
latent states would be an additional JEPA-like experiment; this first version
predicts numeric observations instead.

## Verification

100 focused game/encounter tests and 8 neural experiment tests pass. The wider game suite reports 1,746 passed, 1 skipped, and 6 failures involving character locations, transcript expectations, and a missing sleeping-city GEOMETRY.md. These failures are outside the new encounter tests; the full game suite is not green.
