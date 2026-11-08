# Retreat and flee: separate survival behaviors

Historical v3 experiment. Its 30-degree turn-slot timing has been corrected.
See [TURN_AND_MOVE.md](TURN_AND_MOVE.md) for current commands and results.

This supersedes the movement policy in the [earlier teacher comparison](TEACHER_EXPERIMENT.md).
All work remains uncommitted for review.

## Behavior

| Actor health | Spider | Gil |
|---|---|---|
| Engage normally | 35% or higher | 45% or higher |
| Retreat | 20% to below 35% | 25% to below 45% |
| Flee | Below 20% | Below 25% |

These are authored experimental preferences, not learned optimal thresholds.

**Retreat** turns toward the observed or remembered threat, adjusts pitch toward
its observed body when available, and moves backward. The actor maintains its
weapon direction and tries to retain sensory contact. If the target is already
behind it, the actor turns toward it before backing away.

**Flee** turns toward the direction opposite the observed threat bearing,
levels pitch, and advances. It does not blindly reverse the current heading:
an actor already facing away can advance immediately. Turning consumes ordinary
action slots, so a 180-degree turn takes six 30-degree turns. That delay can be
dangerous while the opponent continues attacking.

After contact expires, fleeing continues along the last evidence-based escape
heading. This is a single remembered direction, not an unseen opponent position
or an escape route. Injury alone supplies no threat bearing: an actor with no
contact history waits rather than inventing a direction.

The spider's panoramic sight may keep seeing an opponent while fleeing;
its fangs still point away. Gil's view cone can lose sight during the turn.
Neither behavior currently has navigation around obstacles or an escape
destination. The spider's home restriction still limits movement.

## Code and network boundary

- `spider_sim/encounter_teacher.py`: `SensoryTeacher.intent()` selects
  engage/retreat/flee; `choose()` implements the movement sequence.
- `SensoryTeacher.update()` retains an escape heading after contact memory
  expires. Its contact estimate still compensates only horizontal own movement.
- `spider_sim/dual_encounter.py`: records `teacher_intent` as a trace label,
  adds critical-health cases, and provides close visible wounded scenarios.
- `spider_sim/encounter_evaluation.py`: records the teacher's declared intent
  and separately scores actual movement, facing, contact, damage and survival.

The neural architecture remains **41 inputs -> 32 encoder features -> 48 GRU
units**, with the existing eight action outputs and prediction heads. No intent
output has been added. The historical primitive named `retreat` means a backward
step; the higher-level retreat behavior also includes turning and pitching.
Flee is a turn/level/advance sequence built from existing primitives.

Teacher intent labels are saved for inspection, but are not extra network inputs
or an extra loss. The networks learn the corresponding motor actions through
imitation. Their intentions are unreported; the scorer never assigns them the
teacher's intent.

## Training and evaluation

From /Users/jeff/Python/driftmoor-spider-ai:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m experiments.run_teacher_experiment \
  --driftmoor-root ../Driftmoor-spider-sim \
  --output /tmp/driftmoor-retreat-flee-v3
```

Use an empty output directory. Defaults retain 160 epochs, unchanged loss,
optimizer and network architecture. Entire episodes stay in one split:
160 training seeds 100000..100159, 24 validation seeds 101000..101023,
32 test seeds 102000..102031.

Collection rotates through mixed, ground, ceiling, wounded spider, wounded Gil,
critical spider and critical Gil scenarios. The wounded and critical fixtures
begin close enough to see the threat, at 25% and 10% starting health respectively,
with the opponent at full health. This is deliberate demonstration coverage.
Mixed scenarios also include 10% health. Exploration remains 15%.

Eight fresh evaluation seeds 105000..105007 run across eight scenarios (the
collection families plus Gil facing away), giving 64 encounters per pairing.
Ten pairings test legacy teachers, current teachers, MLPs and GRUs, including
each learned role against a fixed current-teacher opponent: **640 encounters**.
Earlier exploratory runs at 104000..104007 exposed missing retreat coverage
for Gil; the final covered fixtures use fresh evaluation seeds.

Full artifacts, including checkpoints and private observation traces, are at:

`/Users/jeff/Documents/ChatGPT/White Fang/experiments/retreat-flee-v3-covered/`

The compact report is `results/retreat-flee-v3.json`. The sensor schema remains
`body-center-v2`; teacher version is `retreat-flee-v3`. Prior result files
are retained as historical measurements, not overwritten.

## What the run shows

In current-teacher-versus-current-teacher encounters:

| Measurement | Spider | Gil |
|---|---:|---:|
| Low-health backward steps | 44 | 102 |
| Heading remains toward opponent | 97.7% | 100% |
| Weapon socket remains facing opponent | 100% | 100% |
| Separation increases while both facing checks pass | 59.1% | 100% |
| Low-health forward steps | 69 | 36 |
| Forward heading points away from opponent | 97.1% | 100% |
| Forward movement increases separation while facing away | 46.4% | 100% |

Lower spider progress reflects its confined home area as well as opponent
movement. Good movement-direction scores do not mean fleeing is safe:
only 2/8 critically wounded spiders and 3/8 critically wounded Gils survived
their dedicated scenarios. Turning exposes the actor to attacks.

The scores describe executed movement. A movement command that hits a wall
still appears in the denominator. Backward motion at low health is scored as a
retreat candidate, forward motion as a flee candidate; that geometric
classification is not a claim about a neural network's intent. The report's
`intents` field is `unreported` for networks and the legacy teacher.

Facing toward/away uses horizontal heading within 30 degrees. Weapon focus uses
the actual three-dimensional socket aiming cone. Sight retention requires a
visual contact before and after the action; sensor outages and curtains can
break it independently of facing. Scoring alone uses privileged geometry/HP.
Action selection still receives private observations only.

Against fixed current-teacher opponents, the learned policies remain uneven:
* The GRU spider made **no backward retreat steps** in its 64 encounters,
  despite 153 backward-step examples in training.
* Neither learned Gil made a backward retreat step, despite 187 training examples.
* The MLP spider made 29 backward steps, all maintaining weapon facing.
* GRU teacher-action agreement was 79.7% for the spider and 79.8% for Gil,
  but those averages hide the failure to reproduce retreat.

Thus the teacher now distinguishes the behaviors; we have **not** established
that the neural policies reliably distinguish them. Behavior-specific evaluation
reveals a failure that overall action accuracy misses.

## Verification and next step

Tests cover turning before backward retreat, turning before forward flight,
already-away headings, continued flight after contact expiry, unknown threat
direction, both actors' wounded fixtures, and truthful intent reporting.
The project suite has 88 passing tests.

The next learning experiment should address behavior/action imbalance and
learner-visited states, measuring retreat and flee separately. No larger network,
JEPA planner, reward-based self-play, navigation, or game-main changes have been
introduced.
