# Improved teacher and matched encounter evaluation

Historical v2 run. The current implementation separates retreat and flee; see
[RETREAT_AND_FLEE.md](RETREAT_AND_FLEE.md) for the current command and results.

The neural architecture remains 41 -> 32 -> GRU(48), with the same heads and
memoryless MLP ablation. This experiment improves the private-observation teacher
and measures encounter behavior, rather than adding planning or self-play.

## Review the code

- `spider_sim/encounter_teacher.py`: original `LegacyTeacher` and new
  `SensoryTeacher`; the old `ContactMemory` name aliases the improved teacher.
- `spider_sim/dual_encounter.py`: own weapon/socket profile, revised visible-body
  measurements, explicit evaluation scenarios, and demonstration generation.
- `spider_sim/encounter_evaluation.py`: matched fixtures, fixed teacher opponents,
  and actual encounter scoring.
- `experiments/run_teacher_experiment.py`: reproducible collection, unchanged
  network training, and evaluation.
- `tests/test_encounter_teacher.py`: reach, facing, pitch, vibration privacy,
  retreat direction, memory, and scoring checks.

The teacher estimates distance from its own attack socket to an observed target
body proxy. It requires a current visual contact, a conservative aiming cone,
and estimated reach before requesting an attack. It turns, pitches, or moves
closer otherwise. The engine independently enforces actual geometry and rolls.

Wounded actors select forward or backward movement according to the observed
threat direction. Backward movement is no longer assumed to be an escape when
the actor is already facing away. Home limits and walls can still block escape;
there is no wall sensor, navigation planner, or learned escape route here.

Memory lasts six actor turns and adjusts stale contact range/bearing for own
horizontal displacement. It does not track unseen opponent motion, compensate
changing floor elevations, or estimate a probability distribution.

## Observation change and compatibility

Visual range/elevation now refer to the **estimated visible body center**, rather
than the hidden authored support anchor. Range is rounded to one foot, bearing
and elevation to 15 degrees, and world-axis bounding dimensions to half a foot.
Vibration remains a coarse five-foot/45-degree source measurement with no
elevation, body dimensions, identity, or species.

The teacher gets its own equipped melee reach and attack socket as self-knowledge.
These are fixed constants for each role in this fixture, not variable extra
network inputs. Own HP, pose and position are already present in the 41 numbers.

Both teachers and both network types use the same revised sensory interface.
Thus this compares the teachers fairly under the new interface, but is not a
direct comparison against the original v1 report's sensory data.

New checkpoints contain `observation_schema="body-center-v2"`. To check a load:

```python
from experiments.recurrent_agents import load
agent = load(path, expected_schema="body-center-v2")
```

Old checkpoints still load without an explicit schema check for historical
inspection. They should not be evaluated with the revised sensor interface.
The general recurrent training command also now writes the revised schema.
Older v1 data and results are retained as historical artifacts.

## Run

From /Users/jeff/Python/driftmoor-spider-ai:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m experiments.run_teacher_experiment \
  --driftmoor-root ../Driftmoor-spider-sim \
  --output /tmp/driftmoor-teacher-v2
```

Use an empty output directory. Defaults: 160 training episodes (90000..90159),
24 validation episodes (91000..91023), 32 test episodes (92000..92031), and
160 epochs. Validation loss selects checkpoints; test data does not select them.
There is 15% exploration during collection; the teacher labels the policy,
while future observations follow the action actually executed.

Evaluation uses eight fresh seeds 94000..94007 in each of six scenarios:
mixed, ground, ceiling, wounded spider, wounded Gil, and Gil facing away.
These give 48 encounters per pairing. Ten pairings include homogeneous policies
and each role tested against a fixed improved-teacher opponent. Final evaluation
uses fresh seeds after the initial exploratory comparison at 93000..93007.

Full outputs for the completed run, including weights and per-step traces:
`/Users/jeff/Documents/ChatGPT/White Fang/experiments/teacher-v2-final/`.
Only the compact summary is copied to `results/teacher-v2.json`.

## Results against a fixed improved-teacher opponent

Each row evaluates the named actor policy against the improved teacher controlling
the other actor. Attack admission is geometry/visibility permission, not a dice hit.

| Actor | Policy | Admitted / requested attacks | HP damage dealt / received | Survived |
|---|---|---:|---:|---:|
| Spider | Legacy teacher | 39 / 67 | 124 / 86 | 44 / 48 |
| Spider | Improved teacher | 39 / 41 | 185 / 68 | 45 / 48 |
| Spider | MLP | 28 / 52 | 161 / 38 | 45 / 48 |
| Spider | GRU | 15 / 19 | 68 / 56 | 45 / 48 |
| Gil | Legacy teacher | 12 / 66 | 11 / 119 | 47 / 48 |
| Gil | Improved teacher | 32 / 39 | 68 / 185 | 47 / 48 |
| Gil | MLP | 0 / 0 | 0 / 120 | 48 / 48 |
| Gil | GRU | 2 / 7 | 15 / 129 | 48 / 48 |

The teacher issues substantially fewer invalid attacks. Neural Gil still rarely
attacks despite roughly 77-78% held-out teacher-action agreement. This illustrates
why aggregate imitation accuracy and survival alone are insufficient measures:
passive behavior can survive while failing to engage. The MLP spider deals more
damage than the GRU spider in these fixtures; this run does not demonstrate a
general combat benefit from memory.

Wounded movement increases actual separation on 21% of improved-teacher spider
escape moves and 92% of Gil's (homogeneous teacher encounters). The spider is
often blocked by its home boundary. The legacy spider's corresponding rate is
22%, so this experiment does **not** show an aggregate retreat improvement.

Scoring reports attack requests, admissions/rejections, actual HP loss inflicted
and received, rolled damage separately (overkill can exceed HP loss), survival,
action counts, blocked moves, and separation change on low-health movement.
Privileged positions/HP are used only by the scorer, never to choose an action.
Damage received includes all HP loss during the encounter. Different policies
consume different combat dice even with matching initial seeds.

One training seed, a single cavern, simplified loadouts and a small scenario
suite do not establish statistical superiority. No goals for successful passage,
resource use or encounter completion have been added. No game-main changes.

## Verification and next step

80 project tests pass. The full training/evaluation command completed, with 480
matched evaluation encounters. Network architecture and existing game mechanics
were kept fixed. Changes are intentionally uncommitted pending user review.

The next experiment should inspect/rebalance rare attack demonstrations and
collect learner-visited states with teacher labels. Retreat needs observation
of obstacles or an explicit navigation model before it can handle a confined
chamber well. These should precede claims about better neural strategy or
reward-based self-play.
