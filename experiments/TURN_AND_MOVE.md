# Turn-and-move correction

Historical v4 experiment. Current withdrawal rules use full Speed/Dash and
opportunity reactions; see [WITHDRAWAL_5E.md](WITHDRAWAL_5E.md).

The earlier [retreat/flee experiment](RETREAT_AND_FLEE.md) charged a full action
slot for each 30-degree heading change. Its escape survival results were distorted
by that timing model and should not guide strategy decisions.

## Corrected command semantics

A retreat or flee command applies its complete desired orientation and resolves
movement in **one existing six-second actor slot**. It does not take six
additional slots to turn 180 degrees.

- Retreat faces the perceived threat, aims toward its observed body, and backs away.
- Flee faces away from the perceived threat, levels pitch, and advances.
- Advance can turn toward the contact while moving.
- Face is one complete stationary orientation command, not a series of 30-degree turns.
- Look-up/down remain explicit 30-degree inspection adjustments, not prerequisites
  to fleeing.

Eight network outputs remain: wait, advance, retreat, face, flee, look_up,
look_down, attack. Face and flee replace the old left/right turn outputs.
The 41-input encoder, GRU, prediction heads and training loss retain their sizes.

Each movement command still requests a six-foot research step, capped by the
game's actual movement budget and subject to boundaries, obstacles and home
restrictions. This is not yet Dash, a full-Speed escape, or a complete 5e turn.
Movement and orientation are atomic at this simulation resolution; no sub-turn
angular velocity or continuous collision trajectory is simulated.

## Private command decoding

Teacher commands carry a desired orientation derived from that actor's
observations and remembered contact. The environment applies it; it does not aim
using hidden opponent coordinates.

Neural commands use current contact measurements, or their own learned belief
head's remembered bearing/elevation when contact is absent. With neither,
movement preserves the actor's current heading. Flee can therefore continue
along its already-established escape heading after the contact estimate expires.
The teacher still waits if it has never sensed a threat direction.

The network now explicitly selects retreat or flee; its other motives remain
unreported. There is no hidden rule that converts a neural advance into fleeing
according to health.

## Rear exposure and 5e

The scorer measures whether the opponent is within 60 degrees of the actor's
rear, and counts ordinary incoming rear attack requests, admissions and actual
HP damage. Exposure does not bypass weapon reach, visibility or the aiming cone.

The current resolver has no implemented backstab bonus, Sneak Attack feature,
Disengage protection or opportunity attack reaction. Rear attack damage in this
report uses ordinary combat dice. It is not a backstab damage multiplier.

The natural 5e follow-up is:
- Retreat: Disengage plus normal movement, avoiding opportunity attacks.
- Flee: Dash plus movement, gaining distance but potentially provoking an enemy
  reaction when leaving its reach.
- A turn or facing change alone does not trigger an opportunity attack.

See the official [2014 combat rules](https://www.dndbeyond.com/sources/dnd/basic-rules-2014/combat)
and [2024 combat rules](https://www.dndbeyond.com/sources/dnd/br-2024/playing-the-game).
Those action/reaction mechanics are recommendations, not implemented changes
in this timing correction. A 5e round is approximately six seconds for all
combatants; this experimental runner still advances six seconds per actor slot.

## Reproduction and artifacts

From /Users/jeff/Python/driftmoor-spider-ai:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m experiments.run_teacher_experiment \
  --driftmoor-root ../Driftmoor-spider-sim \
  --output /tmp/driftmoor-turn-and-move-v4
```

Use an empty output directory. Training uses 160 episodes at 110000+, validation
24 at 111000+, test 32 at 112000+, with 160 epochs and the same architecture.
Evaluation uses 114000..114007 across eight scenarios and ten pairings:
640 matched encounters. Frozen checkpoints were rescored after adding explicit
neural command labels to the traces; training was not repeated for that logging change.

Full data and weights:
`/Users/jeff/Documents/ChatGPT/White Fang/experiments/turn-and-move-v4/`

The compact report is `results/turn-and-move-v4.json`.
Checkpoints and datasets use `body-center-motion-v4`: the action names and
prior-action one-hot meanings changed despite unchanged dimensions.
Checkpoint loading checks this schema by default. Historical checkpoints may be
inspected using an explicit `expected_schema=None`, but must not be deployed
with the new action decoder.

## Measured results

Current-teacher-versus-current-teacher:
- All 132 spider retreat backsteps and 143 Gil retreat backsteps retained the
  weapon's aiming cone toward the opponent.
- Rear exposure occurred on 110/112 low-health forward spider moves and 143/143
  Gil moves.
- Across these encounters, the spider received 19 admitted attacks from the rear
  for 31 actual HP damage; Gil received 4 for 9 HP. These counts include all
  rear attacks, not just attacks immediately after a flee command.
- In dedicated critical-health cases, Gil survived 8/8; the spider 0/8.
  The spider remains confined to its authored home. Different seeds and changed
  commands prevent treating old/new survival numbers as an isolated timing
  effect.

Against fixed current-teacher opponents, the GRU spider now made 113 retreat
backsteps, all retaining weapon focus; GRU Gil made 154, retaining weapon focus
on 98.1%. This is a new learned policy and command decoder, not the old network
reevaluated under changed physics. Attack selection remains imperfect.

No reward-based self-play is used. These remain supervised imitation experiments,
not optimal-strategy or broad-generalization results.

## Verification and review

95 project tests pass. Tests check one-slot movement from different headings,
movement-budget limits, contact-memory continuation, neural commands using
predicted rather than hidden bearings, rear exposure beyond reach, and an
admitted rear attack within reach.

Everything remains unstaged and uncommitted for review. Driftmoor's game worktree
and main checkout were not changed by this correction.
