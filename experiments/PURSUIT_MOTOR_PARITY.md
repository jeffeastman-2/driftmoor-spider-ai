# Motor parity and a learned pursuit prior

Checkpoint before this work: spider repository `94f27b6`; companion Driftmoor
worktree `421e8f4`. Both are local commits on `codex/spider-world-model`.
The companion revision and hashes of external weights are recorded in
`results/checkpoint-2026-11-16.json`. No weights or datasets are tracked.
Subsequent experiments below remain changes after that checkpoint.

## Does the motor interface explain the gap?

We froze the previously selected pursuit model and tested three controllers
on 32 fresh seeds per prey behavior (610000–610031), 48 rounds each. The matched
script uses the learner's exact motor inventory: five/ten-foot moves at eight
body-relative headings, wait, and visible-target attack. It projects each
continuous scripted movement to the nearest available endpoint. It adds no
route planner, hidden data, native reach check or collision oracle.

| Catches | Original script | Motor-matched script | Frozen neural controller |
|---|---:|---:|---:|
| Stationary prey | 32/32 | 30/32 | 28/32 |
| Ordinary fleeing | 32/32 | 28/32 | 29/32 |
| Cautious fleeing | 32/32 | 27/32 | 26/32 |

Restricting movement damages the script too, and can leave it repeating rejected
attacks. Thus some of the earlier gap comes from the motor interface. However,
the frozen learner still makes 141 two-step return loops against cautious prey,
with a 44-loop maximum streak; the matched script makes only one. Motor parity
does not explain away that learned failure.

## Give the actor a precise approach option

We add one optional motor proposal: heading toward the measured goal contact,
with `distance = min(10, max(1, planar_range - 5))`. This is the existing script's
five-foot standoff convention, **not a general body-reach calculation**. The
network may choose this proposal, another movement, wait or attack. Nothing
forces approach after a bad choice; no attack is masked by engine legality.

A second frozen-weight test (620000–620031) shows that adding the candidate alone
is insufficient. It reduces blocked movement against ordinary fleeing prey
from 161 to 49, but catches change from 26/32 to 25/32. Rejected attacks and long
loops remain. This is an inference-time candidate expansion, not retraining;
the old network was not trained on those precise headings/step lengths.

## Learn the competent starting policy before using a cost critic

The next experiment keeps the MLP/GRU structure and trains its action chooser
from the script's private-sensory decisions, with the precise approach candidate
available. Inputs remain the same: 36 observation values, a 48-number GRU state,
and 21 action values. At inference the actor scores all candidates through
(48+21)→64→32→1 and chooses the lowest negative logit. No script is called to
select its action. The authored approach motor still determines how a selected
approach is physically executed; the neural actor learns when to choose it,
attack, wait or choose another direction.

Training: 64 teacher episodes (572 frames), 12 validation episodes, 400 epochs
with best validation cross-entropy selection. Then one 32-episode student-state
aggregation pass, followed by another 400 epochs with the same selection rule.
At student-visited states, the teacher supplies a label; every actual action is
resolved by Driftmoor. All recurrent updates and movement memory follow actual
executed actions. Optimizer seed is 42. Teacher/student/validation/test seed
ranges are disjoint and retained in `results/pursuit-imitation-v1.json`.

After freezing the model, 32 new seeds per behavior (653000–653031), 48 rounds:

| Catches | Original script | Learned actor + approach motor |
|---|---:|---:|
| Stationary prey | 32/32 | 32/32 |
| Ordinary fleeing | 32/32 | 32/32 |
| Cautious fleeing | 29/32 | 28/32 |

There are no two-step return loops or exact reversals in any learned test run.
All learned attacks are admitted: 44/44, 44/44 and 38/38 respectively. Native
combat dice still determine hits and damage. Blocked movement requests are
0, 2 and 19, versus the script's 0, 2 and 6. Cautious-prey navigation remains
worse in this small comparison, so this is a promising prior, not a demonstrated
replacement for the game controller.

These scores are **not JEPA cost forecasts**. Checkpoints store
`value_kind=negative_action_logit`, and inference records `action_scores` rather
than presenting them as physical predictions. Earlier goal-cost checkpoints
still load with their original semantics. No weights change during inference.

## Interpretation and next steps

A small network can learn useful pursuit behavior with these inputs and an
appropriate motor interface. This result does not show that the engine-trained
scalar cost critic learned useful planning. Imitation, motor support and state
distribution all change in the new experiment; it is not a clean one-variable
ablation proving any single change responsible. There is one optimizer seed,
96 paired tests, familiar room regions, and only one prey at a time. No claim
of generalization to new landscapes or three learned species is justified.

The practical sequence now is:

1. Keep this actor as the competent initial prior, with the existing script as
   the game default. Test other optimizer seeds and genuinely different layouts.
2. Learn separate, measurable physical outcomes from its actual encounters:
   admitted bite, actual displacement, distance change and contact continuity.
   Measure positive/negative recall and calibration on student-visited states.
3. Evaluate those predictions in parallel without allowing them to steer the
   actor. Enable cost-guided changes only when independent closed-loop tests
   retain catching and avoid new loops; then investigate two-action move/bite
   sequences. Do not return to three learned species before that gate.

This is supervised policy initialization, not learned goal discovery, a full
latent world model, reinforcement learning, or a JEPA joint-embedding objective.
During these experiments neither localhost viewer was reset or replaced. This new work is not committed
or pushed as part of the earlier requested checkpoint.

## Reproduce

```sh
python3 -m experiments.motor_parity --model /path/to/frozen/spider.pt \
  --output /tmp/motor-parity.json
python3 -m experiments.motor_parity --approach --model /path/to/frozen/spider.pt \
  --output /tmp/approach-candidate.json
python3 -m experiments.imitate_pursuit --output /tmp/pursuit-imitation-new
```

Use new output paths. Frozen comparison weights were at
`/Users/jeff/Documents/ChatGPT/White Fang/experiments/single-prey-pursuit-v2/spider.pt`.
The new prior and training traces are at
`/Users/jeff/Documents/ChatGPT/White Fang/experiments/pursuit-imitation-v1`.

## Viewer installation

At the user’s subsequent request, port 8082 now runs this imitation-trained
spider via `--spider-model /path/to/pursuit-imitation-v1/spider.pt`. All other
residents retain scripted controllers. Annotations say **Learned pursuit** and
report the committed contact and actual engine results rather than invented
cost forecasts. The new episode starts paused at round zero in
`/Users/jeff/Documents/ChatGPT/White Fang/experiments/level2-improved-pursuit-live`.
The previous neural run remains in `level2-neural-curriculum-preview`; port 8081
remains the scripted baseline. Performance in the full ecology is still an
extension beyond the one-prey training benchmark.
