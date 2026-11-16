# One learned spider pursuing scripted prey

**Milestone not met. Neither live controller is replaced.**

We narrowed the problem to one spider and one prey creature, then trained a
separate MLP/GRU controller to score candidate actions from private observations.
Keeping a target is not enough to make pursuit reliable. Repeated unreachable
attacks, blocked movement and loops remain measurable failures.

## Independent final comparison

Both model checkpoints were frozen before testing 32 new seeds (590000–590031)
per prey behavior, with a 48-round limit. The same starting poses, creatures,
scripted prey policy and engine seed are paired across controllers. Only the
spider changes. The fixture disables unrelated resident activations, including
the goblin, so these encounters remain exactly two participants.

A catch means the engine reduces prey HP to zero. No wrapping or eating is
implemented. Native reach, attack admission, hit rolls and damage still apply.

| Prey behavior | Scripted catches | Learned pursuit (21 action features) | Added local distances (23 features) |
|---|---:|---:|---:|
| Stationary | 32/32 | 27/32 | 29/32 |
| Ordinary fleeing | 32/32 | 30/32 | 22/32 |
| Cautious fleeing | 31/32 | 27/32 | 27/32 |

The ordinary-fleeing learned controller makes only one two-step return loop,
but still has 70 blocked movement requests and 47 rejected attacks. Against
cautious fleeing prey it makes 85 return loops, with a worst consecutive loop
streak of 43. The script has no such loops. The distance-feature variant does
not reliably help: 312 blocked moves against ordinary fleeing prey and a
40-loop maximum streak against cautious prey.

These results reject promotion. A decent catch percentage does not make a
controller acceptable when some failures consume the encounter in a loop.
See `results/single-prey-final-comparison.json` for all counters and catch times.
Tests confirm software contracts; their passing does not establish behavior.

## What was built

`spider_sim/pursuit.py` contains:

* An authored persistent target task. It chooses a visible contact once and
  retains that track rather than switching whenever another is closer.
* Up to four unseen turns of dead reckoning using actual movement feedback,
  then target expiration. It never reads a hidden opponent position.
* A 36-value observation: the existing 24 self-state values plus the selected
  contact's 12 measured values. Encoder 36→64→32; GRU memory 48.
* A learned action-consequence scorer: (48 + 21)→64→32→1. Candidates retain the
  existing wait, five/ten-foot moves at eight relative headings, and a visible
  attack on the selected track. No authoritative reach mask is applied.

The 21 action values comprise the prior 19-value clearance-aware motor input
plus two direction components relative to the goal. The 23-value variant adds
current planar distance and a simple projected distance after the requested
move, limited by the existing local clearance measurement. These are derived
local kinematics, not native future rollouts or hidden target data.

The persistent goal is authored, not learned. Motor selection and action
consequence values are learned. This is a separate focused experiment, not a
replacement for the older multi-contact seven-output world predictor. Its
single scalar consequence is **not a full latent world model, JEPA joint
embedding loss, learned configurator, or reinforcement-learning system**.

## Training and supervision

For every collected state, offline training clones the engine for every
candidate action, applies the candidate and one scripted prey response, then
scores the actual result. Supervised goal cost is zero for defeated prey;
otherwise it is `1 + min(planar_distance, 60)/10 + remaining_prey_HP/max_prey_HP`.
Distance is in feet. This authored task objective uses privileged engine truth
**only to generate training targets**. Opponent HP, global positions, RNG state
and future outcomes never enter policy inference. The scalar target includes
native combat dice outcomes; the network must learn expected consequences.

Training combines scalar regression with soft ranking of actions by these
engine outcomes. Initial trajectories mix engine-preferred and exploratory
actions; subsequent collection includes student-visited states. The initial
trial exposed a recurrent-state synchronization bug: on mixed trajectories,
the student skipped some updates. Collection now updates its GRU every turn
and records the actual executed action for dead reckoning.

The 21-feature model continuation reuses the initial collected corpus and
adds two 24-episode student-state passes. Closed-loop validation on separate
seeds selects among phases by catches, then rejected attacks, then loops.
The 23-feature trial starts from a fresh 48-episode corpus, plus a corrected
24-episode student pass. Models use optimizer seed 42 and 300 epochs per fit,
retaining best supervised-validation weights within a fit. The initial 21-value
corpus was collected before the synchronization fix; its engine labels remain
valid, but its trajectory distribution reflects that bug. This is not a clean
single-variable architecture ablation, and the report makes no such claim.

Earlier development comparisons reused earlier seed ranges. The final table
above uses a fresh range, after both checkpoints were frozen. It is still one
optimizer seed and the same familiar Level-2 room regions. The script also has
continuous headings while neural movement remains quantized in 45-degree
increments. That motor-interface difference must be controlled in a subsequent
comparison; these numbers do not establish the neural architecture's ceiling.

## What this tells us

Target commitment is useful structure but does not solve action evaluation.
The scalar predictor can rank a rejected bite or a repeated movement loop too
favorably. More examples and extra distance inputs did not eliminate that.
A next controlled test should separate physical outcome prediction (admission,
actual displacement, contact continuity) from task value, and give both
controllers equivalent motor commands before increasing encounter complexity.
Do not paper over failures with a reversal penalty or silently replace the
learned action chooser with the successful script.

## Reproduce

From the spider project, with the experimental Driftmoor worktree available:

```sh
python3 -m experiments.train_single_prey --output /tmp/pursuit-new
python3 -m experiments.continue_single_prey \
  --source /tmp/pursuit-new --output /tmp/pursuit-continued-new
python3 -m pytest -q experiments/test_single_prey.py
```

Current training defaults to 23 action features; pass `--action-features 21`
for the smaller input contract. Both load using explicit input-width metadata.
Current collection includes the synchronization fix, so a fresh 21-value run
will differ from the historical buggy first corpus. The frozen weights,
corpora and reports remain outside git under
`/Users/jeff/Documents/ChatGPT/White Fang/experiments/single-prey-*`.
The final comparison is reproduced by loading each frozen checkpoint and
calling `experiments.train_single_prey.evaluate(model, range(590000,590032))`.
No model binaries or encounter corpora are added to git.

The existing port-8081 baseline and port-8082 three-species experiment were not
reset, stopped, or swapped for these focused controllers. No commit was made.
