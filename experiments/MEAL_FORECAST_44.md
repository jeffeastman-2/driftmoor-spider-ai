# Training perception and memory with 44 inputs

The frozen-memory work was checkpointed as `51629a3` before this experiment.
This experiment trains **a copy** of perception, memory and the outcome head.
The live pursuit encoder, GRU and actor scorer remain unchanged.

## Architecture and data flow

```text
44 current measurements
  → Linear(44,64) → ReLU → Linear(64,32) → ReLU
  → GRU(input=32, hidden=48, one layer), updated each turn
  → 48 memory values + 3 response-choice indicators
  → Linear(51,64) → ReLU → Linear(64,32) → ReLU → Linear(32,12)
```

The extra eight inputs are the same private measurements as before: hunger,
health fraction, incoming damage, meal visibility/range/remaining food/wrapping,
and whether the previous task was food handling. Health is already among the
old inputs, so these are 44 input slots, not 44 independent facts. **All eight
now enter the first MLP and the GRU. None bypass memory into the outcome head.**

The 12 outputs represent injury, net hunger reduction, renewed damaging attack
probability and death probability at 3, 6 and 12 turns, for each proposed
response. They are explicit outcome forecasts, not new motor commands or a
JEPA latent-target objective. Response choices retain the same authored
continuations: pursue, return after observed withdrawal, or return after a
successful retaliation.

## A matched comparison

All 1,330 histories were reconstructed by replaying the original teacher
trajectories. The original frozen latent and context matched at **every**
saved origin. Labels were not changed or resampled: the same 10,128 fork
rollouts, split seeds and unseen five-turn test return delay were used.
The longest history is 36 turns. Histories reset at episode boundaries;
padding is excluded with packed sequences.

The expanded encoder copies the original 36-input weights, sets the additional
eight columns to zero and initially reproduces the original representation.
Then the entire predictor-copy encoder and GRU are trainable. The new columns
acquired nonzero weights, and encoder/GRU changes were measured after fitting.
There is no ambiguity about whether hunger can reach memory: tests verify that
route and that past hunger can affect the history model while leaving the
snapshot ablation unchanged.

Three variants, each fitted with three optimizer seeds:

- **Retrained 36:** train perception and memory without the added context.
- **Retrained 44:** train perception and memory with all eight added inputs.
- **44 snapshot:** the same trainable architecture, but reset memory and supply
  only the current frame. This tests whether history helps beyond current inputs.

Encoder/GRU learning rate is .0003, head rate .002, gradient norm clipped to 1.
Training averages rollout targets per origin and response, producing the same
expected MSE/BCE gradients as repeating the individual draws. Validation selects
epochs with patience 20, maximum 120. All comparisons use normalized outcomes;
these fitting losses are not preferences or decision-cost weights.

## Results on the same test set

Lower values are better. Risk error is Brier score, not accuracy. Hunger error
is measured as a fraction of the full hunger scale.

| Model | 12-turn attack risk error | Injury error | Hunger-reduction error |
|---|---:|---:|---:|
| Previous simple recent-damage baseline | .2122 | 2.79 HP | .211 |
| Previous current-context MLP | **.2079** | 2.85 HP | **.104** |
| Previous frozen memory + context head | .2205 | 2.89 HP | .122 |
| Retrained 36 history | .2301 | 2.82 HP | .157 |
| Retrained 44 history | .2290 | 2.87 HP | .142 |
| Retrained 44 snapshot | .2097 | **2.67 HP** | .148 |

The 44-input history model improved hunger prediction modestly over the
36-input history model. It did not beat the earlier direct-context MLP, and
history did not help attack-risk prediction: the snapshot model was stronger.
This does not establish that extra inputs are useless or that GRUs cannot
learn threat history; it establishes that this training setup did not produce
a superior recurrent predictor.

Risk calibration remains weak. The 44-history model predicted 55.8% damaging
attack incidence over twelve turns; actual incidence was 40.9%. The snapshot
predicted 51.8%. None should be treated as calibrated safety estimates yet.

The injury-only choice diagnostic also failed to establish a 44-input benefit:
44-history and snapshot both matched the fixed-response baseline at 0.50 HP
of average excess injury against the empirical best of four future draws.
Retrained 36 scored 0.41 HP, an exploratory result requiring independent testing;
it did not have the best risk forecasts. These are offline ranking diagnostics,
not full-encounter improvements by a newly deployed actor.

## What the data can and cannot teach

The test origins contain 192 first injuries, 150 first observed retreats and
only 16 renewed injuries. The collector does **not** sample a separate decision
after a second observed retreat. Thus it gives little direct evidence for the
specific memory task we want: a spider sees an apparently similar retreat but
remembers this opponent previously came back. At renewed injury, the current
damage cue itself already reveals trouble.

This sampling limitation is established by the collector and counts; whether
it explains the model ordering is a hypothesis, not a demonstrated cause.
At an opponent's first retreat, hidden future behavior may genuinely be
unidentifiable from its sensory history. More network capacity cannot reveal
information the spider has not observed.

The next useful corpus should include first **and subsequent** withdrawals
with otherwise similar current observations, retaining the opponent's sensed
track across encounters. Evaluate whether history improves forecasts there,
with broader return timings and fresh test seeds, before deepening the MLP.
The new-input pipeline can remain an experimental foundation without promoting
these weights into the game.

## Reproducibility and limits

```sh
python3 -m experiments.meal_forecast_histories /path/to/fork-data --model /path/to/spider.pt --output /new/history-data
python3 -m experiments.train_recurrent_meal_forecast /new/history-data --source-model /path/to/spider.pt --output /new/training-output
```

The original actor checkpoint is checked by hash and its tensors are checked
for equality after fitting. Saved predictor ensembles reproduce test scores
when reloaded. All 222 regression tests pass. `load_predictor()` loads a read-only predictor copy and checks
its source checkpoint contract.

The test set was examined in the preceding experiment. This is a matched
research comparison, **not** a fresh independent final acceptance test. Two
rooms, scripted opponent behaviors and short horizons remain limitations.
No new reward function, combined utility, learned goal selector or live policy
has been introduced.

Weights and full results live outside Git at:
`/Users/jeff/Documents/ChatGPT/White Fang/experiments/meal-forecast-retrained-v1`.
The compact result is `experiments/results/meal-forecast-recurrent44-v1.json`.
