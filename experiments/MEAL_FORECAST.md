# First learned meal-defense forecasts — November 17, 2026

Checkpoints before this experiment: spider `6d9f793`, game `115a935`.
This is the **fixed representation baseline**, not a new live actor.

## What is learned

The pursuit encoder and GRU remain fixed. New MLP heads predict consequences
of committing to each existing authored response: pursue, guard (resume on
observed withdrawal), or resume after successful retaliation. No hidden
attacker behavior label is provided to a predictor, and no goal-cost weights
or scalar reward were introduced.

Each head predicts four quantities at three horizons (3, 6 and 12 turns):
actual further HP lost, net hunger reduction, probability of a further damaging
attack, and probability of death. That is **12 outputs per response**. This is
explicit outcome prediction supervised by engine experience, not a JEPA
joint-embedding objective or latent future-state predictor.

All three head variants use Linear → ReLU → Linear → ReLU → Linear, with
64 and 32 hidden units. They differ only in their inputs:

| Variant | Inputs |
|---|---|
| Frozen memory | 48 fixed GRU numbers + 3 response indicators = 51 |
| Current context | 8 observed context numbers + 3 response indicators = 11 |
| Memory + context | 48 + 8 + 3 = 59 |

The eight context measurements are hunger, health fraction, incoming damage
fraction, visibility of the selected meal, measured meal range, remaining food,
wrapped status and whether the previous task was food handling. They feed the
new head directly. The old 36-input perception encoder still does not receive
hunger or meal state; its weights and its 48-state GRU are unchanged.

Continuous losses use normalized health loss and net hunger reduction; attack
and death use binary cross entropy. Regression outputs are bounded; risk outputs
are probabilities. These fitting losses are not decision costs or preferences.

## Where the experiences come from

The collector drives 672 controlled starting scenarios through the same engine
as the earlier batch, then copies world state and private controller state at
first injury, observed withdrawal and (where encountered) renewed injury.
Each copy tries each response with independent future dice draws, paired across
responses. Original worlds, RNG state, controller memories and model tensors
are not modified by the forks.

There are 1,330 decision origins and 10,128 fork rollouts:

| Split | Dice seeds | Origins | Fork rollouts |
|---|---|---:|---:|
| Training | 81000–81007 | 786 | 4,716 |
| Validation | 82000–82001 | 186 | 1,116 |
| Test | 83000–83003 | 358 | 4,296 |

Training returners withdraw for two or four turns, validation returners for
three, and test returners for five. Thus the test return delay is unseen.
The two rooms, species, starting body conditions and scripted response repertoire
remain familiar. Histories and all their branches stay in a single split.
Only 16 held-out origins contain a renewed injury; that slice is small.

The first action is resolved at the copied spider decision point, without an
extra opponent turn. Later actions include the opponent response. Labels run
through the specified horizon, even if the spider becomes full; death is
absorbing. Three optimizer seeds are fitted for every variant; early stopping
uses validation loss only, and the three heads are averaged at evaluation.
Model selection and normalization never use test outcomes.

## Results

Risk error below is the Brier score: mean squared error of a probability against
the actual event. Lower is better. It is not a classification accuracy.

| Predictor | Attack risk error, 3 turns | 6 turns | 12 turns | Injury error at 12 turns | Hunger-reduction error at 12 turns |
|---|---:|---:|---:|---:|---:|
| Training mean by response | .1711 | .2141 | .2548 | 3.23 HP | .215 |
| Recent-damage cue + response | — | — | .2122 | **2.79 HP** | .211 |
| Frozen memory | .1503 | .1874 | .2197 | 3.13 HP | .170 |
| Current context | **.1407** | **.1736** | **.2079** | 2.85 HP | **.104** |
| Frozen memory + context | .1633 | .1936 | .2205 | 2.89 HP | .122 |

The stronger recent-damage baseline is fitted from training outcomes grouped by
only the observable presence/absence of incoming damage and proposed response.
It never reads the hidden attacker strategy. Neural improvement over the broad
average is real, but much of the attack-risk separation is already explained
by this simple measured cue. The old pursuit memory did not improve on the
context-only head in this test.

At first-retreat origins, the context-only risk score was .1344 versus .2662
for the broad average. At first injury or renewed injury, however, it did not
beat the broad average. The overall gain therefore does not establish that the
network learned an opponent's tendency to return.

Calibration also needs work: the context-only head predicted an average 52.6%
chance of further damage over 12 turns, while 40.9% occurred in the held-out
rollouts. The training-only baselines overpredicted too. The unseen longer
return delay changes the event distribution; no calibration correction was
fitted on the test set. Death was rare (4.0% over twelve turns), so death-risk
quality is not established by apparently small error alone.

## Better forecasts did not yet mean better choices

As separate diagnostics, select the response predicted to minimize injury or
maximize nourishment; do not combine them with arbitrary cost weights. Compare
against the empirically best response averaged over the four held-out draws.
This is a noisy diagnostic, not a claim of knowing each action's true value.

For injury-only choice, the fixed training-average response incurred 0.50 HP
of excess injury on average; the context-only head incurred 0.54 HP, frozen
memory 0.88 HP, and memory-plus-context 0.79 HP. Thus we do **not** have evidence
of improved action selection. Feeding-only choice was closer: context-only
lost .0030 nourishment versus .0032 for the training-average response.

No learned forecast has been installed into the live controller. Its existing
neural pursuit weights remain unchanged; new weight files stay outside Git.
The checkpoint hashes and read-only inference round-trip were verified. All
217 regression tests pass, including fork isolation, input privacy and learning checks.

## What this suggests next

Train a **copy** of perception and memory on the outcome histories, including
hunger, food state and incoming damage, rather than freezing a representation
trained for pursuit. Keep the existing pursuit actor fixed as a control; changing
its latent representation without retraining its action scorer could break it.
Compare the new memory against the same current-context and simple-cue baselines.

Broaden return timings and include longer evaluation horizons, while retaining
new unseen timing patterns for testing. Do not mistake an unknowable initial
retreat for evidence of permanent safety. Some histories genuinely support only
a probability, and the calibrated uncertainty matters.

Before deployment, require both better held-out forecasts and better engine
choices, particularly on recurrent attacks. Architecture depth is not yet the
main demonstrated limitation.

## Reproduce and inspect

```sh
python3 -m experiments.meal_forecast_data --model /path/to/spider.pt --output /new/output
python3 -m experiments.train_meal_forecast /new/output
```

`OutcomeEnsemble` in `experiments/train_meal_forecast.py` loads the three saved
heads and forecasts all responses from 48 current memory values and eight
private context values. It checks the source pursuit model hash, does not
update that memory, and does not choose a command.

Weights, fork experiences and the full report:
`/Users/jeff/Documents/ChatGPT/White Fang/experiments/meal-forecast-forks-v1`.
Compact tracked results: `experiments/results/meal-forecast-frozen-v1.json`.
Pilot files are excluded from these results. No production game files changed
in this experiment; no push or post-experiment commit has been made.
