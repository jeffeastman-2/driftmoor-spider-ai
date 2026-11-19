# Learning lifecycle intentions: first offline checkpoint

This experiment records the current teacher, learns its goal choices, and then
learns one-step consequences of its demonstrated engine commands. It does not
replace the live controller, change the ecology, or establish improved survival.

## Experience and coverage

Twelve ordinary episodes of 1,000 six-second turns produced 118,715 spider
decisions. No favorable runs were substituted for failures. Eight populations
became extinct; four survived to turn 1,000. Seed 42's long-running population
was therefore not a representative guarantee of robustness.

Whole seeds were assigned before collection:

| Split | Seeds | Decisions | Spider lifetimes |
|---|---|---:|---:|
| Train | 11, 23, 37, 42, 53, 67, 79, 83 | 90,594 | 372 |
| Validation | 97, 109 | 1,266 | 4 |
| Test | 127, 139 | 26,855 | 122 |

The recorder captures each spider's observation **before** the teacher chooses.
It adds only the spider's own sex and remaining mating cooldown. Teacher goal,
command, actual result, post-round body state, IDs and species audit labels are
separate labels/metadata. There are no teacher goal flags in the input vector.
The recorder is tested against an unrecorded simulation to check noninterference.

All nine goals and the predeclared critical coverage cases clear a minimal
training screen: 20 contiguous per-actor segments across at least three training
seeds. These segments remain correlated; this is not a statistical claim of
sufficient coverage. All runs start from the same Level-2 arrangement. Different
seeds vary events and trajectories, not the landscape or initial scenario.

The 111 interrupted-meal sequences include 77 returns to a food-handling goal,
33 deaths and one right-censored sequence. Only seven begin within the first
50 turns; 104 occur later. Return means food handling, not necessarily returning
to the original carcass. Simultaneous interruptions of two spiders are correlated.
An actor killed before its next decision cannot contribute a response decision.

A revealing failure: in seed 11 the female chooses `survive` while at maximum
hunger and with two food contacts. She keeps losing starvation HP until death.
The teacher's low-health priority can conflict with urgent feeding. Imitation
will copy this error unless experience-based improvement is added.

## Learned goal actor

This is a **separate experimental lifecycle model**, above the existing pursuit
motor. The installed pursuit model still has its original 36-input encoder.

```text
Private observation -> 99 hand-selected numeric features
                    -> MLP 99 -> 128 ReLU -> 64 ReLU
                    -> GRU 64 input / 64 hidden state
                    -> MLP 64 -> 64 ReLU -> 9 goal logits
```

The 99 features describe own hunger, HP, sex, cooldown and body; previous action
and result; incoming damage; eight local clearances; contact counts; nearest two
non-conspecific contacts; nearest two conspecific contacts with mating signals;
and nearest two food contacts. Bearings are relative to heading. IDs and absolute
positions are not numeric features. This compact selection loses information
about additional contacts and track continuity; the raw recordings retain it.

Nine outputs: hunt, approach_meal, wrap, feed, seek_mate, approach_mate, court,
defend, survive. A goal is an intention, not an engine command or target choice.
No learned goal executor has been installed yet.

Full per-actor sequences train the GRU, resetting memory at birth, not each turn.
Padded timesteps are excluded from loss. The comparison omits the GRU but uses
the same encoder and actor. Both receive the previous action/result as current
sensory feedback. Training uses inverse-square-root class-frequency weights,
AdamW and a fixed 30-epoch budget; checkpoints use validation macro F1.

| Held-out measure | Current observation only | With GRU |
|---|---:|---:|
| Overall teacher agreement | 91.70% | 94.30% |
| Macro F1 across nine goals | 0.780 | 0.863 |
| Defend recall | 43.5% | 95.7% |
| Seek-mate recall | 78.2% | 94.5% |
| Court recall | 63.7% | 81.2% |
| Court F1 | 0.652 | 0.628 |

Memory helps, but courtship precision falls; improved recall alone is not a win
for every goal. Only one court decision occurs in validation, and both validation
populations die out. These are weak model-selection conditions for reproduction.
Test decisions are dominated by the thriving seed 139; per-seed results are
preserved. GRU agreement is 90.9% on seed 127 and 94.4% on seed 139.
There is one training initialization per model, so these are exploratory results,
not uncertainty estimates or a robust architecture ranking.

## Frozen action-conditioned predictor

After actor training, freeze encoder, GRU and actor. A separate predictor receives
the 64-value context plus 24 command descriptors, then uses 128 ReLU units to
predict 66 deltas: 64 next-context values, own HP fraction and own hunger.
The action descriptors include command type, movement direction/distance and
privately observed target measurements. Compare it with a state-only predictor
and a no-change baseline. Predictors train on demonstrated commands only.

There is no next-context target at an actor's final recorded decision, whether
it died or recording ended. No zero observation is invented. Observed post-round
HP/hunger targets remain valid, including terminal rounds. Actor birth/death
metadata is recorded separately. The frozen latent target cannot collapse by
changing the encoder during predictor fitting.

| Held-out error (lower is better) | No change | State only | State + command |
|---|---:|---:|---:|
| Next-context delta RMSE | 0.4073 | 0.3194 | 0.2820 |
| Hunger-change MAE | 0.00899 | 0.00346 | 0.00119 |
| HP-fraction-change MAE | 0.00275 | 0.00353 | 0.00383 |

Command information improves context and hunger forecasts, but not average HP
prediction. Most turns involve no injury, so injury/no-injury and feeding/no-feeding
subsets are also saved. A further horizon limitation matters: these body targets
run from the observation to the **end of that round**, whereas latent targets
run to the next observation. Opponents can attack before the spider receives its
next observation. Such attacks are not included in this body delta. Indeed, all
1,917 negative HP targets in the test split are one-HP losses, consistent with
starvation. This is not yet a valid next-turn combat-damage forecast. A subsequent
experiment needs aligned next-observation body targets and explicit terminal
labels for actors killed before their next decision.

The normalized combined loss is a training criterion,
not a creature's behavioral cost. Latent dimensions use training-set scales;
HP/hunger deltas use a 0.1 scale. Arbitrary latent units are not physical accuracy.

In JEPA terms this supplies a perception encoder, short-term memory, an actor,
and an action-conditioned latent prediction experiment. It remains one-step,
uses an imitation-shaped frozen representation, and has no learned planner,
trained cost/configurator, counterfactual validation, or demonstrated behavioral
improvement. Good predictions along the teacher's path do not justify imagined
alternative actions yet.

## Next experiment

1. Preserve this offline checkpoint. Prospectively add validation scenarios with
   successful courtship, starvation beside food, and interrupted meals; do not
   tune repeatedly against the current test seeds.
2. Separate goal execution from teacher goal selection. Run the learned selector
   headlessly against the exact teacher on new matched seeds. Measure survival,
   births, food consumption, interrupted-meal return and courtship errors, not
   just per-turn agreement. Keep all engine feasibility and resource rules.
3. Branch engine rollouts from saved private-history contexts, varying plausible
   commands. Record alternative outcomes, especially feeding versus withdrawal
   when critically hungry. Train and test counterfactual forecasts before using
   them to select lower-cost plans. This is the step that could improve upon
   the teacher rather than merely copy it.

## Reproduce and inspect

Run from the spider repository. Artifacts remain outside Git under
`/Users/jeff/Documents/ChatGPT/White Fang/experiments/`:

- `lifecycle-corpus-v1`: manifest, compressed private observations, population
  traces, birth/death metadata and `COVERAGE.md`.
- `lifecycle-imitation-v1`: snapshot/GRU weights, metrics and population/goal plots.
- `lifecycle-predictor-v1`: frozen-context predictor weights and metrics.

```sh
python3 -m experiments.collect_lifecycle_experience --output "$ARTIFACT_ROOT/lifecycle-corpus-v1" --spider-model "$PURSUIT_WEIGHTS" --turns 1000 --workers 3
python3 -m experiments.report_lifecycle_coverage "$ARTIFACT_ROOT/lifecycle-corpus-v1"
python3 -m experiments.train_lifecycle_model "$ARTIFACT_ROOT/lifecycle-corpus-v1" --output "$ARTIFACT_ROOT/lifecycle-imitation-v1" --epochs 30
python3 -m experiments.train_lifecycle_predictor "$ARTIFACT_ROOT/lifecycle-corpus-v1" "$ARTIFACT_ROOT/lifecycle-imitation-v1/gru.pt" --output "$ARTIFACT_ROOT/lifecycle-predictor-v1" --epochs 30
python3 -m experiments.report_lifecycle_learning "$ARTIFACT_ROOT" --summary experiments/results/lifecycle-learning-v1.json
```

Compact results: [lifecycle-learning-v1.json](results/lifecycle-learning-v1.json).
Raw observations and checkpoints are not staged or committed.

## Why the founding pairs failed (follow-up audit)

All eight extinct runs had exactly two spider lifetimes and **zero egg sacs**.
They failed to establish a breeding population; they were not established
populations that subsequently collapsed. The four growing runs first accepted
courtship on turns 115 (seed 23), 73 (42), 82 (79), and 278 (139).

The engine admits mating below 20% hunger, with HP at least 35%, opposite sexes,
no cooldown, visual contact within six feet, and three accepted court actions.
The teacher, however, starts seeking only below 10% hunger and resumes hunting
at 20%. Public receptivity and active mate seeking are therefore different.

| Seed | Evidence from the founders | Bottleneck |
|---|---|---|
| 11 | Female never below 81% hunger; dies turn 31. Male first below 10% on turn 63. | Female dies before a joint opportunity. |
| 37 | Male never below 48% hunger; dies turn 54. Female becomes sated but cannot mate with him. | Male never ready before death. |
| 53 | Both healthy and below 20% on turns 92–99. They start approaching on turns 97–98; ranges fall to 15–20 ft on turn 99. At turn 100 male reaches 20% hunger and resumes hunting. | Search/travel consumes the overlapping window. |
| 67 | Both at about 12% hunger on turn 42 and see each other as receptive, but neither has activated mate seeking. On turn 43 the female is still at 10.5%; both hunt. Later, when both seek, they are separated. | 10% initiation threshold misses an early joint opportunity; later failure to rendezvous. |
| 83 | Both below 20% and healthy during several windows, but no receptive partner is recognized during either spider's mate-seeking decisions. Replay places the male in Broken Lift and female in Beast Pens on turn 78; she also repeatedly encounters blocked moves. | Separation, local-only signalling and ineffective search. |
| 97 | Female courts on turn 84. Male is at 11.5% hunger, publicly receptive but not seeking, and moves away. Female follows at about 10 ft until male reaches 20% on turn 93. | Receptivity does not cause a courtship response; pursuit cannot catch the moving partner. |
| 109 | Male never below 69% hunger; dies turn 46. Female first below 10% on turn 45. | Male never ready before death. |
| 127 | Female never below 44% hunger; dies turn 185. Male repeatedly becomes sated. | Female never ready before death. |

For seed 67, replay also confirms later separation: on turn 86 the male is in
Freight Guardroom and female in Fungal Stores; on turn 124 he is on Receiving
Floor while she remains in Fungal Stores. Implicit webs communicate within a
zone, not across the whole dungeon. No missing-room change was made for this
audit.

The teacher has no separate response to being courted, no commitment to wait
for an approaching receptive partner, and no learned rendezvous plan. The
engine does not require both spiders to issue court commands, but a hunting
partner can walk beyond court range before three accepted actions accumulate.
Mating has no random success roll; these are nutrition, goal coordination and
movement failures.

This does not justify widening the user-selected hunger thresholds by itself.
A better experiment would keep those thresholds and compare a partner-response
or rendezvous behavior against the present teacher, then use those alternative
outcomes for learning. The separate critical-health/feeding conflict also needs
attention for founders that never become ready. No behavior was changed by this
audit, and no additional model tuning used these held-out runs.
