# Encoder and memory capacity experiment

The preceding dJEPA work was committed as **85322db** before this experiment.
These follow-up changes are uncommitted for review. Nothing was pushed.

## Matched comparison

All variants start from the original teacher-trained GRU checkpoints and use the
same fixed `curriculum-learning-v1` training/validation sequences, HP-change
initialization, predictor size, costs, optimizer budget, and evaluation fixtures.
The baseline is retrained under the same sequential objective as the joint
variants, not copied from the earlier experiment with a different sampler.

1. **predictor_only:** encoder and GRU frozen; train predictor.
2. **unfrozen:** original `41 → 32` encoder; train encoder, GRU and predictor.
3. **deeper:** `41 → 64 → 32` encoder; train encoder, GRU and predictor.

Every encoder layer has a ReLU. The deeper encoder starts with the original
weights in its first 32 hidden units; an identity second layer selects those
units. The remaining 32 units have nonzero random features with initially zero
output weights, so they can acquire gradients rather than remain permanently
dead. Initial encoder and policy outputs match the original within floating-point
tolerance. Encoder parameter count increases from 1,344 to 4,768. Predictor
architecture and all other module shapes remain unchanged. Checkpoints record
`encoder_depth`, defaulting to 1 for backward compatibility.

Two optimizer seeds (42/43), 60 epochs each, complete recurrent sequences with
padding masks, 32 episodes per batch, inverse-action-frequency training weights,
Adam .001 and gradient clipping 1. No cached detached representations are used
when learning the encoder/GRU. Validation is unweighted and chooses the checkpoint.
No policy/belief preservation objective is added: that would be a separate factor.

Policy and belief **head weights** are fixed, but their inputs change in the joint
variants. Their outputs therefore need not remain stable. We measure this drift;
we do not claim policy or memory semantics are held constant when unfreezing.

## Fresh evaluation

The forecast set contains 32 ordinary +128 focused episodes per actor, generated
once from the same original controller and reused across all variants/seeds.
There are 1,046 spider transitions (46 with injury), and 996 Gil transitions
(74 with injury). These are fresh seeds, not novel terrain families.

Each trained model is evaluated in 8 scenarios ×4 encounter seeds =32 encounters
against the same fixed sensory teacher. Twelve trained models give 384 encounters.
The two optimizer runs share the same encounter seeds; pooled totals describe
those runs, not 64 independent initial fixtures. Action divergence changes dice
consumption. Costs are unchanged.

The offline ranking audit tests six cold-memory fixtures per actor/model, eight
candidate actions and four dice streams each, with the same cost/prior. All actual
outcomes remain evaluator-only. Four streams and six selected fixtures are a
small diagnostic, not a precise ranking benchmark. Teacher-trained models also
face an idle opponent in some audit cases, probing response-distribution mismatch.

## Results averaged over the two optimizer seeds

| Actor / metric | Predictor only | Unfrozen | Deeper |
|---|---:|---:|---:|
| Spider attack-admission MAE | .124 | .096 | .095 |
| Spider injury HP-fraction MAE | .130 | .124 | .121 |
| Spider policy agreement with prior | 100% | 80% | 75% |
| Gil attack-admission MAE | .227 | .196 | .183 |
| Gil injury HP-fraction MAE | .056 | .058 | .058 |
| Gil policy agreement with prior | 100% | 79% | 67% |

Unfreezing improves attack-admission forecasts for both actors. Deepening gives
little extra improvement for spider admission, some for Gil admission, and no
consistent injury improvement across both actors. Encoder capacity alone is not
a demonstrated solution. Aggregate HP metrics, damage predictions and belief
output drift are recorded in `results/capacity-learning-v1.json`.

| Two-run encounter totals (32 encounters/run) | Predictor only | Unfrozen | Deeper |
|---|---:|---:|---:|
| Spider attack requests | 0 | 1 | 0 |
| Spider admitted attacks | 0 | 0 | 0 |
| Spider survived /64 | 58 | 61 | 62 |
| Spider damage received | 35 | 16 | 14 |
| Gil admitted attacks | 0 | 0 | 0 |
| Gil survived /64 | 64 | 64 | 62 |
| Gil damage received | 73 | 237 | 191 |

The only spider request fails the attack arc check. All controllers remain highly
passive. Reduced spider injury is not evidence of successful hunting; Gil suffers
more injury after representation training. Ranking regret does not consistently
improve either: mean spider regret .265/.348/.280, Gil .138/.152/.131 for
predictor-only/unfrozen/deeper. These values depend on the particular costs,
reference action decoder and small sample, not an external optimal policy.

**Conclusion:** the frozen representation was limiting some predictions, but
simply unfreezing or deepening it does not restore useful combat behavior.
Policy agreement drops 20–33 percentage points as its latent inputs change,
and belief outputs drift too. That establishes semantic drift, not proof it is
the sole cause of behavioral failure. Future experiments should test joint
prediction with policy/belief preservation or retraining of those heads, and
separately validate the goal/cost design. Do not increase network depth again
without a more specific failure hypothesis.

## Reproduce

```sh
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/driftmoor-mpl python3 -m experiments.run_capacity_learning \
  --driftmoor-root /Users/jeff/Python/Driftmoor-spider-sim \
  --checkpoints "/Users/jeff/Documents/ChatGPT/White Fang/experiments/withdrawal-cornered-v6" \
  --corpus "/Users/jeff/Documents/ChatGPT/White Fang/experiments/curriculum-learning-v1" \
  --output /private/tmp/capacity-learning-new
```

Use a new directory. Forecast seeds: ordinary 510000–510031, focused
510256–510383. Encounter seeds: 520000–520003. Ranking fixtures: 530000–530005;
dice streams: 540000–540003. Training and validation remain the earlier corpus.
No new evaluation seeds enter optimization or checkpoint selection.

External weights and raw reports:
`/Users/jeff/Documents/ChatGPT/White Fang/experiments/capacity-learning-v1`.
Tracked report adds aggregate summaries and provenance to the raw report.
135 project tests pass, including function-preserving initialization, learning
in added hidden units, joint gradients, padding masks, scope checks, and shallow
and deep checkpoint loading. Game checkout unchanged.
