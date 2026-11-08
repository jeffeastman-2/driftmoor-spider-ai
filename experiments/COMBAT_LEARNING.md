# Learning from combat experience

The experience loop starts from the prior GRU checkpoints, collects real simulation
transitions with exploration, fine-tunes the predictor, and uses the updated
predictor to collect the next round. It is experience-based model learning, not
reward-based self-play. Teacher opponents remain fixed.

## What changes

Encoder, GRU, belief head, action policy, goal modulation and cost weights remain
fixed. Only the existing predictor parameters are optimized. Each actor's
checkpoint is independent. The HP output now predicts **change** added to known
current HP. Its output row starts at zero; all other predictor rows start at their
pretrained values. This reparameterization is tested as a separate untrained
control so it cannot be misreported as learning.

Checkpoint metadata records `prediction_mode`: old checkpoints default to
`absolute`; new ones use `hp_delta`. Parameter shapes and sensory schema remain
unchanged. Both normal forward inference and the planner decode through the same
`predict_futures` method. Saved candidate models are external to Git.

Replay stores each actor's private observation sequences, executed actions,
subsequent private observations after the opponent response, attack feedback,
and terminal flags. No hidden opponent state enters the predictor. GRU states
are reconstructed from complete sequences before caching frozen representations.
Replay batches balance actions; separate losses emphasize HP change, admission
and damage rather than letting them disappear in aggregate observation MSE.

## Episode death and shared templates

Death ends an individual encounter and resets that individual's GRU state for
the next episode. The final transition, including zero own HP when killed,
remains in replay. Training does not reset weights between episodes.

The learned spider checkpoint is a **candidate shared spider template**. Fresh
spider sessions can load those same weights while keeping private GRU memories.
Individual experience can therefore improve later spiders. Candidate updates
should be collected in batches, tested, and then promoted as a version; a single
unlucky death should not immediately mutate all active spiders. Existing live
sessions can keep their checkpoint until an encounter boundary. This experiment
writes candidates only: no automatic promotion into Driftmoor gameplay is enabled.
The initial experiment is specific to this giant-spider fixture, not evidence of
generalization to every spider species or loadout.

## First run

Two rounds × 64 encounters per role, 40% uniformly random action exploration,
80 optimizer epochs per round. Round two includes round one's replay. The final
replay has 2,221 spider transitions and 1,977 Gil transitions, including 5 and 22
terminal transitions respectively (either participant's death ends an episode).
Each role also has 24 fixed validation encounters and 32 held-out experience
episodes. Validation chooses checkpoints; test data never supplies optimizer
batches. Test outcomes are reported after rounds but are not used for selection.

Fresh closed-loop comparison: 64 encounters per role/controller, four controls
(policy, prior planner, untrained residual planner, fine-tuned planner), totaling
512 encounters. Costs are unchanged. All nonpredictor parameters are asserted
bit-for-bit equal to the original checkpoint after training.

| Held-out attack forecast MAE | Spider prior → learned | Gil prior → learned |
|---|---:|---:|
| Admission | .390 → .214 | .258 → .096 |
| Damage, HP units | 1.466 → .937 | 1.274 → .770 |

The HP persistence control is essential. Predicting no HP change already achieves
.00177 spider / .00262 Gil overall HP-fraction MAE. Training achieves .00391 /
.00559, so **overall HP accuracy is worse than that simple control**, though much
better than the old absolute-state predictor. Injuries are rare: only 8 of 543
spider test transitions and 18 of 504 Gil transitions involve own HP loss.
On those events, persistence versus learned MAE is .1202 → .1192 for spider and
.0733 → .0701 for Gil. This is a small improvement, not a reliable injury model.

| Fresh encounter result | Policy | Prior planner | Residual initial | Experience planner |
|---|---:|---:|---:|---:|
| Spider survived /64 | 59 | 59 | 59 | 59 |
| Spider attack requests | 139 | 0 | 0 | 0 |
| Spider HP damage received | 29 | 29 | 29 | 26 |
| Gil survived /64 | 64 | 64 | 63 | 63 |
| Gil admitted attacks | 14 | 14 | 12 | 12 |
| Gil HP damage received | 128 | 274 | 127 | 132 |

**The learning loop works and improves attack forecasting, but better combat
behavior has not been demonstrated.** The spider still does not request attacks.
Most improvements over the original Gil planner come from the residual control,
not from experience fine-tuning. Single-run, sparse-event results do not establish
statistical improvement. Dice consumption diverges when action choices diverge.

## Reproduce

```sh
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/driftmoor-mpl python3 -m experiments.run_combat_learning \
  --driftmoor-root /Users/jeff/Python/Driftmoor-spider-sim \
  --checkpoints "/Users/jeff/Documents/ChatGPT/White Fang/experiments/withdrawal-cornered-v6" \
  --output /private/tmp/combat-learning-new
```

Use a new output directory. Train seeds start at 300000 and 301000; validation
310000–310023; test 320000–320031; fresh encounters 330000–330007. No evaluation
seeds overlap training. Data, candidate weights and raw reports are in
`/Users/jeff/Documents/ChatGPT/White Fang/experiments/combat-learning-v1`.
`results/combat-learning-v1.json` includes the separately calculated event audit.
Future runner output includes those stratified metrics directly.

Next experiment: deliberately gather more close-range legal/illegal attacks and
injury exposures with varied histories, retaining ordinary no-injury experiences
for calibration. Evaluate forecasts by event and action, plus counterfactual
ranking, before increasing planning depth. No costs or policies were tuned to
these test outcomes. The current simulator remains simplified and has no LLM calls.

130 project tests pass. Follow-up changes remain uncommitted for review; game
checkout unchanged.
