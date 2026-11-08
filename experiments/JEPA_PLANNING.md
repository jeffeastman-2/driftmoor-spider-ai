# JEPA-like one-step planning

The frozen observation encoder and recurrent memory now feed an optional planner:

**Private perception → memory → eight candidate actions → learned futures → cost → action.**

`jepa_planner.py` implements the planner; `recurrent_agents.py` exposes the
representation without changing checkpoint parameters. The usual argmax policy
remains available. The GRU updates once per actual observation; evaluating eight
hypothetical actions does not change its memory. Each actor has a private session.
No authoritative environment state, collision query, or opponent health reaches
planning. `encounter_evaluation.py` records all candidate forecasts and costs in
its in-memory trace when a planning session is used.

## Cost and time-sensitive goals

For each action, the existing predictor produces 41 next sensory/body values,
attack admission, and damage/20. Its training target is the actor's next
observation after the intervening opponent response. The cost uses predicted
own HP loss, sensed distance change, attack damage and rejection, and blocked
flight feedback. A small negative-log policy prior discourages implausible
candidates. All eight actions are scored deterministically; no sampling or beam
pruning is necessary at this horizon.

Costs change on every decision with current own HP fraction h:

- Injury weight: `8 * (2-h)`. At half HP it is 50% higher than at full HP.
- Separation preference: `1-2*h`. Healthy actors value approach; half-health actors
  stop receiving an approach reward; wounded actors value separation.
- Other default weights: attack damage 1, rejected attack .5, blocked flee 1,
  policy prior .05. Distances are normalized by 60 ft.

These are explicit gameplay preferences, not learned biological laws. Predicted
values are clamped to plausible scoring ranges, but predictions are not calibrated
probabilities. Cost weights are independent of the frozen predictor, so goals can
change without retraining perception or memory.

Driftmoor's `models/condition.py` already defines dopamine, cortisol,
norepinephrine, endorphins, serotonin, acetylcholine, and oxytocin. The simulator
now supplies a detached snapshot of **the deciding actor's own** values and
baseline deviations in private observation metadata. Zero-baseline dimensions
are treated as structurally absent. The neural inputs remain 41 and the frozen
checkpoint schema remains `withdrawal-cornered-v6`: conditions enter cost
computation directly, not the perception encoder or learned future predictor.

The first explicit gameplay mappings are:

- Cortisol deviation `c`: injury multiplier `1 + .5*c`, plus `.5*c` added to
  separation preference (clamped to -1..1).
- Dopamine deviation `d`: attack-damage reward multiplier `1 + .5*d`.
- Both deviations are bounded to -1..1; own authored baseline produces no change.
- The other five dimensions are available but do not yet change costs. There is
  no invented mapping from every chemical to a combat decision.

These are tunable gameplay assumptions, not claims about biological causality.
Existing HP urgency remains active. The planner does not predict future condition
values or choose actions to maximize dopamine itself.

No LLM is needed for this path. Driftmoor's deterministic combat resolver supplies
attacker condition deltas, while its normal session loop applies target injury
deltas. Our standalone combat adapter now applies those updates through
`CharacterCondition.apply_delta`, preserving authored volatility and clamping.
Ordinary attacks and opportunity attacks share this path. Misses apply the
existing stress change; rejected attacks do not. Other LLM-adjudicated narrative
condition updates remain outside this simulation. No new passive decay is added.

## Reproduce

From the spider project:

```sh
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/driftmoor-mpl python3 -m experiments.run_jepa_planner \
  --driftmoor-root /Users/jeff/Python/Driftmoor-spider-sim \
  --checkpoints "/Users/jeff/Documents/ChatGPT/White Fang/experiments/withdrawal-cornered-v6" \
  --output /private/tmp/jepa-planning/report.json
```

Use a new output file. Default comparison is 8 scenarios × 8 seeds × 2 actors ×
2 controllers = 256 encounters. Each evaluated actor faces the same fixed sensory
teacher. Existing GRU weights are reused. Seeds are 135000–135007. The v2 run
follows an exploratory v1 on the same seeds; these are development results,
not an untouched final benchmark. Different choices consume different dice.

## Measured v2 results

| Actor/controller | Survived /64 | HP damage received | Attack requests | Admitted |
|---|---:|---:|---:|---:|
| Spider policy | 59 | 29 | 129 | 24 |
| Spider planner | 62 | 15 | 0 | 0 |
| Gil policy | 64 | 87 | 12 | 12 |
| Gil planner | 64 | 164 | 16 | 12 |

The spider became safer but stopped requesting attacks; its remaining dealt
damage comes from automatic opportunity attacks. Gil took more damage. This is
**not an overall improvement** or evidence that these costs are well chosen.
The selected-action next-HP forecasts have mean absolute error .139 for spider
and .110 for Gil (fractions of full HP). That is material when ranking injury
costs. The audit excludes terminal turns without a next observation and does not
validate unexecuted counterfactuals. Compact report:
`results/jepa-one-step-v2.json`. Full external report:
`/Users/jeff/Documents/ChatGPT/White Fang/experiments/jepa-one-step-v2/report.json`.

123 project tests pass, including private memory, one memory update per real
observation, exhaustive scoring, policy-only equivalence, injury/attack costs,
smooth health-sensitive goals, private condition snapshots, baseline/absence
handling, and deterministic hit/miss/reaction condition updates. The game checkout is unchanged. Changes
remain uncommitted for review.

## Next experiment

Before adding multi-step beam search, improve and validate action-conditioned
health, admission, and movement predictions across deliberately varied actions.
Use fresh evaluation fixtures after development. Separate forecast quality from
cost design with ablations, and measure successful encounter objectives alongside
survival. The current world model predicts sensory observations rather than
JEPA latent targets and averages over teacher opponent responses. It is not yet
a probabilistic opponent model, recurrent imagined rollout, or self-play learner.

## Condition-connected v3 check

`results/jepa-condition-v3.json` repeats the 256 encounters with own condition
goals and deterministic combat updates enabled. The headline survival, damage,
and admitted-attack totals are identical to v2's table above. This establishes no
performance benefit from the current small modulation. Tests independently
exercise nonbaseline conditions, including authored volatility and absence.
The full external report is in
`/Users/jeff/Documents/ChatGPT/White Fang/experiments/jepa-condition-v3/report.json`.
Earlier reports remain historical snapshots. No model retraining or LLM calls
were required. All follow-up changes remain uncommitted for review.

The [controlled forecast/cost audit](JEPA_AUDIT.md) separates prediction errors from objective limitations with repeated actual counterfactual outcomes.

The [experience-learning experiment](COMBAT_LEARNING.md) now updates predictor weights between batches of combat simulations; see its controls and measured limitations.
