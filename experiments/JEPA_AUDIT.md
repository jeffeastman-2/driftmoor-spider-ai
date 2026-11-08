# Separating forecast errors from cost design

The experiment freezes both GRU checkpoints and all cost weights. For six
controlled cases per actor, it compares all eight learned forecasts against
actual counterfactual simulations using 32 paired dice streams per action:
12 decision states, 96 action alternatives, 3,072 simulated transitions.

This is an **offline diagnostic instrument**. The authoritative simulator is
never supplied to the actor. Production planning remains private-observation
inference. There is no deeper search, retraining, cost tuning, or deployment
change in this experiment.

## Controls

The cases are healthy/close with an idle opponent, healthy/close with a sensory
teacher opponent, healthy/distant with idle opponent, wounded/close with teacher,
critically wounded with observed blocked-flight feedback, and no contact behind
the authored curtain. Both roles have explicit grounded poses and headings.
The blocked-flight case supplies an already-observed failed-attempt flag, not a
proof that every escape route is impossible. Contact absence is asserted in tests.

Each branch starts from an independent deep copy and the same dice-stream seed.
The candidate uses the usual private observation and neural belief to aim. Its
opponent either waits or takes one fresh sensory-teacher turn. Reference futures
include the deciding actor's next observation plus its own admission/damage
feedback, matching the trained predictor's horizon. Terminal states are included.

The reference ranking minimizes the mean **cost of each sampled outcome**, rather
than the cost of the mean outcome. It uses the same policy prior as the deployed
planner. Pure-cost choices are also recorded to identify prior-driven ties.
A reference-best choice means best under these particular costs, not good dungeon
behavior. Different action branches can consume different random draws.

## What failed in the world model

For a healthy spider already in bite range of an idle opponent:

| Bite outcome | Learned forecast | Reference |
|---|---:|---:|
| Next own HP fraction | .586 | 1.000 |
| Admission | .547 | 1.000 |
| Mean damage | .216 | 3.094 |

The predictor imagines a large injury and discounts a legal bite. Consequently
the learned-cost planner advances. **With actual futures and identical costs,
it attacks.** This directly isolates a forecast-ranking failure from cost design.
The reference damage is a 32-stream sample mean, not a dice expectation formula.

Replacing one predicted component at a time with reference outcomes is another
offline probe. Correcting only next HP changes the healthy spider facing an
active opponent from advance to the reference choice, retreat. It also changes
the unseen-opponent case from advance to the reference choice, wait. Correcting
HP alone does not fix the idle-opponent bite case: attack feedback remains wrong.
For Gil's distant-target case, replacing any single tested component does not
repair the ranking. Errors interact; one field or a larger attack reward is not
a demonstrated general fix.

The learned and reference choices match in 7 of 12 cases. This is a small selected
set of cold-memory states, not a performance estimate. Some choice differences
are small prior-driven ties; numeric reference regret is recorded per state.

## What remains wrong or incomplete in the objective

With actual forecasts, both healthy actors prefer attacking an idle target in
reach. The cost therefore does not inherently forbid attacks. Against an active
opponent the current weights can favor withdrawal, which may be reasonable for
survival but does not establish successful hunting or encounter completion.

The range term only applies when both current and future observations have a
contact. Losing contact removes that term. Thus an unproductive movement that
loses the target can tie waiting, and the policy prior breaks ties. We have no
explicit objective for retaining a useful target belief, catching prey, defending
the vault, escaping to safety, or completing the encounter. A one-step horizon
also cannot value an approach that enables a later attack. Longer search with
these inaccurate forecasts would not solve those problems by itself.

## Next implementation target

First learn action-conditioned **HP change** rather than reconstructing absolute
HP, anchored on known current own HP. Give injury and admission explicit training
losses and evaluate them separately from average observation MSE. Collect balanced
candidate actions in varied private histories, and measure counterfactual ranking
on fresh fixtures. No oracle state should enter deployed inference.

Separately specify an actor's encounter objective and test cost preferences with
reference futures before evaluating them through learned forecasts. Retaining
contact must not simply reward staring at the enemy; progress and successful
outcomes need deliberate definitions. Neurochemical modulation can remain bounded
around those objectives. These are next steps, not changes silently made by this
audit.

## Reproduce and review

```sh
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/driftmoor-mpl python3 -m experiments.audit_jepa_planning \
  --driftmoor-root /Users/jeff/Python/Driftmoor-spider-sim \
  --checkpoints "/Users/jeff/Documents/ChatGPT/White Fang/experiments/withdrawal-cornered-v6" \
  --output /private/tmp/jepa-audit/report.json
```

Seeds: fixtures 230000–230005, dice streams 240000–240031. Use a new output file.
`results/jepa-audit-v2.json` contains the action-level costs and replacement probes.
The full external copy is
`/Users/jeff/Documents/ChatGPT/White Fang/experiments/jepa-audit-v2/report.json`.

126 tests pass, including branch isolation, reproducible RNG, feedback/horizon
alignment, absent contact, and separate forecast/cost reporting. The game checkout
is unchanged. All follow-up work remains uncommitted for review.
