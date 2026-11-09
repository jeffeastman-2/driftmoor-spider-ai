# Approach instead of a full-stride advance

The completed capacity and engine-cost experiments were committed as **445b2fc**.
This follow-up tests one motor change before retraining any network or changing
costs: shorten advance so it stops near estimated attack range instead of always
using the actor's full Speed.

## Private motor primitive

`approach_action.py` constructs the same local body proxy used by the sensory
teacher, from current measured contact range, bearing, elevation and body extents,
plus own reach/socket and declared orientation. It searches along the declared
heading for the first location estimated to be .25 ft inside reach, within Speed.
If reach cannot be entered horizontally, it chooses a point that reduces estimated
socket/body distance; if no improvement exists it stays put. If body measurements
are unavailable, ordinary exploratory full-stride motion remains available.

This is a hand-authored motor primitive, not a learned action policy or an oracle
query about the target. It receives no authoritative opponent position or hidden
body state. Coarse perception can still yield a bad stopping point. The real
engine continues to enforce boundaries, collision checks, movement budgets,
reactions and weapon legality. Orientation and action timing are unchanged.

Only the evaluated actor's `advance` distance changes. Its opponent keeps the old
full-stride action. Retreat and flee retain their existing distances and costs.
`DualEncounter` has a default distance hook preserving normal full-stride behavior;
`ApproachEncounter` is an explicit opt-in experiment subclass.

The eight neural output labels and dimensions are unchanged, but advance's motor
meaning differs. Old learned future predictions are **not used to rank** this
new action. The checkpoint supplies only private memory/belief and the optional
policy preference. A future learned-controller experiment must generate new
experience with these motor semantics and version that corpus/checkpoint contract
before treating its forecasts as valid. This is not silently promoted to the
normal neural rollout or game engine.

## Unchanged-cost engine test

Use the same offline engine-outcome evaluator as the preceding experiment:
eight sampled dice streams per candidate, one action plus opponent response,
current costs, six own turns, three actual dice seeds per case. Run both costs
with policy preference and costs alone. Same six controlled cases, two actors,
two controllers: 72 episodes. Baseline results are reused from the preceding run
with identical states/seeds, rather than redundantly rerun. Shared authored poses
and repeated development fixtures do not establish novel-world generalization.

## Result: pursuit now works in the distant-idle fixture

| Evaluated actor / controller | Full-stride legal attacks | Approach legal attacks |
|---|---:|---:|
| Spider, engine + prior | 0 | 15 |
| Spider, engine costs only | 0 | 15 |
| Gil, engine + prior | 0 | 15 |
| Gil, engine costs only | 0 | 15 |

In each approach episode the actor advances on turn one and can attack on each
of the following five turns. All 15 attack requests per group are admitted;
none is rejected. Three spider episodes cause 53 HP loss and three Gil episodes
cause 33, with no own injury in this idle-opponent case. Reached enemies are not
necessarily defeated in the six-turn window.

Costs, policy preference, neural weights and horizon were not changed. Removing
the policy preference does not remove the benefit. **The previous pursuit failure
was substantially caused by the available movement choice:** advance overshot,
lost contact, and removed its progress score. A useful short approach creates
immediate measurable progress even under one-step costs.

Reachable idle targets still receive 18 admitted attacks per group. Healthy
active encounters retain their earlier cautious behavior. Wounded and critical
actors retreat in all tested cases with either cost-only or prior control;
costs-only previously chose advance to pass through/past the enemy. Removing that
motor loophole changes its preferred action without changing a goal weight.
No-contact behavior remains waiting or the prior's look-up action.

## Meaning and next step

We now have evidence that the existing costs can choose approach then attack
when given (a) usable motor choices and (b) engine-generated consequences.
This is not evidence that the neural predictor has learned those consequences.
It narrows the next experiment: collect focused and ordinary experience with
this action primitive, retrain its world model, and measure the gap to the
engine-outcome control. Retain the shallow predictor-only baseline; optional
deeper models need not be promoted. More search depth or cost tuning is not
yet necessary to solve this particular fixture.

## Reproduce and review

```sh
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/driftmoor-mpl python3 -m experiments.run_approach_control \
  --driftmoor-root /Users/jeff/Python/Driftmoor-spider-sim \
  --checkpoints "/Users/jeff/Documents/ChatGPT/White Fang/experiments/curriculum-learning-v1" \
  --output /private/tmp/approach-control-new
```

Use a new directory. Same actual seeds 610000–610002 and paired candidate dice
formula as `ENGINE_COST_CONTROL.md`. Full traces and sampled costs:
`/Users/jeff/Documents/ChatGPT/White Fang/experiments/approach-control-v1`.
Tracked report `results/approach-control-v1.json` includes matching baseline groups.

142 tests pass, including unchanged opponent movement, original retreat/flee,
private-observation stopping, retained contact, a subsequent legal bite, and
selection/execution by the unchanged-cost evaluator. Game checkout unchanged.
These follow-up changes remain uncommitted for review. Nothing was pushed.
