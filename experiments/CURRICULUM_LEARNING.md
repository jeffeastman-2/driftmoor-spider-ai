# dJEPA focused engine curriculum: first experiment

Steps 1–4 are implemented: focused experience collection mixed with ordinary
rollouts, predictor training, separate forecast/ranking measurements, and fresh
closed-loop behavior comparisons. The experiment keeps perception, GRU, policy,
belief head, neurochemical modulation, and cost weights fixed. Teacher-trained
checkpoints are the starting point; the HP-change reparameterization has its own
untrained control. No new goals, nutritional needs, or deeper search were added.

## Experience coverage

`combat_curriculum.py` uses real Driftmoor geometry, sensing, combat, movement,
and condition updates. It defines reachable and distant targets, sideways poses,
ceiling positions, wounds, critical health, boundary positions, and occluded
contact. Starting distance and position vary within the fixtures. For each case,
the first action cycles through all eight actions independently of the case.
Further actions use the planner with 40% random exploration. Focused sequences
last up to four own turns; ordinary encounters last up to eighteen. Death ends
an encounter and its terminal transition is retained. Only private numeric
observations and feedback train the predictor.

Per role, each of two rounds adds 64 ordinary and 128 focused episodes; all prior
round replay is retained. Validation has 16 ordinary +64 focused episodes; test
has 32 ordinary +128 focused. Validation selects checkpoints; no test outcomes
select weights or costs. Case/action coverage and sequential observation horizons
are tested. Frozen parameters are checked for exact equality after training.

The resulting replay has 3,148 spider transitions (106 injuries) and 3,057 Gil
transitions (172 injuries). Spider held-out data has 7 injuries in 543 ordinary
transitions and 43 in 509 focused transitions. Gil has 22/473 and 56/505.
The focused corpus therefore delivers substantially more injury examples.

This is still the Echoing Caverns geometry, with its existing boundaries and
curtain, not a corpus of multiple authored landscapes or different opponents.
Terrain/world generalization is not demonstrated by holding out random seeds
within these same fixture families.

## Forecast results on the same mixed held-out corpus

MAE below is fraction of full HP for injury, absolute admission error, and HP
units for attack damage. These are prediction errors, not probability calibration
or classification accuracy. Admission/damage metrics are conditional on requesting
an attack; injury metrics are conditional on observed own HP loss.

| Actor / metric | Original | HP-persistence control | Curriculum learned |
|---|---:|---:|---:|
| Spider injury HP error | .290 | .146 | .110 |
| Spider attack admission error | .420 | .420 | .111 |
| Spider attack damage error | 1.752 | 1.752 | 1.444 |
| Gil injury HP error | .167 | .079 | .062 |
| Gil attack admission error | .354 | .354 | .225 |
| Gil attack damage error | 1.467 | 1.467 | 1.310 |

There is real improvement on injury events relative to the untrained residual
control, unlike the very small gains in the earlier sparse-injury corpus.
However, aggregate HP error remains worse than assuming no injury: spider
.01289 versus .00695, Gil .01087 versus .00628. Event forecasts must improve
without introducing too many false injury predictions on ordinary turns.

## Does ranking improve?

An offline audit compares all eight action forecasts with actual repeated engine
outcomes in six cold-memory reference fixtures per actor. It uses eight dice
streams per action, the same costs and policy prior, and fresh fixture/dice seeds.
It is an evaluator only; privileged outcomes never enter deployed planning.

| Actor / measure | Original | Residual control | Learned |
|---|---:|---:|---:|
| Spider reference-best choices /6 | 2 | 3 | 4 |
| Spider mean reference cost regret | .113 | .243 | .147 |
| Gil reference-best choices /6 | 4 | 4 | 4 |
| Gil mean reference cost regret | .074 | .074 | .074 |

Spider choice agreement improves, but regret remains worse than the original
model: more matching choices does not imply better ranking overall. The learned
spider still retreats from a reachable idle opponent where the reference cost
prefers attacking. Small fixed fixtures and eight streams are diagnostic only.
The predictor was trained on teacher responses, so idle-opponent audit cases
also probe response-distribution mismatch. No claim of overall planning gain.

## Fresh behavior comparison: 512 encounters

Eight scenarios × eight fresh seeds × two evaluated actors × four controllers.
Each actor faces a fixed sensory teacher. Action divergence changes dice
consumption. The same cost weights are used throughout.

| Actor / result | Policy | Original planner | Residual control | Learned planner |
|---|---:|---:|---:|---:|
| Spider survived /64 | 57 | 59 | 59 | 59 |
| Spider attack requests | 207 | 0 | 0 | 9 |
| Spider admitted attacks | 57 | 0 | 0 | 0 |
| Spider damage received | 35 | 41 | 26 | 26 |
| Gil survived /64 | 62 | 62 | 63 | 63 |
| Gil admitted attacks | 14 | 14 | 12 | 12 |
| Gil damage received | 94 | 373 | 59 | 72 |

The spider requests attacks again, but none is admitted. Six requests fail reach
checks (10-ft sensed range, steep downward bearing); three fail the attack arc
at short range. Despite those failures the predictor scores admission .54–.73
and damage about 2.3–2.9 HP. Diagnostics for every request are saved in the report.
This is evidence of remaining false positives, not successful hunting. Most of
Gil's improvement over the original planner comes from the residual control;
experience training worsens received damage relative to that control.

**The curriculum improves important forecasts, but the behavioral milestone is
not achieved.** Costs were not altered to manufacture an improvement. Next work
should target reach/aiming boundary errors and reliable expected injury estimates,
and consider whether the frozen imitation representation retains the information
those predictions require. Multiple independent training runs and unseen geometry
will be needed before research-level performance claims.

## Reproduce and artifacts

```sh
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/driftmoor-mpl python3 -m experiments.run_curriculum_learning \
  --driftmoor-root /Users/jeff/Python/Driftmoor-spider-sim \
  --checkpoints "/Users/jeff/Documents/ChatGPT/White Fang/experiments/withdrawal-cornered-v6" \
  --output /private/tmp/curriculum-learning-new
```

Use a new directory. Ordinary train seeds: 400000–400063 and 401000–401063;
focused train: 400256–400383 and 401256–401383. Validation: 410000–410015 and
410256–410319. Test: 420000–420031 and 420256–420383. Fresh encounters:
430000–430007. Ranking fixtures: 470000–470005; dice streams: 480000–480007.
No split overlaps. One deterministic optimizer initialization per round,
80 epochs, action-balanced replay; see the runner and report for losses.

Data and model weights remain outside Git in
`/Users/jeff/Documents/ChatGPT/White Fang/experiments/curriculum-learning-v1`.
`diagnostic-report.json` adds source-stratified forecasts and rejected-attack
inspection to the original run report; the future runner records those directly.
`results/curriculum-learning-v1.json` is the tracked report without model files.
132 tests pass. Changes are uncommitted for review; the game checkout is unchanged.
