# Closed-loop cost control with engine-generated futures

Experiment 1 replaces learned forecasts with repeated actual simulation outcomes
while leaving the costs unchanged. It extends the earlier single-decision audit
into six-turn controlled encounters. **This is a privileged offline diagnostic,
not a controller promoted to Driftmoor or a model-training procedure.**

## Controls

Four controllers share the same curriculum-trained, shallow predictor-only
checkpoint, private observation/GRU history and candidate action/aiming decoder:

- Policy argmax.
- Learned one-step planner with its usual policy preference.
- Engine-sampled outcomes with the same costs and policy preference.
- Engine-sampled outcomes with costs alone, removing the policy preference.

Candidate evaluation forks the current world **and the current opponent memory**.
Each action is followed by the specified actual opponent's response, then the
actor's next observation. Idle opponents wait; active opponents use the fixed
sensory teacher. Actual encounters use those same response rules. The reference
controller therefore has privileged opponent-model/state information as an
experimental control, never as an input to a deployed creature network.

For each of eight candidates, eight independently seeded combat streams provide
outcomes. We average the cost of each outcome, not the cost of the average
outcome. The actual world's RNG is neither read for future rolls nor advanced by
hypothetical trials. Sampling streams are shared across actions for paired
comparisons, though actions can consume different numbers of draws. These are
sampled expected costs, **not perfect/exact expected outcomes**. Terminal
transitions remain included. No cost weights or network weights are adjusted.

Six cases ×three actual dice seeds ×two evaluated actors ×four controllers =
144 episodes, each up to six own turns. Three seeds share authored starting poses;
they are dice replicates, not independent maps. Cases: healthy reachable idle
prey; healthy reachable active opponent; distant idle target; wounded active
encounter; critical health with previously observed blocked-flight feedback;
and no contact behind the curtain. Blocked-flight feedback does not prove all
escape routes are blocked. All cases use the existing cavern geometry.

## Primary spider result

Totals below pool three episodes, six turns maximum per episode.

| Case / outcome | Policy | Learned planner | Engine + prior | Engine costs only |
|---|---:|---:|---:|---:|
| Reachable idle: admitted bites | 18 | 0 | 18 | 18 |
| Reachable idle: opponent HP lost | 62 | 0 | 62 | 62 |
| Reachable active: admitted bites | 18 | 0 | 6 | 6 |
| Reachable active: own HP lost | 41 | 0 | 0 | 0 |
| Distant idle: admitted bites | 9 | 0 | 0 | 0 |

The learned planner retreats immediately from reachable idle prey. Both engine
controls bite immediately and repeatedly, without rejected bite requests.
Removing the policy preference does not change this success. This demonstrates
that **forecast errors suppress valid attacks; the existing costs do not
inherently forbid biting**. In the first idle sample, bite cost is -.109 versus
0 for waiting, before the policy preference.

Against an active opponent, both engine controls retreat first, then sometimes
bite later. Wounded/critical actors with the policy preference retreat and
survive all three episodes. With costs alone they instead initially advance
through/past the opponent and still survive: the discrete movement/visibility
mechanics make this a cheap alternative. It should not be interpreted as a
learned retreat motive or a general survival policy.

## What accurate futures still do not solve

The engine controls do not pursue the distant idle target. The prior version
faces it; costs-only waits. In that starting state, waiting, advancing, facing,
and looking all have zero raw cost; attacking is rejected and costs .5.
Advancing can lose contact, eliminating the distance term, while full-speed
movement can overshoot rather than stop in bite range. A one-step objective
also cannot credit a move for enabling a later bite.

Therefore this remaining failure is in the **planning specification**—goals,
horizon and candidate movement semantics—not a forecast problem alone. This
experiment does not isolate which of those three is the dominant cause. Do not
respond merely by enlarging the network or increasing an attack reward until
failed attacks become attractive.

For Gil, engine controls likewise produce 18 admitted attacks against a reachable
idle target, versus three from the learned planner. Against an active opponent,
engine+prior produces one admitted attack with four HP lost; costs-only produces
none. Both fail to pursue distant idle prey. The full per-case report includes
first actions, damage, survival and rejection counts for both actors.

## Conclusion and next decision

We can recover valid immediate biting without changing costs when we supply
engine-generated futures. Prediction accuracy remains worth improving. But a
successful hunter also needs purposeful pursuit, which these same costs and
one-step candidate actions fail to produce even with authoritative outcomes.

Before more capacity experiments, define and test explicit behavioral contracts:
healthy reachable prey should be worth biting; a short approach should set up
an attack; wounded actors should find safe separation; losing contact should not
silently count as progress. Test those contracts with this offline control,
including movement that stops at a useful distance and, separately, a short
lookahead. Then use nonprivileged predicted futures to measure the remaining
gap. No such cost/action/horizon changes were made in this experiment.

## Reproduce and review

```sh
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/driftmoor-mpl python3 -m experiments.run_engine_cost_control \
  --driftmoor-root /Users/jeff/Python/Driftmoor-spider-sim \
  --checkpoints "/Users/jeff/Documents/ChatGPT/White Fang/experiments/curriculum-learning-v1" \
  --output /private/tmp/engine-cost-control-new
```

Use a new directory. Actual seeds: 610000–610002. Counterfactual seeds:
`630000 + (actual_seed-610000)*1000 + own_turn*8 + draw`. Data and detailed
cost/action traces live in
`/Users/jeff/Documents/ChatGPT/White Fang/experiments/engine-cost-control-v1`.
Tracked summary: `results/engine-cost-control-v1.json`.

138 project tests pass. New checks cover fork/RNG/opponent isolation, repeatable
sample costs, legal idle-target bites and actual execution of the reference
choice. Game checkout and costs unchanged. Follow-up changes remain uncommitted
for review alongside the optional capacity experiment.
