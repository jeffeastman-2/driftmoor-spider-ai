# Meal defense comparison — November 17, 2026

The experiment compares three **authored** responses around unchanged learned
pursuit weights. It does not train the predictor or actor and does not change
the live controller.

- **Pursue:** keep fighting/chasing the identified attacker (current behavior).
- **Guard:** return to food when visible opponent motion is outward and beyond
  six feet. Correct for the spider's actual own movement before interpreting
  relative contact motion. Do not read the attacker's hidden goal.
- **Resume:** return to food after a successful damaging retaliation, provided
  no new incoming damage has arrived. This may resume before retreat is observed.

## Controlled experiment

4,608 episodes: 1,536 paired conditions, each evaluated by all three policies.
Thirty-two seeds cross two hunger levels (0.4/0.9), two initial spider health
levels (13/26 HP), two attacker durability levels (12/24 HP), two authored rooms
(Fungal Stores/Freight Guardroom), and three attacker behaviors. Weapon damage
and accuracy stay unchanged; higher attacker HP means durability, not stronger
attacks. Full and half-health cases use the same maximum spider HP (26).

A spider has begun wrapping a standardized 1.2-unit food portion; unrelated
residents are excluded. It is not an enlarged living rat. Engine visibility,
web movement, collision, attack admission and dice remain active. Episodes end
at fullness, death or 72 rounds. Reported survival means survival to that
endpoint, not guaranteed long-term survival. Starting seeds are paired, but
subsequent RNG consumption can diverge when policies choose different actions.

| Policy | Survived to endpoint | Reached fullness | Mean damage taken |
|---|---:|---:|---:|
| Pursue | 86.7% | 58.7% | 7.53 HP |
| Guard | 75.1% | 56.6% | 8.69 HP |
| Resume | 70.8% | 52.0% | 9.60 HP |

Equal weighting of these scripted attacker types is an experimental design,
not a claim about how often each occurs in Driftmoor. There is no universal
winner based on meal speed alone.

## What changes the answer

**Current Shadowbeast rule:** it retreats below 35% HP, rather than after any
hit. Pursue and guard had exactly the same survival (73.4%) and fullness (39.8%)
across the same 512 conditions. For the 204 cases in which both became full,
guard finished 5.34 turns sooner on average, without extra damage. Returning
after just one successful hit was worse: 60.5% survival and 25.8% fullness.

**Flees after its first hit and stays away:** guard and resume matched pursuit's
survival (96.9%) and fullness (75.0%), finishing about 13 turns sooner in the
384 paired successes. Continued pursuit buys little in this constructed case.

**Retreats for three turns, then returns:** pursuit survived 89.8%; guard and
resume survived 55.1%. Fullness was 61.3% versus 55.1%. Early return to food can
invite repeated interruptions. This returner intentionally re-engages despite
injury; it is a stress condition, not the default game behavior.

The overall guard-minus-pursuit fullness difference was -2.08 percentage points;
a seed-cluster bootstrap interval was -3.78 to -0.39 points. These intervals
measure dice-seed variation within this experiment, not new-world generalization.
Timing comparisons among paired successes avoid comparing a policy's lucky
survivors with another policy's entire set of successes.

Many alive-but-unfed cases expose the existing critical-health behavior: a
wounded spider can withdraw and fail to return to food during the horizon.
This experiment does not establish a solution to that problem.

## Ordinary ecology check

A further 24 runs used eight different seeds (74000–74007), the normal initial
population, both spiders, random prey replacements and 120 rounds, with no
prepared meal. All modes had the same aggregate outcomes: both spiders reached
fullness in each run, 15 of 16 spiders survived through the horizon, and there
were eight accepted courtships per mode. Guard/resume eliminated the small
amount of defense chasing (nine chase-turns across the eight pursuit runs),
but did not improve survival or damage in this small check. This is a sanity
check, not enough ecology coverage to select a controller.

## Implication for learning

Do not teach a universal 'stop chasing' or 'always chase' rule. The useful
forecast is whether a sensed retreat will remain safe or lead to another attack,
combined with likely injury, nourishment and time away from the meal. Learning
should use sensory histories and action outcomes; the attacker's experimental
behavior label belongs in evaluation metadata, not the creature's observation.

The collector saves private observations, chosen commands, engine results and
next observations outside Git. Run the alignment step before training: the
raw post-round observation omits the next enemy action. Aligned records pair
an action with the next spider decision, including the opponent response.
Terminal/censored steps have no fabricated next observation and retain actual
final HP/hunger outcomes. The aligned dataset contains 129,538 transitions across 4,608 episodes. No
reward scalar or trained model was produced here. All 214 regression tests pass.
Turn order remains the harness's sorted actor order; this is not an initiative study.

## Reproduce

```sh
python3 -m experiments.compare_meal_defense --model /path/to/spider.pt --output /new/output --seeds 32 --horizon 72
python3 -m experiments.analyze_meal_defense /new/output
python3 -m experiments.align_meal_experience /new/output
python3 -m experiments.validate_meal_defense --model /path/to/spider.pt --output /another/new/output
```

Compact results: `experiments/results/meal-defense-comparison-v1.json`.
Raw data: `/Users/jeff/Documents/ChatGPT/White Fang/experiments/meal-defense-batch-v2`.
Earlier pilot/v1 directories are preliminary or incomplete and are excluded
from the reported final results. Model binaries and large traces remain outside Git.
