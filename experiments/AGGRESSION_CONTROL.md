# Making the healthy spider bolder

## Plain-English finding

We can make the spider more willing to bite. The strongest tested setting makes
it attack reachable prey and still retreat when wounded. But it also attacks
prey it cannot reach. In broader fresh encounters all seven chosen bites were
rejected. It is an experimental option, not a new default or a reliable hunter.
The moderate setting barely changes its behavior.

## What changed

No model weights or movement rules changed. CostWeights has two optional neutral
by default multipliers: healthy injury avoidance and healthy attack reward.
Three settings were declared before the run:

| Setting | Healthy injury multiplier | Healthy damage reward multiplier |
|---|---:|---:|
| baseline | 1 | 1 |
| bold | .5 | 2 |
| hunter | .25 | 4 |

They apply fully at 80% HP or higher and fade linearly to no change at 35% HP.
Below that, the entire cost function equals the old one, including the existing
increase in injury aversion as health declines. Goal modulation from own
conditions remains active. Attack rejection penalty .5, progress, blocked-flight
cost, policy preference and the shorter advance primitive are unchanged.
No hunger or nutritional-depletion mechanics were introduced.

The reward applies to predicted damage, so erroneous damage forecasts can make
out-of-range bites look valuable. Keeping a rejection penalty does not make a
misinformed controller able to recognize actual rejection. The simulator control
knows the sampled consequences and provides a useful diagnostic of this gap.

## Test protocol

Same frozen shallow `approach-learning-v1` spider for every setting. Each setting
runs six controlled cases ×three actual dice seeds ×two controllers (learned and
engine-outcome), six own turns each: 108 controlled episodes total. Engine scoring
averages costs over eight independent candidate dice streams without peeking at
the live future RNG. Weights are forwarded identically to both evaluators.

Then each setting runs 64 learned-controller encounters: eight existing scenarios
×eight new seeds, versus the same sensory teacher. 192 broad encounters total.
No model training, test-based weight selection or automatic default promotion.
Identical initial seeds do not mean identical rolls after actions diverge. These
are development results in the same cavern, not an untouched generalization test.

## Controlled results (three encounters per row)

| Learned-controller case | Baseline admitted bites | Bold | Hunter |
|---|---:|---:|---:|
| Healthy, reachable idle prey | 0 | 0 | 18 |
| Healthy, reachable active opponent | 0 | 0 | 16 |
| Healthy, distant idle prey | 0 | 0 | 0 |

Hunter requests 18 invalid bites in the distant case, repeatedly attacking instead
of approaching. In the active close case it receives 33 HP damage across three
encounters; all survive. Wounded and critical cases start by retreating for every
setting and both evaluators. No-contact cases wait.

The engine-outcome control makes 18 legal bites against reachable idle prey and
15 against distant idle prey for all settings. Against a healthy active opponent,
baseline/bold make six legal bites and initially retreat; hunter makes nine and
initially attacks, receiving 25 HP damage. This is an actual change in risk
preference, not merely altered forecasts. The engine control still retreats in
wounded/critical cases and does not request invalid bites.

## Broad fresh encounter results

| Result /64 encounters | Baseline | Bold | Hunter |
|---|---:|---:|---:|
| Chosen bite attempts | 0 | 0 | 7 |
| Admitted bites | 0 | 0 | 0 |
| Rejected bites | 0 | 0 | 7 |
| Own damage received | 19 | 19 | 19 |
| Survived | 61 | 61 | 61 |

Each controller deals 46 total HP through automatic opportunity attacks. That
must not be presented as successful chosen biting. Hunter improves aggression
in favorable controlled cases but **does not improve general combat here**.
The setting is useful as a research knob, not a candidate default winner.

A diagnostic of the initial reachable-idle state helps explain the threshold:
the model forecasts a bite leaving own HP at .875 and dealing about 3.16 damage,
although the idle opponent cannot injure it. Baseline still prefers retreat;
bold nearly makes attack competitive; hunter makes it preferable. Changing costs
can override this mistaken fear but does not correct the wrong prediction.

## Conclusion

Healthy risk appetite is adjustable independently of wounded caution. A larger
bite reward alone cannot repair attack geometry forecasts: it can amplify their
false positives. Keep the neutral default and retain hunter only as an optional
setting until reach/arc forecasts or a private-sensory feasibility check can
support it. Future changes must be separately tested rather than retroactively
counted as success for this sweep.

## Reproduce

```sh
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/driftmoor-mpl python3 -m experiments.run_aggression_control \
  --driftmoor-root /Users/jeff/Python/Driftmoor-spider-sim \
  --checkpoints "/Users/jeff/Documents/ChatGPT/White Fang/experiments/approach-learning-v1" \
  --output /private/tmp/aggression-control-new
```

Use a new directory. Controlled actual seeds 810000–810002; fresh encounter seeds
820000–820007. Detailed action/cost traces and raw reports:
`/Users/jeff/Documents/ChatGPT/White Fang/experiments/aggression-control-v1`.
Tracked summary: `results/aggression-control-v1.json`.

146 tests pass, including exact preservation of wounded costs and preserved
penalty for invalid, zero-damage attack predictions. Game checkout unchanged.
New settings and experiment remain uncommitted for review, alongside the previous
approach-learning work. Nothing was pushed or promoted into gameplay.
