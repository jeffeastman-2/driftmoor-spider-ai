# Bite advice in encounter decisions

The separate body-geometry classifier now supplies a soft estimate of whether a proposed bite can reach its target. In this experiment it helped the spider attempt more physically valid bites, but did not solve pursuit of distant prey.

“Admitted” means the attack passed the visibility, reach and aiming checks. It then goes through the native combat resolver: d20 plus attack bonus against Armor Class, followed by damage dice on a hit. Rejected attempts never reach that dice roll. Admission accuracy is therefore not hit accuracy.

## Paired encounter results

Each row uses 128 encounters: eight scenario families with seeds 950000–950015. Opponent policy, movement, model weights and cost settings are identical within each baseline/advice pair.

| Setting | Attack requests | Admitted | Rejected | Spider HP lost | Survived |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ordinary baseline | 0 | 0 | 0 | 65 | 117 |
| Ordinary with advice | 8 | 8 | 0 | 65 | 117 |
| Aggressive baseline | 26 | 10 | 16 | 81 | 117 |
| Aggressive with advice | 29 | 18 | 11 | 81 | 117 |

The ordinary controller's additional eight attempts are legal attacks, not necessarily eight hits. Total damage dealt, including opportunity attacks, increased from 21 to 41 HP. With the aggressive setting it increased from 79 to 117 HP. These are descriptive results from one controller checkpoint, not evidence of general superiority.

## What changed

`bite_advice.py` averages the logits of the three accurate-input classifiers. A positive-slope sigmoid calibration is fitted using only the existing validation split (slope 0.9101403, bias -1.6608474). No classifier or controller weights are retrained.

The adviser receives the 57-number private body/sensory observation. Target geometry is transformed into the candidate bite orientation using only the current observation and proposed pose. It receives no engine legality answer, hidden opponent state or simulated future.

Only the candidate attack's admission forecast is replaced. All movement, damage, injury and other action forecasts, policy priors and goal costs stay fixed. The adviser is a cost input, not a hard veto. The recurrent memory advances once per decision.

## Remaining failures

Six controlled cases were also run with four fresh seeds each (960000–960003), for at most six own turns. The aggressive controller still attempted 24 out-of-range bites against distant idle prey; none were admitted. Improved permission advice alone is insufficient when the unchanged damage forecast still rewards an impossible bite.

Both controllers retained their initial retreat response in the wounded and critically injured controlled cases. Advice enabled some later counterattacks in the blocked critical case, including rejected attempts. Thus the broad comparison's zero rejected ordinary bites does not apply to every controlled fixture.

The experiment uses familiar cavern geometry and one recurrent controller checkpoint. It does not test new landscapes or independently trained controllers, and does not promote this controller to the game default.

## Artifacts and verification

- Runner: `run_bite_advice.py`.
- Compact report: `results/bite-advice-v1.json`.
- Detailed encounter traces: `/Users/jeff/Documents/ChatGPT/White Fang/experiments/bite-advice-v1/`.
- 152 tests passed before the encounter evaluation, including candidate-pose geometry and replacement of only the attack-admission forecast.

## Proposed next experiment

Make predicted bite damage consistent with predicted bite permission. Collect engine outcomes at candidate bite poses, recording permission, hit/miss and actual damage separately. Repeat dice rolls for sampled states so a legal miss is not mistaken for an impossible bite. Split by encounter state before generating repetitions, keeping related rolls out of different data splits.

Train a small damage head on permitted attacks, including their zero-damage misses. Its target is expected damage conditional on permission, not conditional on a hit. Combine it with the calibrated admission estimate:

`expected bite damage = P(permission) * E(damage | permission)`

Do not multiply the existing damage forecast by admission blindly: that forecast was trained on all outcomes and already includes rejected attacks. Use the explicitly conditional head instead. Opponent armor and hidden statistics remain unavailable to inference; uncertainty about them must remain uncertainty rather than an oracle input.

First evaluate permission calibration and unconditional expected damage on held-out states, especially reach/aim boundaries and distant prey. Then compare the existing adviser against this consistent version using fresh paired encounter seeds, the same controller checkpoint, unchanged movement/cost settings, and the same controlled cases. Report legal attempts, rejected attempts, actual hits, damage and survival separately. Include a damage-head-only comparison to distinguish better damage estimation from the permission combination.

Success requires fewer impossible attempts without suppressing useful close attacks or increasing injury. If the spider merely waits or retreats, inspect its advance forecasts next; that would be a separate experiment. No default promotion or architecture expansion is planned in this step.
