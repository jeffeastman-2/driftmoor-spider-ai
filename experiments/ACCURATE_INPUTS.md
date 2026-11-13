# Accurate body-and-bite inputs

Previous learning and aggression work was committed as **264dbbe** first.
This experiment establishes a new input contract and tests one narrow skill:
**can a small model learn whether the engine will permit a bite?** It does not
replace the full recurrent creature controller or claim better autonomous combat.

## Plain-English finding

The revised measurements contain enough information to reconstruct the engine's
mouth-to-body distance exactly. A small model learns about 94% of bite decisions
correctly on fresh examples. A small model trained directly on the old measurements
also learns about 93% correctly. Accurate inputs help reject more illegal bites,
but this is not a dramatic difference, and balanced accuracy is almost identical.

That matters: we cannot blame the old measurements alone for the full controller's
failure. Its compressed representation, training objective or experience coverage
remain likely bottlenecks. This experiment does not isolate which one. It does
show that a bite-admission task is learnable without a hand-coded reach gate.

## Revised 57-number contract

`accurate_senses.py` declares every input name in `FIELDS`, schema
`accurate-body-bite-v1`. Distances/body dimensions are scaled by 10 ft, movement
speed by 60 ft/turn. Masks distinguish absent sensing from zero measurements.

| Group | Count | Content |
|---|---:|---|
| Own body, pose and health | 11 | width/depth/height, floor-relative height, HP fraction, sin/cos roll/pitch/heading |
| Own bite | 8 | mouth origin xyz, bite direction xyz, reach, cosine of half-angle |
| Own movement state | 3 | speed, reaction availability, latest blocked-action flag |
| Detection | 5 | vision available, visible contact, web available, vibration, web evidence |
| Visible opponent geometry | 15 | exact relative centre xyz, body dimensions, three relative body basis vectors |
| Web vibration | 3 | coarse range and bearing sin/cos; no shape/elevation |
| Previous action | 8 | one-hot motor action |
| Recent result | 4 | admission, hit, damage, own injury |

Relative geometry is in the spider's body axes: forward, left, up. Visible box
orientation is represented by a rotation basis, avoiding angle wrapping. The
opponent's centre and oriented dimensions allow reconstruction of its support-face
anchor without revealing hidden identity or statistics. Own mouth geometry is
read from the actual socket, not the teacher's approximate 45-degree proxy.
The engine's existing reach tolerance is learned from outcomes, not supplied as
a legality margin. No closest-point distance, facing verdict or permission flag
for the current bite is provided.

Visibility must pass both the fixture's obstruction/sensor gate and the native
engine's vision gate. Unseen target geometry is all zeros. Web signals retain
the existing coarse measurements. This is an **ideal exact box-fitting sensor**,
not evidence that a camera or biological spider can observe every body surface.
No target HP, armor, identity, weapon, intention or out-of-sight body pose is
included. Neurochemical state remains outside prediction for goal adjustment;
hunger/nutritional-depletion mechanics are not added.

The old 41-vector remains untouched for existing controller/checkpoint use.
The new contract intentionally removes absolute own XY and changes geometry
encoding as well as precision. This bundled comparison does not isolate each
added feature's individual value.

## Engine-generated corpus

4,096 training, 1,024 validation, 2,048 test states from separate seeds. Vary
relative distance/bearing, own heading/pitch/roll, both floor-relative elevations,
opponent box dimensions/orientation, visibility and own health. Positions must
lie in authored scene zones. The simulator remains simplified and does not
validate full body occupancy for every sampled pose. This is the same cavern,
not a new-map/species generalization test.

A bite is executed at the explicitly supplied **current pose**, without an
automatic aim correction. The label is engine `admitted`, not hit or damage.
Independent combat rolls yield both admitted hits and admitted misses. Neither
outcome enters pre-action inputs. Native engine statuses are recorded only for
evaluation. If a future planner proposes turning before biting, it must encode
that candidate pose consistently; this classifier is not trained on an implicit
turn-and-bite decoder.

Both input sets train the same direct MLP shape (input →64→32→1), from scratch,
three optimizer seeds 42/43/44, 120 epochs, Adam .001. No frozen encoder/GRU is
involved. Class-weighted binary loss addresses the many more illegal examples;
validation selects checkpoints. The .5 score threshold is fixed. Scores are
not automatically calibrated probabilities because of the class-weighted loss.

## Held-out results, averaged over three optimizer seeds

Test data: 218 permitted bites, 1,830 rejected bites, including 1,609 range
rejections, 84 attack-arc rejections and 137 unseen-target rejections. Legal
outcomes include 103 hits and 115 misses.

| Measure | Old coarse inputs, direct MLP | Revised accurate inputs, direct MLP |
|---|---:|---:|
| Overall correct decisions | 92.7% | 93.9% |
| Recognized permitted bites | 96.5% | 94.8% |
| Recognized rejected bites | 92.3% | 93.8% |
| Balanced accuracy | 94.4% | 94.3% |
| Mean squared score error (Brier) | .0533 | .0461 |

The majority-class baseline would already score 89.4% overall by rejecting every
bite; it has only 50% balanced accuracy and recognizes no legal bites. These
models do learn beyond that baseline. Nevertheless, accurate models falsely
accept 102–130 rejected bites per seed. They are not ready to act as a reliable
hard permission check. Do not compare this percentage directly with earlier
43-output world-model MAE or interpret it as encounter success.

The spider's own body and bite capabilities are fixed in this single-creature
corpus, so their usefulness across different creatures is not tested yet.
The new inputs make them explicit rather than relying on species-specific weights.

## What remains to do

Keep this narrow task as a diagnostic before more aggression or architecture
sweeps. Check difficult reach/arc boundaries, then test a dedicated admission head
or direct sensory access in the world model, preserving useful policy/memory
semantics. A comparison with frozen latent inputs is still needed to separate a
representation bottleneck from the full predictor's objective/exposure problems.
Do not claim that adding 16 numbers by itself fixed the controller.

## Reproduce and review

```sh
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/driftmoor-mpl python3 -m experiments.learn_bite_geometry \
  --driftmoor-root /Users/jeff/Python/Driftmoor-spider-sim \
  --output /private/tmp/accurate-bite-new
```

Use a new directory. Data and standalone classifier weights:
`/Users/jeff/Documents/ChatGPT/White Fang/experiments/accurate-bite-v2`.
Tracked summary: `results/accurate-bite-v2.json`. These classifiers have their
own task/checkpoint format and are not full `Agent` checkpoints. The earlier v1
used a repeated combat roll per split; admission labels, input states and model
results were unchanged by fixing the roll independence. v2 retains that fix and
reports hits/misses separately.

150 tests pass, including hidden-stat independence, unseen-geometry masking,
actual admission feedback and exact reconstruction of native mouth-to-box
distance under nontrivial rotations. No game checkout changes or model promotion.
Model checkpoints and generated datasets remain outside the repository.
