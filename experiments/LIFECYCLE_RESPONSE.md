# Courtship response experiment

Baseline checkpoint: `56cd6c6`. New response behavior remains a change to the
**authored teacher**, not a newly trained neural controller. Pursuit weights,
MLP/GRU architecture, mating thresholds, cooldown and food rules are unchanged.
The running viewer has not been restarted or upgraded.

## Behavior

A visible partner currently engaged in courtship provides a `courtship_signal`
on its visual contact. The existing web vibration is also an invitation. A
spider below 20% hunger, at least 35% HP and off cooldown can respond without
first becoming less than 10% hungry. The 10% threshold still starts independent
mate seeking.

The responder uses the existing `approach_mate` and `court` goals. It retains a
currently sensed, receptive partner, preferring visual measurements over a
duplicate web contact. Losing contact or readiness clears that commitment.
Defense and critical-health behavior keep their priority. No remote partner
location or hidden goal is exposed. Existing engine admission still decides
whether courtship is physically possible and can succeed.

## Matched-seed results

Reran the same twelve seeds for 1,000 turns with identical engine settings and
fixed pursuit weights. These are diagnostic regression runs, not independent
held-out evidence of generalization: the failures motivated the change.
After behavior diverges, random-event consumption and later encounters can also
diverge, so final counts do not isolate a single causal pathway.

| Outcome | Baseline | Response teacher |
|---|---:|---:|
| Runs producing any egg sac | 4 / 12 | 5 / 12 |
| Populations alive at turn 1,000 | 4 / 12 | 4 / 12 |
| Total egg sacs across runs | 241 | 294 |

The clearest correction is seed 97:

- Turn 84: female begins courting; male is receptive but has not begun seeking.
- Turn 85: male responds, female completes courtship, an egg sac is laid.
- Turn 87: two offspring hatch.
- Both offspring starve on turn 132; parents eventually also starve, with the
  final death on turn 552. Thus successful courtship did not establish a durable
  population in this run.

Seed 42 mates one turn earlier (72 rather than 73). Other first-mating times
remain unchanged: 115 for seed 23, 82 for seed 79, and 278 for seed 139. All
seven runs that still never reproduce follow the same trajectories as before.
A local courtship response cannot fix absence of an invitation, geographic
separation, or a founder that never becomes sufficiently fed and healthy.

This demonstrates the intended response, **not improved population survival**.
The next coordination issue is communicating an approach before arriving within
court range; the next survival issue is critical-health behavior overriding
feeding. Those changes should be evaluated separately rather than widening the
hunger thresholds to force reproduction.

## Validation and artifacts

Regression tests cover real engine courtship completion by a non-seeking
recipient, health/hunger/cooldown rejection, loss of contact, and absence of
remote visual invitations. Existing feeding, defense, mating, lifecycle and
viewer replay tests remain in the suite.

Raw new experience is outside Git:
`/Users/jeff/Documents/ChatGPT/White Fang/experiments/lifecycle-response-v1`.
The original corpus is preserved. Neither corpus overwrites the other.

Reproduce the comparison:

```sh
python3 -m experiments.compare_lifecycle_responses "$ARTIFACT_ROOT/lifecycle-corpus-v1" "$ARTIFACT_ROOT/lifecycle-response-v1" --output experiments/results/lifecycle-response-v1.json
```

[Per-seed compact results](results/lifecycle-response-v1.json).
New private observations include the perceptible invitation and its response,
providing demonstrations for later goal-model training. No model was retrained
or deployed in this pass.
