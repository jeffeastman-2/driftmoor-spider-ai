# Balanced encounters and local clearance

The trained controllers remain experimental. The latest combined encounter
still exposes spider navigation and overly passive hunting; it is not promoted
as a replacement for the scripted game behavior.

## What changed

The curriculum now has 64 training and four validation episodes per species,
up to 12 rounds each, with one scripted opponent at a time. Close and distant
starts are stratified separately, along with health, headings, prey versus
danger, three room regions, and starts biased toward sampling-box edges.
A sampling-box edge is not necessarily a dungeon wall: authored geometry may
be angled or connect through it. Wall-escape evaluation checks native measured
forward clearance under five feet, then asks whether the actor moved at least
five feet from its start within six rounds.

Attack-permission loss gives equal influence to admitted and rejected requests.
This balances the loss, not the exact number of samples. Dice misses remain
admitted attacks with zero damage. Proposal and prediction heads still train
together with the encoder and GRU. We vectorize candidate scoring during
training to make the larger corpus practical. The model size and cost formulas
are initially unchanged from the first experiment.

After 300 epochs (best validation-loss checkpoint retained), the small
validation sets contain both attack classes:

| Species | Admitted / rejected examples | Admit recall | Reject recall |
|---|---:|---:|---:|
| Rat | 25 / 4 | 100% | 100% |
| Spider | 8 / 32 | 100% | 93.75% |
| Shadowbeast | 11 / 39 | 63.6% | 100% |

These are small validation samples used to select checkpoints, **not a final
independent estimate of generalization**. The earlier all-reject shortcut is
removed, but this alone does not establish useful behavior.

## Isolate one creature before testing all three together

`evaluate_ecology_isolated.py` compares scripts, cautious teacher, neural
proposal-only and neural predictive scoring. Only the focal actor changes;
opponents always use the existing scripts. Sixteen held-out seeds are tested
for 24 rounds from both focused encounters and the authored dungeon positions.

In the initial retrained models, the focused neural spider survived 16/16
versus 14/16 scripted, with 16/17 admitted attack requests versus 25/25 scripted.
On authored starts it attacked only twice, both rejected, and had 63 blocked
moves. The Shadowbeast survived all 16 authored starts and seldom hit walls,
but only attacked twice. Survival can therefore reflect avoidance, not better
hunting. Rats still behaved poorly. Full results: `results/ecology-isolated-v2.json`.

## One-input rat experiment

The rat candidate vector is optionally expanded from 18 to 19 values: clearance
in the requested movement direction, drawn from its existing eight private
local sensors. This gives the scoring MLP an explicit relationship between an
action and a measurement it already receives. It adds no sensor range, global
map, route planner, collision mask or direct engine-admission answer. All
moves remain candidates. The rat head inputs become 67 numbers (48 memory +
19 candidate); spider and Shadowbeast remain at 66 in this experiment.

We reuse exactly the rat corpus, optimizer seed and 300-epoch budget, keeping
other species checkpoints unchanged. Another 16 paired held-out seeds compare
old and expanded rat inputs against the same scripted opponents:

| Authored-start rat outcome | 18 inputs | 19 inputs | Existing script |
|---|---:|---:|---:|
| Survived | 0/16 | 5/16 | 4/16 |
| Blocked moves / requested moves | 22/50 | 0/164 | 52/282 |
| Longest consecutive blocked moves | 7 | 0 | 8 |

In focused encounters, survival changed from 1/16 to 6/16. Of four starts
confirmed near a wall by native geometry, early escape changed from 0/4 to
3/4. This is useful limited evidence for the action-relative feature, not
proof of robust navigation. One optimizer seed is insufficient to establish
reliability. Reports retain all comparison modes, including proposal-only.

## Joint run remains difficult

The latest eight fresh-seed, 48-round joint tests use the new rat and unchanged
retrained spider/Shadowbeast. The neural spider has 16/17 admitted attacks but
238/367 blocked movement requests; neural rats have 103/390 blocked movements,
and the Shadowbeast attacks only once. Proposal-only rat movement has fewer
blocks (21/321), so outcome scoring can still make choices worse.

Improving rats changes what the spider encounters. Success against fixed
opponents does not automatically carry over to all learned actors together.
The next focused extension is action-relative clearance for predators, followed
by longer escape sequences and hunting/avoidance measurements. No checkpoint
is promoted to the scripted default on these results.

## Goblin activation

The ecology harness now calls Driftmoor's existing activation checker after
actions, supplies only living active residents as potential intruders, and
prepares controllers for dormant residents. A different-faction resident
entering Factor's Office can wake the goblin; a corpse cannot. Defeated goblins
are not revived. Awakening is shown in observer annotations but its identity
is not injected into the actor's private motor feedback. The goblin retains
its scripted controller; it is not a fourth neural species.

Existing live servers retain their already-imported code. The new preview
includes this activation fix; no old episode is reset automatically.

## Run / reproduce

New neural preview: **http://localhost:8082/**, starts paused at round zero.
The scripted encounter on port 8081 is retained. The preview uses experimental
weights from `ecology-djepa-rat-clearance-v3` under the external experiments
directory. It is for inspecting remaining problems, not a claim of improved
overall game AI. Restart resets all individual memories and restores the scene.

```sh
python3 -m experiments.train_ecology_djepa --output /tmp/ecology-curriculum-new
python3 -m experiments.evaluate_ecology_isolated --models /tmp/ecology-curriculum-new
python3 -m experiments.rat_clearance_experiment \
  --source /tmp/ecology-curriculum-new --output /tmp/ecology-rat-clearance-new
```

The curriculum defaults to the original 18 candidate values. For a separate
all-species architecture experiment, `--candidate-clearance` adds the extra
value to every species; that experiment has not been run here. The paired rat
script changes only the rat, preserving other species weights. Old 18-value
checkpoints and new 19-value checkpoints both load with explicit input widths.
Datasets, traces and weights stay outside git. Compact reports are retained in
`experiments/results/`. No feeding, wrapping, breeding or fast-forward waiting
mechanics were added in this pass.
