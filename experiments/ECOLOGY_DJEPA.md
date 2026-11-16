# Three-species dJEPA experiment

Status: trained experimental controllers, **not promoted over the scripted baseline**.
No language-model calls. Weights, datasets and encounter traces stay outside git.

## Architecture and privacy

Rats, giant spiders and Shadowbeasts share an architecture, not weights. Rats
share rat weights but each individual owns its GRU state. The other species
have separate checkpoints. Crickets and lizards retain the existing scripts.

Each controller receives its own body/health/movement state, last motor result,
private sensed contacts, and eight ideal local floor-clearance measurements
out to ten feet. No opponent IDs, species labels, hidden HP, global routes or
future engine results enter the controller. Clearance is an added simulated
sensor using native point collision, not a full-body navigation system.
Observer annotations may identify creatures by screen name.

Each contact passes through a 12→24→24 MLP. Mean and maximum pooling give a
fixed-size summary of any number of contacts. Combined with 24 self-state
values, the encoder is 72→64→32; the GRU retains 48 numbers. Candidate features
have 18 numbers. The action proposal is 66→32→1 per candidate; the outcome
predictor is 66→64→7. All these weights train together, separately per species.

Candidates are wait, eight headings at either five or ten feet, and one attack
per visible contact. This is a new ecology action contract, not the earlier
eight-action single-opponent controller. Native engine movement and combat
still adjudicate actions, including attack admission and dice rolls.

Forecasts cover injury, actual movement, attack admission, damage conditional
on admission (including misses), and future sensed-contact presence/range.
Expected attack damage is permission probability × conditional damage.
The planner evaluates all current candidates for one step; it does not search
several future turns. Proposal-only mode omits cost scoring as a diagnostic.

The configurator and HP-based costs are authored. Species-specific cautious
teachers initialize action proposals through imitation; engine transitions
supply outcome supervision. This is **JEPA-inspired numeric prediction**, not
joint-embedding JEPA training, reinforcement learning, or learned goal discovery.

For the subsequent balanced curriculum and rat navigation experiment, see
[ECOLOGY_CURRICULUM.md](ECOLOGY_CURRICULUM.md). The tables below retain the
first run for comparison; current training defaults and preview weights differ.

## First results

One optimizer seed per species; 16 training and four validation episodes each,
12 rounds maximum with randomized positions/health in four familiar rooms.
Closed-loop comparisons use eight separate engine seeds, 48 rounds each, with
the same authored starting positions. These test dice variation in this map,
not generalization across landscapes. All three neural species act together,
so encounter changes cannot establish the isolated quality of one species.

| Mode | Rat wall blocks / moves | Spider admitted / attempted attacks | Shadowbeast wall blocks / moves |
|---|---:|---:|---:|
| Existing scripts | 43 / 439 | 82 / 82 | 6 / 139 |
| Cautious teacher | 18 / 324 | 20 / 20 | 38 / 342 |
| Neural with forecast scoring | 508 / 580 | 8 / 384 | 316 / 384 |
| Neural proposal only | 508 / 580 | 22 / 22 | 317 / 384 |

The neural spider's action proposal is usable in this limited comparison, but
forecast scoring overrides it badly. Learned rat and Shadowbeast navigation
fails even without cost scoring. Apparent survival gains reflect avoided
encounters and wall sticking, not demonstrated tactical improvement.

Admission validation is especially weak: the spider validation set contains
**no admitted attacks**, and all three models classify every validation attack
as rejected at the 0.5 threshold. Thus the spider's reported 100% admission
accuracy is an all-reject baseline, not success. See the retained full report
in `results/ecology-djepa-v1.json`. No model is promoted.

Next: collect balanced close/far attacks and wall-escape transitions, measure
positive and negative recall plus navigation completion, then validate each
species against fixed opponents before another joint encounter comparison.
Do not fix this by silently giving agents a global path oracle or hiding
unreachable attacks with an authoritative engine mask.

## Reproduce / inspect

Train the current balanced curriculum into a new directory (existing output is refused):

```sh
python3 -m experiments.train_ecology_djepa --output /tmp/ecology-djepa-new
```

The first run's weights and data are at
`/Users/jeff/Documents/ChatGPT/White Fang/experiments/ecology-djepa-v1`.
To inspect them in a separate, initially paused viewer:

```sh
python3 -m experiments.watch_level2 --port 8082 \
  --output /tmp/level2-neural-preview \
  --models '/Users/jeff/Documents/ChatGPT/White Fang/experiments/ecology-djepa-v1'
```

Add `--selection proposal` to diagnose the learned action chooser without
forecast scoring. Annotations distinguish neural decisions and show forecasts
beside actual engine results. Other species remain labeled scripted.
Omitting `--models` retains the scripted viewer. Neural restart resets memories;
resume validates checkpoint hashes, selection mode and deterministic replay.
The existing port-8081 scripted episode has not been reset or replaced.

No feeding duration, wrapping, breeding, nutritional depletion or accelerated
waiting was added. Those lifecycle mechanics remain separate future work.
