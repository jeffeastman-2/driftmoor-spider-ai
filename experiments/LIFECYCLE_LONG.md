# Long headless lifecycle baseline — November 18, 2026

The user revised the planned 10,000-turn run to 9,000 turns. Analysis covers
turns 0–9,000. The stop signal arrived after turn 9,025; the 25 extra completed
turns remain in the raw log but are excluded from every reported result.
The separate live UI episode was not paused, restarted or changed by this run.

## Protocol

Seed 42, two initial spiders, fixed existing pursuit weights, current lifecycle
rules: 1% hunger per turn, 1 HP starvation damage per turn at maximum hunger,
mate-seeking below 10%, return to food at 20%, 20-turn mating cooldown,
two-turn hatching, zone-wide implicit webs, immediate exhausted-carcass removal,
and random one-for-one rat, cricket and Shadowbeast replacement. The lizards
and warden are finite initial food sources. Other creatures remain scripted.
No weights are updated. Turns have no UI sleep or map rendering after setup.

Reproduce with:

```sh
python3 -m experiments.run_lifecycle_long --turns 9000 --seed 42 \
  --cache-geometry --spider-model /path/to/pursuit-imitation-v1/spider.pt \
  --output /path/to/new-output-directory
```

The batch-only optimization caches exact coordinate queries against unchanged
geometry and rejects points outside a polygon's bounding rectangle before its
original ray test. It does not change visibility rules, coordinates, actor
weights or encounter rules. All recorded per-turn population, hunger, food,
event and goal metrics match the uncached reference through turn 1,897.
Random-coordinate and polygon-vertex geometry equivalence tests also pass.

## Results through turn 9,000

- Final population: 42 (21 males, 21 females).
- Peak population: 62, first reached at turn 2,640.
- After turn 1,000: mean 35.91, range 9–62.
- Births: 1,136 individual spiders; deaths: 1,096.
- Starvation deaths: 1,074 (98.0% of deaths).
- Post-initial 1,000-turn block means: 37.30, 38.51, 32.25, 34.59,
  32.57, 32.47, 39.09 and 40.51.

This run supports a broadly maintained average population with sustained,
irregular booms and crashes. It does not show convergence to a constant count,
and the later swings do not simply fade away. This is one fixed seed and an
authored-goal baseline, not evidence of learned ecological adaptation or a
formal demonstration of stationarity. Future learned controllers should be
compared across seeds, food supplies, geometries and starting populations.

## Artifacts

Compact parameters, window statistics and validation details are in
`results/lifecycle-long-v1.json`. Raw per-turn metrics, events, CSV, figure,
source hashes and summary are outside Git at:

`/Users/jeff/Documents/ChatGPT/White Fang/experiments/lifecycle-10k-seed42-fast-v1`

Use `population-through-9000.png` for the figure and
`population-through-9000.csv` for the analyzed data. Slower validation runs
were stopped after comparison and are marked as incomplete validation runs.
