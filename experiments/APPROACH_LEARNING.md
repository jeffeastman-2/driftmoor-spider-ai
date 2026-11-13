# Practice with shorter approach movement

Previous work was committed as **4c95de8** before starting this experiment.

## Plain-English result

The spider practiced taking shorter steps, then tried fresh fights using its own
learned expectations. It made four legal bites in 64 encounters, compared with
one before this extra practice. It is still far too cautious. In the simple
walk-closer-then-bite test it still made no attacks, while the simulator-based
benchmark made 15. This is a small behavioral change, not a reliable learned
hunter or a model ready to replace Driftmoor's current spider.

## What was trained

Start with the shallow `curriculum-learning-v1` weights and fine-tune only the
predictor. The existing HP-change output is retained without reinitializing any
row. Encoder, GRU, policy/belief heads and all costs are unchanged and verified
bit-for-bit against the transferred checkpoint after training.

All new experiences use the approach primitive for the evaluated actor; the
opponent retains normal full-stride movement. No old full-stride replay enters
training. Two rounds add 64 ordinary +128 focused episodes per actor each,
with 40% random exploration and balanced forced first actions in focused cases.
The second round reuses the first round's replay. Per round: 80 epochs,
validation checkpoint selection, same separate feedback losses as before.

Separate validation and test sets use the same new motor semantics. Test data
never enters optimizer batches or checkpoint selection. Terminal transitions are
retained. Replay totals: 3,126 spider transitions and 3,073 Gil transitions,
including 27 and 26 terminal transitions respectively (either actor's death).

Episode metadata and new checkpoint schema are `approach-advance-v1`. The old
schema loader refuses these new files by default. Load deliberately with
`load(path, expected_schema="approach-advance-v1")`. Original checkpoint weights
are explicitly transferred into training; their old action forecasts are not
assumed correct. Observation width remains 41 and action count remains eight.

## Forecasts on the same held-out approach corpus

| Metric | Spider before → after | Gil before → after |
|---|---:|---:|
| Attack-admission MAE | .181 → .188 | .136 → .111 |
| Attack-damage MAE, HP units | 1.852 → 1.826 | 1.226 → 1.288 |
| Own HP-fraction error on injury turns | .145 → .114 | .066 → .056 |
| Own HP-fraction error overall | .0198 → .0233 | .0113 → .0114 |

Results are mixed. Injury-event estimates improve, but overall HP accuracy does
not; spider attack-admission estimates slightly worsen. These errors are not
calibrated probabilities or evidence of global prediction improvement.

## Fresh fights using only learned predictions

384 encounters: 8 scenarios ×8 new seeds ×2 evaluated actors ×3 controllers.
Same initial fixtures and fixed teacher opponents; different actions consume
different dice streams. Original policy baseline is retained for context.

| Actor / result, 64 encounters each | Original policy | Transferred planner | Fine-tuned planner |
|---|---:|---:|---:|
| Spider admitted attacks | 52 | 1 | 4 |
| Spider rejected attacks | 99 | 0 | 0 |
| Spider survived | 58 | 63 | 63 |
| Spider damage received | 36 | 6 | 6 |
| Gil admitted attacks | 33 | 14 | 20 |
| Gil rejected attacks | 4 | 0 | 2 |
| Gil survived | 64 | 64 | 64 |
| Gil damage received | 172 | 113 | 165 |

The four spider bites are genuine chosen, admitted attacks, not automatic
opportunity attacks. Most of its 50 dealt HP still comes from reactions (36 HP).
Gil attacks more after practice but also receives more damage. These small,
single-run counts do not establish reliable combat improvement.

## Controlled walk-closer-then-bite test

For each role, compare transferred and fine-tuned planners in six cases ×three
new dice seeds, up to six own turns. A separate engine-outcome control tests
only the distant-idle case with eight sampled outcomes per candidate.

- Spider: no attacks in either reachable-idle or distant-idle cases before or
  after practice. Distant-idle engine control: 15 legal bites.
- Gil: reachable-idle admitted attacks increase from 3 to 17. Against a reachable
  active teacher, admitted attacks increase from 3 to 10, while own damage drops
  from 68 to 25 in those selected cases.
- Gil still cannot handle the distant-idle case: zero admitted attacks, despite
  six requests after practice. Engine control: 15 legal attacks.

Thus the useful motor action exists, but its consequences have not been learned
well enough to reproduce the benchmark sequence. The broad-encounter spider
result does not mean the controlled pursuit milestone has been achieved.

## Reproduce

```sh
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/driftmoor-mpl python3 -m experiments.run_approach_learning \
  --driftmoor-root /Users/jeff/Python/Driftmoor-spider-sim \
  --checkpoints "/Users/jeff/Documents/ChatGPT/White Fang/experiments/curriculum-learning-v1" \
  --output /private/tmp/approach-learning-new
```

Use a new directory. Training ordinary seeds start at 700000 and 701000;
focused episodes start 256 seeds later. Validation: 710000 ordinary /710256
focused. Forecast test: 720000 ordinary /720256 focused. Encounter evaluation:
730000–730007. Controlled cases: 740000–740002. Ranking audit: fixtures
770000–770005, candidate dice 780000–780007. Forecast and encounter evaluation
seeds do not enter training. Cases remain selected cavern fixtures, not new maps.

Data and weights:
`/Users/jeff/Documents/ChatGPT/White Fang/experiments/approach-learning-v1`.
`results/approach-learning-v1.json` records forecasts, encounter counts and
controlled tests. The raw external report contains an identical redundant
initial-model control; the compact report/future runner omit that duplicate.

144 project tests pass. Game checkout unchanged. No candidate model is promoted
into gameplay. New learning work remains uncommitted for review; nothing pushed.
