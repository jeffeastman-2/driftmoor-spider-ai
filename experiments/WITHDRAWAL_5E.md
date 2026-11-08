# 5e withdrawal rules and a cornered spider

The reviewed turn-and-move experiment was committed and pushed as `1963872`.
This follow-up implements withdrawal mechanics and the requested response to
blocked flight. These new changes remain uncommitted for review.

## Implemented mechanics

- **Retreat = Disengage + movement up to Speed.** Turn toward the perceived
  threat while backing away; movement is protected from opportunity attacks.
- **Flee = Dash + movement up to twice Speed.** Turn away and move; leaving
  a visible enemy's melee reach can provoke one opportunity attack.
- **Advance = movement up to Speed.** No Disengage protection.
- A reaction is restored at the start of the creature's next turn. A miss spends
  it too. A reaction adds no extra turn or elapsed time.
- Opportunity attacks resolve just before the mover leaves reach. If defeated,
  the mover stops there; otherwise it continues using its remaining movement.
- Both actors' turns now fit into one six-second round in the fixed turn order.
  The previous arbitrary six-foot movement step has been removed.

The movement preview clips to existing home restrictions and impassable geometry.
A quarter-foot path scan brackets an inside-to-outside reach transition, then
bisection locates the trigger just inside reach. This detects passing into and
out of reach even when both move endpoints are outside. Actual movement and
damage still go through the game adapter and combat resolver.

Reactions currently happen automatically whenever an eligible attack is admitted;
the neural policy does not decide whether to reserve the reaction. Visibility,
weapon/body reach and the existing attack cone are enforced. The native
half-foot weapon tolerance is retained. These are 5e-inspired rules within the
experiment's body geometry, not a complete grid-based 5e implementation.

Official rule references:
[2014 combat](https://www.dndbeyond.com/sources/dnd/basic-rules-2014/combat),
[2024 combat](https://www.dndbeyond.com/sources/dnd/br-2024/playing-the-game).
There is no automatic rear damage bonus, Sneak Attack, or extra backstab damage.

## Blocked flight -> counterattack

The spider does not know in advance that its zone prevents escape. After a
surviving Dash, it compares actual displacement with the requested movement.
If geometry or a home restriction shortened it, its own `escape_blocked`
feedback becomes true. Being killed by a reaction is not reported as a physical
blockage. A successful later flight clears the flag; stationary actions retain it.

At critical health (below 20%), a spider with that feedback switches the teacher's
intent from `flee` to `cornered`. It faces the observed threat and attacks if its
estimated body/socket geometry permits. Otherwise it can turn, approach or wait.
Aiming and attacking fit in one action; there is no 30-degree turn-slot penalty.

This is an authored do-or-die policy learned through demonstrations. It grants no
damage, accuracy, armor or Dodge bonus. It is not proof that every escape route
is blocked: it records a failed flight in the chosen direction and commits to
fighting while critically wounded. No route planner or alternative-exit search
has been added. Gil's existing withdrawal preference is unchanged.

A flee-only spider teacher is evaluated against the same current Gil teacher as
a matched control. It ignores blocked-flight feedback and continues trying to flee.

## Private observations and networks

Both networks still have a 41 -> 32 encoder, GRU(48), eight action scores, and
the existing contact-memory and future-prediction heads. Input index 29 now holds
own blocked-flight feedback instead of the duplicate pain indicator; pain remains
in the existing feedback values. No map or hidden opponent state enters policies.
Own Speed and reaction availability are available in self metadata.

The checkpoint/data schema is `withdrawal-cornered-v6`. Older weights must not be
used with these changed input and action dynamics. The normal loader rejects
older schemas by default; explicit historical inspection can override the check.

The current action head chooses **argmax**, not sampling or beam search.
The 15% random exploration used while gathering demonstrations is separate.
Future-prediction heads are trained but are not used to rank imagined branches.

## Run

From /Users/jeff/Python/driftmoor-spider-ai:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m experiments.run_teacher_experiment \
  --driftmoor-root ../Driftmoor-spider-sim \
  --output /tmp/driftmoor-withdrawal-cornered-v6
```

Use an empty output directory. Defaults: 160 training episodes from seed 130000,
24 validation from 131000, 32 test from 132000, and 160 training epochs.
Eight evaluation seeds 134000..134007 run in eight scenarios and eleven
pairings: **704 encounters**. Checkpoint selection uses validation loss only.

Full weights, datasets and traces are outside Git at:
`/Users/jeff/Documents/ChatGPT/White Fang/experiments/withdrawal-cornered-v6/`

The compact report is `results/withdrawal-cornered-v6.json`.
Blocked-flight decision counts are derived from saved private-observation traces
by `blocked_flight_behavior()`; the completed run was summarized afterward
without changing weights or rerunning policy choices.

## Results

In critically wounded spider states following blocked flight, against the same
current-teacher Gil:

| Spider policy | Decisions | Attack requests | Admitted attacks | Repeated flight |
|---|---:|---:|---:|---:|
| Flee-only teacher | 132 | 0 | 0 | 132 |
| Cornered teacher | 146 | 45 | 14 | 0 |
| MLP | 114 | 39 | 9 | 0 |
| GRU | 116 | 51 | 8 | 0 |

The behavior switch is present in both learned controllers: they stop repeating
the failed flight and attempt to fight. They still make many invalid attack
requests. The physical engine rejects those requests, and action agreement alone
does not establish good combat judgment.

Spider survival was 56/64 for the cornered teacher, flee-only control, and GRU
against the fixed teacher opponent. This run therefore does **not** establish a
survival benefit from the new policy. One training seed and this small single-scene
suite do not demonstrate broad strategic competence.

## Verification and remaining scope

112 project tests pass, including Disengage suppression, Dash distance, reaction
consumption and refresh, lethal interruption, pass-through movement, unseen
targets, blocked paths, round timing, reaction damage accounting, and the
observed blocked-flight counterattack.

Generic game mechanics and the main game checkout were not modified. Current
limits include 2.5D collision/occlusion, authored home restrictions, fixed Gil-first
initiative, simplified loadouts (no spells or poison), and no death saves or
condition-specific reaction rules. Ordinary advance remains move-and-hold;
move-and-attack combinations and full turn action economy are not implemented.
There is no reinforcement learning or reward-based self-play yet.
