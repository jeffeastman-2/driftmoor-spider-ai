# Freight Quarter ecology and live observation

Level-2 is now a runnable, multi-resident environment using Driftmoor's authored
geometry, scene loading, sensory gates, movement, body geometry and melee dice.
The original Level-1 encounter and its checkpoints remain separate.

## Watch it

From the spider project, with the sibling game's dependencies installed:

```sh
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/driftmoor-mpl python3 -m experiments.watch_level2 \
  --output /tmp/driftmoor-level2-watch-1 --seed 42 --rounds 120 --port 8081
```

Open http://localhost:8081 . The viewer starts paused. **Run / resume**,
**Pause** and **One round** control the actual simulation. **Restart from
beginning** resets actors, HP, webs, RNG and policy memory using the same seed,
then pauses at round zero. The previous episode is preserved under
`previous_episodes/episode-NNNN/`, and the viewer clears its old log/history. The map's own Play
button and turn slider replay recorded history; they do not advance creatures.
Choose **All creatures** to expand the default spider-only decision log.
Researcher annotations use names such as Giant Spider, Cave Rat 1 and
Shadowbeast. These names do not enter the creature sensory inputs.

If the spider loses all live contacts, the current baseline selects Rest.
There is no search-after-kill or exploration behavior yet; a stopped spider
with empty contacts is not necessarily blocked or satiated.

This uses the game's existing web spectator and turn-slider dungeon renderer.
It is an observer map, not the spider's visual field. Policies receive private
observations only. The annotations describe recorded sensory evidence, the
scripted goal rule, commanded movement/attack and actual engine result; they
are not invented neural explanations. No LLM calls are made.

Every round writes an atomic native-shaped `day1_*.json` checkpoint containing
positions, HP, web state, private observations, commands and outcomes. These
are observation/analysis artifacts, not full `run_session` resume checkpoints.
They can be opened later with the game's `web_spectator.py` (read-only).
Web overlays show latest silk condition; creature positions have history.

## Shared game additions

The experimental game branch contains:

- `scenes/echoing_caverns_level2.md`, using the shared Level-2 PLACE/GEOMETRY/zone files.
- A separate freight-quarter spider; two cave rats, two large cave crickets,
  three tethered giant lizards, the freight master, and a shadowbeast scout.
- Explicit authored creature stat/body definitions. Small prey are homebrew
  Driftmoor ecology templates, not purported official rules stat blocks.
- Two physically connected sensing sheets in Fungal Stores and a disconnected
  Cistern sheet. Geometry and zone item definitions place them; `WEBS.md`
  declares their contact volumes and durability.
- Shared `engine.webs`: silk contact gated by actual connectivity, height and
  Web Sense; two-for-one movement cost inside intact silk unless Web Walker;
  destruction, fire vulnerability and owner repair helpers. Item state holds
  mutable HP, and both narrated and structured perception can report contacts.
- Native renderer web overlays and reusable spectator app construction.

The spider senses coarse silk range/bearing without opponent body, HP, armor,
race or identity. Visible bodies provide ideal measured geometry. Opaque track
tokens replace actor names in controller observations. Corpses stop generating
live contacts, and destroyed web patches stop conducting and slowing movement.

## Existing scene defects repaired

Duplicate zone item IDs and missing definitions formerly caused all Level-2
items to be discarded by the loader. Instances now have unique IDs and
geometry placements. The referenced freight-master source files were missing;
they now exist with actual HP, equipment and physical dimensions.

The corrected authored layout from Driftmoor main (5275976, retained by
e50fa98) is the source of truth. Existing vertex elevations and scene metadata
are preserved; resident/web z is height above the floor. The experimental
connector zones added to the stale layout have been removed. Shared engine
movement checks the entire segment through the zone union, preventing a move
from jumping across solid rock merely because its endpoint is another room.

Unsupported draft puzzle kinds are preserved under **Narrative Challenges**
instead of pretending to be executable puzzles and failing registry loading.
Gate/lock/hazard prose is not newly implemented. Lizards are confined by their
explicit five-foot home radius, representing tethers; releasing them through a
working gate mechanism remains future work.

## Controller baseline and limitations

This first environment milestone uses scripted controllers, not the previous
41-input neural controller. Its variable-length observation contract is
`ecology-private-contacts-v1`; old weights must not be silently applied to it.

The spider and shadowbeast use explicit goal baselines: feed (pursue/attack),
survive (withdraw at critically low health), rest, and maintain locally observed
damaged webs. Other animals wander or flee visibly larger creatures. All
policies receive observations rather than an environment object. The baseline
is deliberately simple: no route planner, learned subgoals or clever ambush.
Appetite is an episode preference, not nutritional depletion. Feeding does not
yet include carcass consumption, energy recovery or satiation. Rest does not
magically heal. Reproduction, full web construction and live language-command
routing to the repair/damage helpers are not implemented.

Movement and combat are a bounded offline adapter rather than the full narrated
session: actors act in stable order each six-second round, attacks are stationary,
and there is no initiative roll, opportunity-attack/reaction system, poison
extension or complete 5e combat simulation in this adapter. Narrated scene play
continues to use its normal session mechanics. Noncombat web sensing is coarse,
not a full elastic propagation or entanglement simulation. The narrated session
currently uses authored height for web contact; the offline adapter tracks its
explicit actor heights. Time-of-day is fixed during this short baseline.

## Next learning work

1. Collect multi-contact experiences in this environment and establish behavioral
   measurements (prey encounters, legal bites, damage, survival, web repairs and
   movement blocked by geometry). These baselines establish mechanics, not
   evidence of neural progress.
2. Design an encoder for a variable set of sensed contacts plus own state and an
   explicit goal. Preserve the learned bite component where its input contract
   applies; do not force several opponents into the old single-opponent vector.
3. Train action-conditioned consequences from engine transitions, including
   consistent permission/damage predictions. Compare against these baselines.
4. Add longer-timescale goal/subgoal evaluation before claiming a configurator
   or hierarchy. Give web preparation a measurable future payoff before learning
   complete construction.
5. Use the Fey Glade as a held-out environment after setting up a suitable
   ecology population there. Its original story scene is unchanged for now.

## Batch run

```sh
python3 -m experiments.run_level2_ecology --seed 42 --rounds 30 \
  --output /tmp/level2-ecology.json
```

Use new output paths. Generated traces and maps stay outside git. This baseline run trains no models and needs no LLM provider. The separate
[three-species neural experiment](ECOLOGY_DJEPA.md) now has trained experimental
weights; none has been promoted over this baseline.


Defeated creatures remain as corpses: their markers are dimmed and crossed in
red, with a dead tooltip. This is turn-dependent, so replaying an earlier turn
shows the living marker. Corpse consumption remains future work.

After stopping the viewer, pass `--resume` with the same output directory and
seed to resume its deterministic scripted episode. It validates replayed
positions, HP, web state and final decisions before touching saved checkpoints;
if they differ (for example after mechanics/scene changes), it refuses to
resume. A validated episode opens paused at its saved round.

Dormant resident activation is now processed by the ecology harness using
Driftmoor’s native triggers. The goblin can wake when a living opposing-faction
resident enters its office. See [the curriculum report](ECOLOGY_CURRICULUM.md).
