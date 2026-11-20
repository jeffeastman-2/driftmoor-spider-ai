# Live ecology lifecycle — November 18, 2026

Enable with `--lifecycle --spider-pair --feeding-cycle --spider-model PATH`.
The live experiment also uses `--random-prey --replenish-prey --spider-local-motion`.
A turn is six simulated seconds. These are authored sensory, goal and world
rules around the existing learned pursuit actor. They are not capabilities
learned by the neural network. The offline 44-input predictor is not installed.

## Current settings

| Setting | Current value |
| --- | --- |
| Hunger increase | 1 percentage point per turn, including waiting |
| Start mate-seeking | Below 10% hunger |
| Stop mate-seeking and return to food | At 20% hunger |
| Readiness to mate | Both below 20% hunger, at least 35% HP, cooldown complete |
| Courtship proximity | Visible partner within six feet; different sexes |
| Accepted courtship | Three admitted courtship actions for the pair |
| Mating cooldown | 20 turns (120 seconds), for both parents |
| Egg-sac hatching | Two turns after creation |
| Offspring | One hungry male and one hungry female |
| Starvation damage | 1 HP every turn at 100% hunger |
| Exhausted carcass removal | Immediate, at round end |
| Web sensing | Implicit network throughout each zone |

## Feeding and goal transitions

Spiders hunt prey, wrap a reachable carcass, and feed on its remaining food.
Wrapping takes three actions; feeding consumes up to 0.1 food units per action
and reduces hunger by the amount consumed. Cricket carcasses contain 0.2 food
units, rats 0.5, and every Shadowbeast generation 2.5. Other creatures use the
existing body-size-based food calculation. Feeding does not heal injuries.

Below 10% hunger, a spider starts seeking a mate. It can continue searching as
hunger rises, but at 20% it abandons that goal and returns to feeding or hunting.
These separate thresholds prevent constant switching while avoiding the old
behavior of seeking mates all the way to 60% hunger. Defensive and critically
wounded survival behavior can take priority over feeding or mating.

A searching spider checks its own cooldown and injury limits before approaching
or courting a receptive partner. If no suitable partner is sensed, it roams
and periodically advertises through the web. Seeking does not guarantee mating.

## Implicit zone webs and courtship signals

Each zone provides an implicit silk network. A spider senses living creatures
in its own zone through coarse web contacts: range rounded to five feet and
bearing to 45 degrees. The cue alone does not reveal identity, body shape,
health or intent. Zone membership, rather than standing inside a small authored
web rectangle, supplies this contact. Ordinary visual sensing remains separate.

When no receptive partner is sensed, a searching spider can vibrate the web on
every fifth search turn. The signal lasts two turns (12 seconds), advertising
whether the sender is receptive. Another spider detecting it through web sensing
receives a conspecific courtship cue and may approach. Actual courtship still
requires visual contact within six feet; sensing a vibration is not permission
to mate remotely. Visual conspecific cues can also indicate receptivity.

The observer map no longer draws the old blue rectangular web overlays in this
implicit-web simulation. The authored web items remain in the scene; their
small footprints no longer define the added zone-wide sensing network. Spiders
do not currently build new webs, and there is no learned communication protocol.

## Reproduction

Accepted courtship produces a visible egg-sac item at the female's position and
starts a 20-turn cooldown for both parents. Both must be below 20% hunger, at
least 35% HP, and off cooldown. Existing long cooldowns in an older episode are
capped at 20 remaining turns when the faster reproduction rules first apply.

The sac hatches two further turns after its creation frame into one male and
one female. Offspring immediately use the full giant-spider body/HP template,
start at 80% hunger, and receive fresh GRU and motor state. They share the fixed
pursuit weights; they do not inherit a parent's memories, wounds or current
hunger. They start acting the turn after hatching. Stable opaque contact IDs
are added without renumbering existing contacts.

Hatching requires room for both bodies on connected nearby floor. A crowded sac
waits rather than creating half a pair or moving offspring to a distant room.
Empty sacs remain visible as hatched items; carcass cleanup does not remove them.

Two-turn development is deliberate accelerated simulation pacing. Driftmoor's
`scenario/creatures/miscellaneous/spider_swarm` describes newly hatched broods
and is a possible future juvenile stage. Its swarm combat statistics are not
used for these individual offspring. Eggs have no damage or predation mechanics
yet; kinship restrictions and sexual cannibalism are not implemented.

## Replacements and carcasses

Each dead rat or cricket produces one replacement at a random valid floor
position. Each dead Shadowbeast likewise produces exactly one distinct
replacement at a random valid floor position at round end, using the original
Shadowbeast template and a fresh scripted controller. New residents first act
in the next turn. Spiders reproduce through offspring; death does not create
an automatic replacement spider.

A combat victim's edible carcass stays available while it contains food.
As soon as its food value reaches zero, it disappears from the observer map at
round end; there is no five-turn delay. Dead bodies with no food entry are also
removed. Identity, position and death records remain internally so contact IDs
cannot be reused and old deaths cannot trigger duplicate replacements. Removed
carcasses are not active actors or available meals.

Unlimited prey and Shadowbeast replacement continually adds food. Starvation
therefore measures an individual's difficulty finding or competing for food,
not depletion of a closed dungeon food supply.

## Hunger and starvation

Hunger increases by 0.01 per turn, bounded at 1. At 100% hunger, a living spider
loses 1 HP every turn. Damage is applied at round end, after an opportunity to
eat. Feeding below maximum hunger stops starvation damage and resets its
starvation counter. Dead spiders stop metabolizing.

For a 26-HP spider that never eats, starting at 80% hunger means roughly 20 turns
to maximum hunger, then about 26 damaging turns to death. The first damaging
turn is also the turn maximum hunger is reached: approximately 45 turns total.
Starting at zero hunger gives approximately 125 turns total. Wounds shorten
these times. These are experiment settings, not biological rates.

Starvation removes a dead spider from active actors. It does not invent an
enemy attack or trigger retaliation against a bystander. Wounded-spider behavior
also affects whether it obtains food, so results are not a pure population
carrying-capacity measurement.

## Viewer and replay

The live episode has a 3,000-turn limit. The **All spiders** filter includes
future generations; **All creatures** includes the other residents. Each
creature's turn appears as one table row: creature, HP, hunger, goal, action and
result. Clicking its name opens the full explanation. The transcript follows
new entries when the reader is at the bottom.

Transcript events distinguish egg laying, hatching, Shadowbeast replacement,
starvation damage and starvation death. Egg sacs appear as small white dots without labels; their details remain on
hover. Persistent yellow item and landmark labels are hidden, with hover
tooltips retained. Mate-seeking, approaching and courting spiders have pink
halos for females and blue for males, matching the displayed historical turn.
Spider body outlines and trails use the same pink/female and blue/male colors.
Shadowbeast body outlines and trails are red, including replacement generations.
Blue web-box overlays are hidden for this simulation.

Saved state records lifecycle events, parents and offspring, sex registry,
starvation counters, replaced-beast IDs, food and hunger. Rule-change schedules
preserve the hunger rate, starvation interval, reproduction pacing and zone-web
settings used by past turns. Resume deterministically replays those turns and
checks the reconstructed state before applying current rules to future turns.
The current population is preserved rather than reset when settings change.

Restart archives the previous episode and restores the original two spiders.
Other experiments retain their own saved rule schedules; the map renderer's
web-overlay option remains enabled by default for ordinary Driftmoor scenes.

## Courtship responses (experimental, not yet installed in the live viewer)

The 10% threshold starts an unsolicited mate search; it is not a requirement
for responding to an invitation. A spider below 20% hunger, at least 35% HP and
off cooldown may respond to a perceived web invitation or visible courtship.
Visible courtship is exposed only on an actually visible conspecific contact
while the pair has courtship progress; it does not expose hidden goals or
remote positions.

A responder approaches or reciprocates using the existing approach_mate/court
goals and engine commands. It retains a sensed, receptive partner during the
approach, preferring visual measurements over a duplicate web contact. Lost
contact, lost receptivity, its own hunger/health or cooldown ends the response.
There is no pursuit of an unseen mate. Defense and critical-health behavior
retain their existing priority. Successful mating, nutrition costs, cooldown,
hatching and food supplies are unchanged.

## Opt-in nutrition recovery / 30% cutoff experiment

Checkpoint `69ad4cd` preserves the response teacher and the first learned
lifecycle benchmark. The new headless experiment keeps independent mate-seeking
initiation below 10% hunger, extends the search/receptivity/engine mating cutoff
to 30%, and enables up to 1 HP recovery at round end after actual feeding or
while post-round hunger is below 10%. Recovery is capped at maximum HP and
never revives a dead actor. Feeding supports recovery even when hunger is still
above 10%; the passive, sated recovery ends at 10%. No extra food is consumed
solely for healing.

These are opt-in world/policy settings, not changed live/default rules. Existing
episodes still have no recovery and a 20% mating cutoff. Newborns inherit the
experiment's rule settings from the runner on their first acting turn. Health
requirements, mating cooldown, hunger growth, prey replacement, food quantities
and starvation damage remain unchanged.

`run_nutrition_recovery.py` runs the complete twelve-seed, 1,000-turn batch with
the response teacher and fixed learned pursuit weights. It does not retrain
or deploy the learned lifecycle selector. `report_nutrition_recovery.py` compares
against the checkpoint's `lifecycle-response-v1` recordings.

Full 1,000-turn comparison against the checkpoint response teacher:

| Outcome across twelve seeds | Checkpoint | Recovery below 10% + mating cutoff 30% |
|---|---:|---:|
| Runs ending with more than two living spiders | 4 | 6 |
| Runs ending with exactly one survivor | 0 | 1 |
| Runs that produce an egg sac | 5 | 6 |
| Total living spiders at turn 1,000 | 160 | 216 |
| Total egg sacs | 294 | 459 |

Seeds 53, 67 and 97 gain populations, ending at 45, 31 and 23 spiders. Seed 23
loses its previous growing population and ends at zero. Seed 109 retains one
survivor but never reproduces. Other new final counts are 44 (42), 32 (79),
40 (139), and zero (11, 37, 83, 127). Thus recovery and the wider window improve
the diagnostic batch overall without guaranteeing establishment in each seed.
Starvation still occurs; the higher aggregate death count accompanies more
births and does not by itself demonstrate higher per-spider mortality.

These combined changes cannot identify the individual contribution of healing
versus the cutoff. They also alter later actions, encounters and random-event
consumption. A 1,000-turn endpoint does not establish long-term equilibrium.
The default-rule seed-42 control still produces its first egg on turn 72, matching
the checkpoint. All 258 tests pass. No trained weights or live viewer were changed.

Compact results: [nutrition-recovery-10-30-v1.json](results/nutrition-recovery-10-30-v1.json).
Population traces, compressed decisions, manifest and comparison plot are outside
Git at `/Users/jeff/Documents/ChatGPT/White Fang/experiments/nutrition-recovery-10-30-v1`.

## Damage interrupts mating activities too

Incoming damage now interrupts `seek_mate`, `approach_mate` and `court`, as well
as the existing feeding/wrapping phases. When the attacker is identified and
visible, the teacher immediately selects defense using the existing attack and
movement controllers. An unidentified attack causes an alert wait; the spider
does not accuse its mate or another bystander. Existing short threat memory and
post-threat goal selection are retained. Hunger, injury and cooldown still
control whether mating can resume after the threat is gone.

The learned lifecycle executor tracks an attacker during those mating phases
too, but its network still chooses its own goals. This teacher change does not
silently override learned goal logits or retrain any weights. The running viewer
has not been restarted.

Regression checks cover each of the three mating phases, ordinary and critical
HP, unidentified attackers, and resumption after the attacker is defeated.

In the targeted seed-23 replay with the 10% recovery / 30% mating-cutoff profile,
the female now attacks the identified Shadowbeast immediately on turn 39 instead
of vibrating the web. She survives the encounter; the pair produces its first
egg sac on turn 74. The full 1,000-turn run ends with 50 living spiders (peak 52)
and 78 egg sacs, versus no reproduction and extinction before this interruption
change. This is a single-seed regression result, not a new twelve-seed benchmark.
[Compact replay comparison](results/mating-defense-seed23-v1.json).
