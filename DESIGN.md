# Current spider controller — design reference

Start here for the improved **learned pursuit spider**. Older experiments have
other action sets; this page describes the imitation-trained pursuit model.

Quick navigation: [diagram](#the-mental-model) ·
[commands](#everything-the-pursuit-network-can-choose) ·
[36 observation values](#observation-vector-36-numbers) ·
[GRU explained](#gru-inputs-memory-and-learned-parameters) ·
[21 candidate values](#candidate-vector-21-numbers) ·
[example turn](#one-concrete-turn) ·
[parameter counts](#how-many-learned-parameters-altogether)

## The mental model

```text
Private senses + an explicitly selected prey contact
                  │
          36 observation numbers
                  │
        Encoder MLP: 36 → 64 → 32
                  │
          GRU: 48-number memory ─────────────┐
                                           │
Measured prey + local sensors → candidate commands
                                           │
                           21 numbers per candidate
                                           │
                         memory + candidate = 69
                                           │
                  Scoring MLP: 69 → 64 → 32 → 1
                         (one score per candidate)
                                           │
                          Choose lowest score
                                           │
                              Driftmoor engine
                                           │
                   Actual movement / hit / miss / damage
                         feed the next observation
```

**The last MLP scores commands that were already constructed.** It does not
output a fixed vector of eight actions or invent a new command. The same MLP
runs over every candidate with the same current GRU state. The GRU advances
once per turn, not once per candidate.

For this trained model, scores are **negative action logits learned by
imitation**. They are not calibrated probabilities, damage forecasts, or JEPA
cost estimates. The separate 23-input candidate variant is an experiment;
the improved installed checkpoint uses 21 candidate inputs.

## How to read the numbers in the diagram

A number such as **36** is the length of a vector: a list of numerical values
for this turn. It is not a count of neurons in the whole model or a count of
learned weights. A number such as **36 → 64** describes a learned transformation
from a 36-value list into a 64-value list.

There are three different kinds of numbers here:

| Kind | Example | Changes when? |
|---|---|---|
| Measured/encoded inputs | Own health, target distance, candidate direction | When the observation or proposed command changes |
| Learned parameters | MLP matrix entries; GRU gate weights and biases | During training; fixed during live inference |
| Recurrent state | The creature's 48 GRU memory values | Once per decision turn |

The 21 candidate values describe **one possible command**. They do not mean
there are 21 commands. A typical turn might score 19 commands, each described
by 21 values, using one shared scoring network.

## Observation vector: 36 numbers

Indices below are **zero-based**, matching Python lists. For example, index 0
is the first number, and indices 1–3 are the next three numbers in order.

### First 24: own body, recent feedback, local surroundings and context

| Indices | Count | Meaning, in order | Encoding |
|---|---:|---|---|
| 0 | 1 | Own HP fraction | current HP / initial HP |
| 1–3 | 3 | Own width, depth, height | each in feet / 10 |
| 4 | 1 | Own movement budget | feet / 30 |
| 5 | 1 | Own weapon reach | feet / 10 |
| 6–8 | 3 | Previous action was wait, move, attack | three 0/1 flags |
| 9 | 1 | Actual distance moved by previous action | feet / 30 |
| 10 | 1 | Previous action was admitted as an attack | 0 or 1 |
| 11 | 1 | Damage dealt by previous action | HP / 20 |
| 12–19 | 8 | Local clearance at relative headings 0°, 45°, …, 315° | each in feet / 10; measured to ten feet |
| 20–23 | 4 | Context flags: feed, survive, forage, explore | four 0/1 flags |
| **Subtotal** | **24** | | |

**Previous action** means the spider's previous motor result, not the prey's
previous action. An admitted attack can miss: admission = 1 and damage = 0.
A rejected attack has admission = 0. Before the first action, the previous-action
flags and result values are zero.

The four context flags are inherited from the earlier ecology encoder. They
are authored inputs, not goals invented by the network. For the spider,
`forage` is normally zero; adequate health plus contacts gives `feed`, and
low health with contacts gives `survive`. No contacts normally gives `explore`,
or an all-zero context if critically injured (`rest` has no dedicated flag).
These legacy context names are distinct from the explicit pursuit task and
the proposed roaming behavior. There is no separate `roam` input flag here.

### Last 12: the contact that the pursuit task has selected

| Indices | Count | Meaning, in order | Encoding |
|---|---:|---|---|
| 24–26 | 3 | Measurement modality: vision, web, nonvisual | three 0/1 flags |
| 27–29 | 3 | Target offset: forward, left, up | forward/30, left/30, up/10; feet |
| 30–32 | 3 | Observed target width, depth, height | each in feet / 10 |
| 33 | 1 | Reported target range | feet / 30 |
| 34–35 | 2 | Bearing relative to own heading | cosine, sine |
| **Subtotal** | **12** | | |
| **Grand total** | **36** | **24 + 12** | |

Forward and left are in the spider's horizontal body frame. Vertical offset
uses floor-relative elevations from the sensory adapter. Bearing comes from
the sensory report and may be quantized; range is also rounded by that adapter.
Consequently, bearing/range need not exactly equal the direction/length of the
more precise visible offset vector. The redundant representations are present
in the current contract; they are not extra contacts.

For this pursuit task, a newly selected contact must be visible. The three
modality flags exist because this format was inherited from the broader ecology
model. When the task briefly remembers an occluded target, the stored contact
still has its **last measurement's** modality. A vision flag therefore does not
by itself guarantee fresh visibility. Target-memory age is tracked separately
by `PursuitGoal` and logged, but is **not an explicit one of these 36 inputs**.
With no selected or remembered contact, all 12 contact values are zero.

These are normalized numerical features, not text or an image embedding.
Dividing by 10, 20 or 30 puts quantities on roughly comparable scales; it does
not generally clamp them to [0,1]. Direction components can be negative.

There is no target HP, armor class, creature name, race label, global map,
absolute world position, hunger or neurotransmitter vector in these 36 values.
Body heading is used to construct relative coordinates rather than appearing
as a raw angle. Own roll and pitch are not separately encoded here.

## The first MLP: measurements become 32 learned features

The observation encoder is:

```text
36 measured/context values
    → Linear(36, 64) → ReLU
    → Linear(64, 32) → ReLU
    → 32 learned features
```

A linear layer has weights **and biases**, then the ReLU sets negative outputs
to zero. The two layers can combine inputs nonlinearly: for example, proximity
and body geometry together might matter differently from either alone.

We do not assign fixed meanings to the resulting 32 numbers. They are a learned
representation useful for predicting the teacher's decisions. It would be
misleading to label feature 7 “hunger” or feature 12 “fear” without examining
what the trained network actually represents.

## GRU: inputs, memory and learned parameters

GRU means **Gated Recurrent Unit**. Its current configuration is exactly:

```python
nn.GRU(input_size=32, hidden_size=48, num_layers=1,
       bias=True, batch_first=True, dropout=0, bidirectional=False)
```

| Setting | Value | Meaning |
|---|---:|---|
| Input size | 32 | Values from the first MLP at this turn |
| Hidden size | 48 | Values carried from the previous turn |
| Recurrent layers | 1 | One GRU layer, not a stack |
| Direction | Forward only | It cannot read future observations |
| Biases | Enabled | Learned offsets in the gate calculations |
| Dropout | 0 | No GRU dropout |
| `batch_first` | True | Tensor axes are batch, time, features |

At each turn the computation is:

```text
32 new encoder features x[t] ──────┐
                                  ├── GRU ──→ 48 updated state values h[t]
48 previous state values h[t−1] ──┘                   │
                                                     ├─→ scoring MLP
                                                     └─→ next turn's GRU
```

Because this is a single forward GRU layer, its output at a turn is also its
updated hidden state. We do not concatenate separate 48-value “output” and
48-value “memory” vectors. There are just **48 values** to pass to the scorer.

The initial hidden state is zero (`hidden=None` asks PyTorch to create it).
Thereafter the policy stores it in `self.hidden`. A fresh policy, including a
viewer restart, starts with new memory. Creatures can share weights but must
not share their individual hidden state.

### What the gates do

The GRU learns two gates and a candidate replacement state. Each has 48 values:

* **Reset gate `r`:** how much old state participates in constructing new content.
* **Update gate `z`:** how much old state to retain in the final mixture.
* **Candidate state `n`:** proposed new content from this turn and gated history.

For this implementation, the final mixture is:

```text
h[t] = (1 − z) × n + z × h[t−1]
```

Multiplication is element by element. A `z` value near 1 retains that component
of the old state; near 0 replaces it with the candidate. The gates themselves
are computed from both current encoded observations and previous state using
learned matrices. Sigmoid makes gate values lie between 0 and 1; tanh bounds
the candidate state between −1 and 1.

For readers wanting the exact PyTorch gate equations:

```text
r = sigmoid(W_ir x + b_ir + W_hr h_previous + b_hr)
z = sigmoid(W_iz x + b_iz + W_hz h_previous + b_hz)
n = tanh(   W_in x + b_in + r × (W_hn h_previous + b_hn))
h = (1 − z) × n + z × h_previous
```

The reset multiplication in the third equation is outside the recurrent
matrix product, as implemented by PyTorch. These are learned operations, not
handwritten “remember the last three turns” rules.

### Learned GRU weights are different from live memory

PyTorch packs the three sets of gate/candidate calculations into these tensors:

| Parameter tensor | Shape | Number of learned values |
|---|---|---:|
| Input-to-gates weights | 144 × 32 | 4,608 |
| State-to-gates weights | 144 × 48 | 6,912 |
| Input-side biases | 144 | 144 |
| State-side biases | 144 | 144 |
| **Total GRU parameters** | | **11,808** |

Here **144 = 3 × 48**, not 144 units of persistent memory. Only the 48 hidden
values are carried between turns. During training, gradients adjust the
11,808 weights/biases. During live inference, those parameters stay fixed while
the 48 state values change with experience.

### Turns, tensors and memory horizon

For one creature deciding one turn at a time:

| Tensor | Shape |
|---|---|
| Raw observation | `[1, 1, 36]` |
| Encoder output into GRU | `[1, 1, 32]` |
| GRU hidden state | `[1, 1, 48]` = layers × batch × hidden values |
| GRU output for this turn | `[1, 1, 48]` = batch × time × hidden values |

The two GRU tensor shapes happen to look identical here; their axes mean
different things. `batch_first` affects input/output layout, not hidden-state
layout. During training, batch size and sequence length can be larger than 1.

One input turn per call does **not** mean one-turn memory. The previous hidden
state is supplied to every subsequent call. There is no fixed “last two turns”
window, nor does a 48-value state mean 48 turns. Learned gates decide what
history to retain in compressed form; they may forget important information.
The imitation training uses whole collected episodes, up to 48 turns, often
ending much earlier when prey is defeated. Its loss can train the encoder and
GRU through those sequences. This does not guarantee reliable memory for every
longer live encounter.

Also keep **planning depth** separate: remembering history does not mean
searching several future action sequences. The current imitation policy scores
this turn's commands; it does not roll out a multi-turn planning tree.

## Candidate vector: 21 numbers

Every candidate gets its own feature vector. The vector changes between
commands, while the 48 current memory values stay the same.

| Indices | Count | Meaning, in order | Encoding |
|---|---:|---|---|
| 0–2 | 3 | Command kind: wait, move, attack | three 0/1 flags |
| 3–4 | 2 | Proposed travel direction relative to body heading | cosine, sine |
| 5 | 1 | Requested movement distance | feet / 30 |
| 6–17 | 12 | Target-contact description, in the same order as observation indices 24–35 | populated for an attack's target; otherwise zeros |
| 18 | 1 | Local clearance along the proposed move direction | feet / 10; zero for wait/attack |
| 19–20 | 2 | Command direction relative to the pursuit target direction | cosine, sine |
| **Total** | **21** | **3 + 2 + 1 + 12 + 1 + 2** | |

Important details:

* **Approach is encoded as `move`.** It does not get a fourth action-kind flag.
  Its direction and distance distinguish it from sampled movement commands.
* A move/wait command has no attack target field, so indices 6–17 are zero.
  The scorer still has the selected prey's context through the GRU state.
* Direction encodings avoid a numerical discontinuity between 359° and 0°.
  A direction matching the reference is `(cos, sin) = (1, 0)`; a reversal is
  `(-1, 0)`; a 90° leftward direction is `(0, 1)`.
* Wait and attack have no travel-heading field. The current implementation
  substitutes body heading for the direction features; the action-kind flags
  tell the scorer these are not movement commands.
* The local clearance is taken from the nearest of the eight sensed directions.
  For a precise approach between rays, it is an approximation, not a new engine
  query for the exact command path.
* These features describe a **proposal**, not guaranteed physical success.
  Requested distance and sensed clearance do not replace engine adjudication.
* The optional 23-value experiment appends current planar target distance/10
  and an estimated post-move planar distance/10. Those extra two values are not
  part of the installed 21-value checkpoint.

## The last MLP: the same memory, a different candidate each time

```text
48 current GRU values + 21 candidate values = 69 inputs
    → Linear(69, 64) → ReLU
    → Linear(64, 32) → ReLU
    → Linear(32, 1)
    → one numerical score
```

There are three linear layers here, with no final sigmoid or ReLU. Scores may
be positive or negative. All candidates share exactly the same learned weights.
If there are 19 candidates, inference produces 19 scores and picks the minimum.
The GRU is **not** updated 19 times. The implementation evaluates the candidate
batch together, which is equivalent to scoring each against the same state.

During imitation training, **negative scores** act as logits for a softmax over
the available candidates, and cross-entropy rewards the teacher's choice.
During live play, selection is deterministic minimum-score selection; it does
not sample a random action from that distribution. Engine dice remain random.
A good imitation score is not automatically a reliable future-outcome estimate.

## One concrete turn

Suppose the spider is healthy, has body dimensions 4 × 4 × 2 feet, speed budget
30 feet, reach 5 feet, and ten feet of locally clear floor in every sensed
direction. It sees its selected prey **15 feet directly ahead**, level with its
body root, with observed dimensions 1 × 0.5 × 0.5 feet. Assume no previous action.
This is an illustrative input, not a claim about a particular recorded turn.

The inputs include:

| Feature | Encoded value |
|---|---|
| Own HP fraction | `1` |
| Own width/depth/height | `0.4, 0.4, 0.2` |
| Own speed and reach | `1, 0.5` |
| Last-action flags/results | zeros |
| Eight clearances | eight `1`s |
| Feed/survive/forage/explore | `1, 0, 0, 0` |
| Contact vision/web/nonvisual | `1, 0, 0` |
| Contact forward/left/up | `0.5, 0, 0` |
| Contact dimensions | `0.1, 0.05, 0.05` |
| Contact range and bearing | `0.5, 1, 0` |

Now compare **move ten feet forward** with **bite**:

| Candidate detail | Move forward 10 ft | Bite selected prey |
|---|---|---|
| Kind flags | `0, 1, 0` | `0, 0, 1` |
| Direction relative to body | `1, 0` | `1, 0` (body-heading default) |
| Requested distance | `10/30` | `0` |
| Candidate target block | twelve zeros | selected contact's twelve values |
| Move clearance | `1` | `0` |
| Direction relative to target | `1, 0` | `1, 0` |

Both candidates receive the **same updated 48-value memory**. The scorer can
therefore distinguish “this prey is ahead” from “this proposal moves toward it”
and compare movement against an attempted bite. We cannot derive its actual
scores just by reading the table: they depend on trained weights and memory.

At this particular distance, precise approach proposes ten feet forward too,
so that duplicate is omitted. With one visible selected prey, the resulting
list has 18 commands rather than 19.

After selecting a command, the engine resolves it. On a move it may cover less
than requested because of geometry or other mechanics. On an attack it may
reject the bite, admit it but roll a miss, or deal damage. The **actual** result
becomes next turn's feedback; the memory and target task update from that new
observation, not from an assumed successful move or hit.

## How many learned parameters altogether?

For the current 21-value pursuit checkpoint, verified directly from the model:

| Component | Learned parameters |
|---|---:|
| Encoder MLP | 4,448 |
| GRU | 11,808 |
| Candidate-scoring MLP | 6,593 |
| **Total** | **22,849** |

All counts include biases. The 48 live GRU state values are **not** additional
learned parameters. Nor is the explicit target task's remembered contact: that
is separate controller state. The 23-candidate-value variant has a different
scorer input width and therefore a different parameter count.

## Everything the pursuit network can choose

| Candidate | Number | What the engine receives |
|---|---:|---|
| Wait | 1 | `wait` |
| Move 5 feet | 8 | `move`, direction, distance 5 |
| Move 10 feet | 8 | `move`, direction, distance 10 |
| Bite selected prey | 0 or 1 | `attack`, visible selected contact |
| Approach selected prey precisely | 0 or 1 | `move`, calculated direction and step length |

The eight sampled directions are offsets **0°, 45°, 90°, 135°, 180°, 225°,
270°, 315° from the spider's current body heading**, not fixed compass bearings.
Five- and ten-foot movement candidates exist at every offset.

Approach aims along the measured vector toward the committed contact. Its
step is `min(10, max(1, planar_distance − 5))` feet, following the teacher's
five-foot standoff convention. It is generated before scoring, not a scripted
fallback that overrides the network. It may use a briefly remembered contact.
An exactly duplicate approach command is not added twice.

Thus the usual list has **17–19 candidates**, not eight: 17 basic options,
possibly a visible bite and a distinct precise approach. Attack candidates for
other contacts are filtered out while pursuing the committed target. Reach
is not used to remove the bite option; the engine can reject it.

There are no separate neural candidates named `turn`, `retreat`, `flee`,
`roam`, `feed`, `repair_web`, `wrap`, `eat`, or `breed` in this pursuit model.
Moving away can result from selecting a move direction, but that does not
mean the model has a separately learned retreat/flee goal.

## Goal, command, and engine are different levels

| Level | Example | Current responsibility |
|---|---|---|
| Goal/task | Pursue this sensed contact | Authored target selection and persistence |
| Candidate construction | Approach, side-step, bite, wait | Authored motor options from private measurements |
| Action selection | Choose one candidate | Learned MLP → GRU → scoring MLP |
| Physical resolution | Actual displacement, legal bite, dice outcome | Driftmoor engine |

The target task retains a contact instead of switching whenever something
else becomes closer. It can remember the last measured position for up to
four unseen turns, adjusting for actual own movement, then expires it. No
hidden opponent position or HP enters inference. Learned weights are fixed
during live play; the GRU's individual memory changes each turn.

## Body state, goals and local movement

The live feeding experiment uses the same trained 36-input encoder, 48-number
GRU and candidate scorer. Hunger and carcass measurements go to an **authored
goal controller**, not into those trained weights. There is no new learned
feeding action and no retraining in this change.

`FeedingPolicy` in `spider_sim/feeding.py` configures the following tasks:

* **Hunt:** existing learned pursuit, with local movement recovery.
* **Approach meal:** walk toward locally visible food, preferring the current meal.
* **Wrap:** three reachable turns to secure a carcass (18 simulated seconds).
* **Feed:** consume at most 0.1 food units per turn until full or food is exhausted.
* **Seek mate:** once full, roam even if no partner is present. Resume hunting at
  hunger 0.6. In the optional paired mode, visible conspecific cues and patterned web vibrations identify potential partners. Courtship is admitted only when both spiders are at hunger 0.2 or less and at least 35% HP, within six feet and visible. Three uninterrupted courtship actions produce an accepted-courtship event and a one-hour cooldown. Eggs, offspring and sexual cannibalism are not modeled.
* **Defend:** any actual incoming attack damage while wrapping, feeding or approaching a meal interrupts food handling. Fight the identified visible attacker; retain the meal commitment and return to food when the threat is defeated or no longer observed for four turns. This takes priority over the critical-HP rule. The rule uses private damage feedback and measured attacker positions, not remote world state.
* **Survive:** otherwise below 35% HP, interrupt feeding/search and withdraw from the nearest
  visible contact; with no visible threat, wait. This is an authored precaution.

Hunger is bounded from 0 (full) to 1 (very hungry), starts at 0.8 and increases
0.001 per six-second round, including waiting. Feeding reduces it by the amount
actually consumed. There is no starvation injury, healing or nutrition effect
on combat yet. The constants are tunable experimental pacing, not biological
measurements. A full spider takes approximately one simulated hour to reach the
hunting threshold again; active search does not currently burn extra energy.

A cricket supplies 0.2 units, a rat 0.5, and a Shadowbeast 2.5. Other prey use a
bounded body-volume proxy. These food values are explicit rules: the authored
cave cricket is physically larger than the authored rat, so geometry alone
would give the opposite ordering. No species name enters the learned actor.
Remaining food stays in the carcass; feeding cannot consume a living, remote,
unseen, unwrapped or exhausted target. Corpses are distinct from replacement
prey. No dragging, carcass decay, digestive delay or persistent food-location
memory is implemented yet.

`LocalLocomotion` in `spider_sim/locomotion.py` adds gentle heading drift during
roaming and larger turns after local obstruction or measured movement failure.
It uses local clearance and own movement feedback, not global routes. It can
still fail in complex layouts. The adapter may replace a neural move with an
authored detour; decision annotations distinguish the source of each command.
Travel and body heading are separate; a visible pursuit contact can be tracked
while moving sideways, without reading hidden target coordinates.

With `--replenish-prey`, each dead rat/cricket creates one distinct, full-health
replacement at round end. With `--random-prey`, a separate seeded placement RNG chooses a random zone and valid unoccupied floor position; without it, the original authored home is used. Newborns act next round;
old contact tokens and dead carcasses remain. This is a population-maintenance
rule, not learned reproduction or a model of maturation.

Installed on localhost:8082 on November 17, 2026, in a fresh paused episode;
localhost:8081 remains the unchanged scripted control. The old 8082 episode
is preserved in its previous output directory.

Run the optional mode with `--spider-model PATH --spider-local-motion
--replenish-prey --feeding-cycle`. Existing scripted baselines retain their old
behavior. Inspect the live mode header and saved controller signature to tell
which implementation a running process has loaded.

## Where to look in code

| Question | Definition |
|---|---|
| Base wait/move/attack inventory | `spider_sim/djepa_ecology.py`: `candidates()` |
| Filter to selected prey; add precise approach | `spider_sim/pursuit.py`: `actions()` |
| Target commitment and memory | `spider_sim/pursuit.py`: `PursuitGoal` |
| Input numbers for observation and candidate | `spider_sim/pursuit.py`: `features()`, `action_features()` |
| Network layers | `spider_sim/pursuit.py`: `PursuitModel` |
| Scores, selection, optional facing/rest controls | `spider_sim/pursuit.py`: `PursuitPolicy.choose()` |
| Feeding goals and food values | `spider_sim/feeding.py` |
| Local roaming and blocked recovery | `spider_sim/locomotion.py` |
| Native movement and combat dispatch | `spider_sim/ecology.py`: `EcologyEncounter.act()` |
| Viewer mode and readable decision annotations | `experiments/watch_level2.py`: `LiveRun`, `explanation()` |

The earlier eight-action encounter model and the three-species ecology model
are separate experiments. For the active pursuit design, use this page rather
than combining their action lists. Results and training details are in
`experiments/PURSUIT_MOTOR_PARITY.md`.

## Two-spider and web-signal experiment

Add `--spider-pair --random-prey` to the feeding-mode command. Mr and Ms Spider
share fixed neural weights, but own separate GRU hidden states, local motor
state, hunger, meal commitments and reproductive cooldowns. Ms Spider is a
simulation resident cloned from the current species template, with her own
random valid starting position; the authored game scene is unchanged.

During mate search, a spider touching silk can spend one action every five
search turns sending a patterned courtship vibration. For twelve seconds,
another web-sensing spider touching the same connected intact web network
can sense that pattern, coarse bearing/range, and the sender's advertised
readiness. Breaking silk, losing contact or expiry removes the signal. The
signal offers no remote coordinates or exact hunger/HP. Following a signal
still uses local movement; acceptance requires visual contact at close range.
These are experiment-side rules using the game's existing web geometry and
contact propagation, not new learned neural capabilities. Hungry spiders do
not signal readiness. Visible conspecifics and recognized signaling contacts
are excluded from the hunting actor's prey contacts.

Random spawning rejects floor footprints outside rooms, local obstacles and
occupied locations. Whole-scene background rock is carved out by the authored
room polygons, matching the engine. Placement does not guarantee that every
random spawn has a traversable route to every other room.

### Meal interruption trial

In the seed-42, 240-round paired run, Mr Spider switched from feeding to
defense on turn 23, defeated the Shadowbeast on turn 34, and resumed wrapping
and feeding on turn 35. Accepted courtship occurred at 336 simulated seconds.
Both spiders survived. This is one trial, not a general performance claim.
Damage feedback carries an opaque attacker token only when the victim could
see the attacker; unidentified damage causes an alert pause rather than an
attack against an arbitrary bystander. Misses do not trigger interruption.

## Batch evidence for defense choices

See [the meal-defense comparison](experiments/MEAL_DEFENSE_COMPARISON.md) for
4,608 controlled episodes and 24 ordinary ecology runs. Observed withdrawal
is useful evidence for returning to a meal, but does not guarantee an attacker
will stay away. These results compare authored alternatives; no neural weights
have been updated and the live defense behavior is unchanged.
