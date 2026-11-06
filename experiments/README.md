# Echoing Caverns spider experiment

Driftmoor runs from the isolated checkout on `codex/spider-world-model`.
The original main checkout and the old toy simulation are not used for changes.

## Run

From ~/Python/driftmoor-spider-ai, with Driftmoor dependencies available:

```bash
python3 experiments/run_cavern_baseline.py \
  --driftmoor-root ../Driftmoor-spider-sim \
  --mode encounters --seeds 5 --output /tmp/cavern-encounters.jsonl
```

Modes:
- baseline: seven prescribed placements using existing session perception/activation.
- resolved: eight resolved movement/wait transitions.
- encounters: seven episodes per seed; five seeds produce 260 transitions.

The launcher isolates Driftmoor imports in a subprocess. No LLM requests or
model downloads occur. Output is JSONL.

## Mechanics

The actual scene loader merges PLACE residents with scene players.
Movement uses existing speed clamping, home restriction, boundary and
impassable checks. Each sequential action slot is six seconds with a combat
movement budget; this is not the full narrated session or initiative loop.
The secret passage is assumed open and other party members stay at their
initial positions. Geometry is still 2.5D.

Stored z is floor-relative. Scene-local elevation for distance calculations
is floor_z_at(x,y) + z. Structured sensory ranges use 3D separation with
existing horizontal occlusion and nonvisual range rules. Contacts include
tracking ID, modality, rounded range, horizontal bearing and vertical angle.
Unobserved coordinates/health are not supplied to the observer.

Body geometry has width, depth and height. Local +x is forward/depth, +y is
left/width, +z is up. The anchor is the center of the local support face
(normally between the feet). Rotation is Rz(heading) Ry(pitch) Rx(roll);
0 heading points along +x, 90 along +y. Angles are right-handed degrees.
Serialized orientation_roll_pitch_heading_deg is ordered roll, pitch, heading.
Interpret the pose intrinsically: establish heading about scene vertical,
pitch about the headed lateral axis, then roll about the pitched forward
axis. Positive pitch points the nose downward with these right-handed axes.
Angles describe absolute pose, not incremental world-axis rotations.
Euler heading/roll become coupled at pitch +/-90; a future continuous
orientation controller may use quaternions internally.

Gil's IDENTITY.md supplies provisional dimensions: width 2 ft, depth 1 ft,
height 5.5 ft. These are editable experiment values, not inferred racial
measurements. The spider uses an explicitly labeled 4 x 4 x 2 ft proxy.
Profiles and provenance are recorded. Ceiling pose starts at roll=180.

Bite and spinneret have distinct local origins and opposite directions.
They transform with body pose. A bite aims at the target oriented box's
closest point and checks a 60-degree half-angle cone and weapon reach.
This conservative aim-point rule is not an exact cone/box intersection solver.
Spinneret geometry is defined; web production/damage is not implemented.

Stationary attacks use the real CombatResolver rolls and HP damage, with
auto-approach disabled. Range uses fang-to-target-volume separation, not
anchor-to-anchor distance. The existing 0.5-ft melee tolerance is retained.
Per-experiment RNG is temporarily installed and restored after combat.
Defeated targets leave the active set. An orient action consumes a slot
without translating the anchor.

A target may be reachable from above but outside the current bite arc.
Seven scripted scenarios include retreat before entry, waiting, attacking,
offsetting the target, a grounded control, turning away, and aiming downward.
The grounded control is an initialization fixture, not an executed descent.

## Limitations and training boundary

No learned policy/predictor is implemented. No drop/climb dynamics, attachment
validation, body collisions, full 3D occlusion, poison saves, web restraint,
spells, ammunition consumption or neurochemical effects are applied.
Orientation can be set without simulating how a supported body achieves it.
Planar movement budget is retained on slopes; rounding may add about 0.071 ft.

Legacy full-session callers still use point/planar distances unless they
explicitly supply height/body-contact data. Scene activation and spider
behavior prompts remain unchanged.

Records separate observer_before/after, action, outcome, baseline diagnostics,
and privileged ground_truth. Combat geometry/debug fields include exact aim
points and must not be given to a policy as sensory input. Player commands
are exogenous interventions, not the spider's actions. Do not train on player
destinations, episode labels, seeds or privileged diagnostics. Split by whole
encounter; repeated dice seeds alone do not provide new sensory trajectories.
This small fixture verifies integration, not learning effectiveness.

## Tests

From ~/Python/Driftmoor-spider-sim:

```bash
python3 -m pytest -q tests/test_body_geometry.py tests/test_spider_experiment.py \
  tests/test_activation.py tests/test_senses.py tests/test_perception_items.py \
  tests/test_geometry_loader.py tests/test_combat_resolver.py
```

Tests cover geometry transforms, body-surface reach, direction, elevations,
actual combat damage, visibility, activation, movement/collision, stable
records, deterministic encounters and RNG isolation.
