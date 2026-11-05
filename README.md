# Driftmoor Giant Spider AI - Documentation

## Overview

This project implements a deterministic spider NPC simulation for the Driftmoor game world. The spiders use intelligent behavior for hunting, movement, and energy management.

## Current State

### Implemented Features ✅

#### Deterministic Simulation
- All behavior is deterministic (no random elements in core logic)
- Movement respects speed limits
- Attack range checks work correctly
- Energy management priorities are enforced

#### Spider Brain (`src/spider_brain.py`)
- `_decide()`: Prioritizes actions correctly
  1. Rest when energy < 30 (even if hungry)
  2. Hunt when hunger > 40 and energy is normal
  3. Wander as default behavior

#### Movement System
- `_apply_movement()`: Enforces speed limit of 5 units per timestep
- `plan_movement()`: Plans path toward target
- Movement actions respect distance constraints

#### Hunting System
- `select_target()`: Finds closest living NPC
- `_hunting_actions()`: Generates movement + attack sequence
- Attack range: 50 units (must be within range to attack)

#### Dead Target Handling
- Dead NPCs (health <= 0) are ignored
- Only living NPCs are considered for hunting

#### NPC Consistency
- Stable NPC IDs persist across timesteps
- Position changes tracked for history

### Not Yet Implemented ⏳

- V-JEPA 2 visual perception (requires manual download of ~4.8GB checkpoint)
- Real-time simulation integration with Driftmoor game engine
- Complex terrain navigation
- Predator/prey dynamics
- Social behaviors (mating, aggregation)

## Running the Tests

```bash
cd /Users/jeff/Python/driftmoor-spider-ai/src
python3 ../tests/test_spider_deterministic.py
```

### Test Results

```
Ran 10 tests in 0.025s

OK
```

All tests pass:
1. `test_movement_respects_speed_limit` - Movement respects speed limit
2. `test_distant_target_cannot_be_attacked_immediately` - Far targets require movement first
3. `test_nearby_target_can_be_attacked` - Nearby targets can be attacked
4. `test_dead_targets_are_ignored` - Dead NPCs are filtered
5. `test_hunting_ignores_dead_npcs` - Hunting ignores dead targets
6. `test_critically_low_energy_causes_rest` - Rest priority over hunger
7. `test_normal_energy_allows_hunting` - Normal energy allows hunting
8. `test_one_npc_retains_history_as_position_changes` - NPC ID consistency
9. `test_spider_approaches_and_attacks_stationary_target` - Full simulation
10. `test_same_scenario_gives_same_results` - Deterministic behavior

## Example: Spider Approaches Target

```python
Simulation Results:
  Start position: (0.0, 0.0)
  Target position: (200.0, 0.0)
  Final position: (150.0, 0.0)
  Elapsed time: 31.0 seconds
  Steps taken: 31
  Final distance to target: 50.0 units
  Actions: ['move', 'move', 'move', ..., 'attack']
```

The spider:
1. Starts at (0, 0)
2. Moves toward target at (200, 0) at 5 units/timestep
3. After 30 steps, reaches (150, 0) - within attack range (50 units)
4. Attacks the target

## Files

| File | Purpose |
|------|---------|
| `src/spider_brain.py` | Main spider AI logic |
| `src/action_executor.py` | Action execution with movement enforcement |
| `src/perception.py` | Visual perception (mock - requires V-JEPA 2) |
| `src/prediction.py` | Movement prediction |
| `src/memory.py` | Memory system |
| `tests/test_spider_deterministic.py` | Automated tests |
| `tests/test_defect_detection.py` | Defect demonstration tests |

## Known Limitations

1. **No Driftmoor integration**: Current implementation is standalone; requires game engine interface
2. **V-JEPA 2 not included**: The ~4.8GB model checkpoint must be downloaded separately
3. **2D world only**: Current simulation uses 2D coordinates
4. **Simplified terrain**: No obstacle avoidance or complex terrain handling

## Future Work

- Integrate with Driftmoor game engine
- Add V-JEPA 2 visual perception
- Implement 3D world coordinates
- Add terrain navigation
- Implement complex social behaviors