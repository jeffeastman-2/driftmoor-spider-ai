# src/README.md - Driftmoor Giant Spider AI Source Code

## Overview

This directory contains the core source code for the Driftmoor Giant Spider AI project,
which implements intelligent NPC behavior using Yann LeCun's V-JEPA 2 world model.

## Architecture

```
src/
├── vjepa_integration.py   # V-JEPA 2 integration layer (NEW)
├── spider_brain.py        # Main AI brain orchestration
├── action_executor.py     # Action execution interface
├── perception.py          # Visual perception (simplified)
├── prediction.py          # NPC movement prediction
├── memory.py              # Experience memory and learning
└── __init__.py            # Package initialization
```

## Integration Approach

### V-JEPA 2 World Model

The V-JEPA 2 integration (`vjepa_integration.py`) provides:

1. **VJEPAModel**: PyTorch wrapper around V-JEPA 2 encoder-predictor pair
   - Loads pre-trained models from Meta FAIR
   - Extracts visual features from game frames
   - Predicts future states using learned world model

2. **VisualPerception**: NPC detection and tracking system
   - Processes game frames to detect NPCs
   - Tracks NPC movements across frames
   - Extracts features for decision-making

3. **WorldModel**: Maintains internal world state and predictions
   - Integrates observations over time
   - Predicts future NPC positions
   - Stores learned world models for adaptation

### Driftmoor Spider AI Components

**spider_brain.py** - Main orchestration:
- Coordinates perception, prediction, memory, and action
- Implements spider state management (health, hunger, energy)
- Makes decisions based on priorities (hunt, rest, wander, mate)

**action_executor.py** - Action execution:
- Translates AI decisions to game actions
- Plans movement paths
- Selects targets for hunting

**perception.py** - Visual perception (current implementation):
- Basic frame analysis
- NPC detection from features
- Movement prediction

**prediction.py** - Movement prediction:
- Predicts NPC trajectories
- Handles multi-frame tracking

**memory.py** - Experience memory:
- Stores past interactions
- Learns NPC behavioral patterns
- Adapts to new environments

## Usage

### Basic Integration

```python
from src.vjepa_integration import create_perception_system

# Create perception system
perception = create_perception_system(device='cuda')

# Process game frames
frame = get_frame_from_driftmoor()  # Your game frame
result = perception.process_frame(frame)

# Get NPC detections
npcs = result['npcs']

# Get features for decision-making
features = result['features']
```

### Using World Model

```python
from src.vjepa_integration import WorldModel, VisualPerception

# Initialize
perception = VisualPerception(device='cuda')
world_model = WorldModel(perception)

# Update with game frames
for frame in game_frames:
    state = world_model.update(frame)
    
# Get predictions
future = world_model.predict_future(steps=10)
```

### Integration with Spider Brain

```python
from src.spider_brain import SpiderBrain
from src.vjepa_integration import VisualPerception

# Initialize components
perception = VisualPerception(device='cuda')
spider = SpiderBrain(spider_id=1)

# Main game loop
while game_running:
    frame = get_frame_from_driftmoor()
    
    # Process frame
    result = perception.process_frame(frame)
    
    # Update spider brain
    game_state = {
        'position': spider.state.position,
        'health': spider.state.health,
        'hunger': spider.state.hunger,
        'energy': spider.state.energy,
        'detected_npcs': result['npcs']
    }
    
    # Get actions
    actions = spider.update(game_state)
    
    # Execute actions
    for action in actions:
        spider.executor.execute_action(action)
```

## Model Variants

V-JEPA 2 offers multiple model variants:

| Model | Parameters | Resolution | Use Case |
|-------|-----------|------------|----------|
| vjepa2_vit_large | 300M | 256x256 | Balanced performance |
| vjepa2_vit_huge | 600M | 256x256 | Higher accuracy |
| vjepa2_vit_giant | 1B | 256x256 | Best performance |
| vjepa2_1_vit_large_384 | 300M | 384x384 | V-JEPA 2.1, higher res |
| vjepa2_ac_vit_giant | 1B | 256x256 | Action-conditioned |

### Loading Different Models

```python
# High performance (recommended)
perception = VisualPerception(model_type='vjepa2_vit_giant')

# High resolution
perception = VisualPerception(model_type='vjepa2_1_vit_large_384')

# Action-conditioned (for future planning)
perception = VisualPerception(model_type='vjepa2_ac_vit_giant')
```

## File Reference

### vjepa_integration.py

**VJEPAModel**
- `load_model(force=False)` - Load V-JEPA 2 weights
- `extract_features(frames)` - Extract visual features
- `predict_future(frames, horizon)` - Predict future states

**VisualPerception**
- `process_frame(frame)` - Process single frame
- `process_sequence(frames)` - Process frame sequence
- `track_npcs(frames)` - Track NPCs across frames

**WorldModel**
- `update(frame)` - Update world state
- `predict_future(steps)` - Predict future state

### spider_brain.py

**SpiderBrain**
- `update(game_state)` - Main update loop
- `_decide(npcs)` - Decision making
- `_hunting_actions(npcs)` - Hunting behavior
- `_mate_actions(npcs)` - Mating behavior
- `_wander_actions()` - Wandering behavior

### action_executor.py

**ActionExecutor**
- `execute_action(action)` - Execute spider action
- `plan_movement(start, target)` - Plan movement path
- `select_target(npcs, state)` - Select hunting target

### memory.py

**MemorySystem**
- `record_interaction(interaction)` - Store interaction
- `get_npc_pattern(npc_id)` - Get learned NPC pattern
- `get_similar_interactions(interaction)` - Find similar cases

## Testing

Run the test suite:

```bash
python -m src.vjepa_integration
python -m src.spider_brain
```

## Next Steps

1. **Phase 1** - ✅ V-JEPA 2 integration complete
2. **Phase 2** - Connect to Driftmoor game engine
3. **Phase 3** - Train NPC detection head on game frames
4. **Phase 4** - Implement world model adaptation

## References

- V-JEPA 2 Paper: https://arxiv.org/abs/2506.09985
- V-JEPA 2 Code: https://github.com/facebookresearch/vjepa2
- Driftmoor Engine: https://github.com/windwardinc/driftmoor
