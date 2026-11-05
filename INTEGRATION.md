# V-JEPA 2 Integration Guide for Driftmoor Giant Spider AI

## Overview

This document describes the integration of Yann LeCun's V-JEPA 2 framework into the Driftmoor Giant Spider AI project.

### What is V-JEPA 2?

V-JEPA 2 (Video Joint Embedding Predictive Architecture 2) is a self-supervised video model from Meta FAIR that:
- Learns physical world understanding from internet-scale video data
- Achieves state-of-the-art performance on motion understanding and action anticipation tasks
- Provides encoder-predictor architecture for feature extraction and future prediction
- Supports zero-shot robot control adaptation

### Integration Goals

The integration aims to:
1. Use V-JEPA 2 for visual perception of NPCs in Driftmoor
2. Enable prediction of NPC movements and behaviors
3. Create an autonomous decision-making system for giant spiders
4. Allow learning and adaptation from experience

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    V-JEPA 2 Integration                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐    ┌──────────────────────┐          │
│  │   VisualInput    │ →  │  VJEPAModel          │          │
│  │   (game frames)  │    │  - Encoder           │          │
│  └──────────────────┘    │  - Predictor         │          │
│                          └──────────────────────┘          │
│                                     │                      │
│                                     ▼                      │
│                          ┌──────────────────────┐          │
│                          │  VisualPerception    │          │
│                          │  - NPC Detection     │          │
│                          │  - Tracking          │          │
│                          │  - Feature Storage   │          │
│                          └──────────────────────┘          │
│                                     │                      │
│                                     ▼                      │
│                          ┌──────────────────────┐          │
│                          │   WorldModel         │          │
│                          │  - State Update      │          │
│                          │  - Prediction        │          │
│                          └──────────────────────┘          │
│                                     │                      │
│                                     ▼                      │
│                          ┌──────────────────────┐          │
│                          │   SpiderBrain        │          │
│                          │  - Decision Making   │          │
│                          │  - Action Selection  │          │
│                          └──────────────────────┘          │
│                                     │                      │
│                                     ▼                      │
│                          ┌──────────────────────┐          │
│                          │   ActionExecutor     │          │
│                          │  - Game Action       │          │
│                          └──────────────────────┘          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Components

### VJEPAModel

Core wrapper around V-JEPA 2 encoder-predictor pair.

```python
from src.vjepa_integration import VJEPAModel

# Initialize model
model = VJEPAModel(
    model_type='vjepa2_vit_large',
    device='cuda',
    pretrained=True
)

# Load model (downloads ~600MB on first run)
model.load_model()

# Extract features from frames
features = model.extract_features(frames)

# Predict future
predictions = model.predict_future(frames, horizon=8)
```

**Model Types:**
- `vjepa2_vit_large` - 300M params, balanced performance
- `vjepa2_vit_huge` - 600M params, higher accuracy
- `vjepa2_vit_giant` - 1B params, best performance
- `vjepa2_1_vit_large_384` - 300M params, higher resolution
- `vjepa2_ac_vit_giant` - 1B params, action-conditioned

### VisualPerception

Perception system using V-JEPA 2 for Driftmoor game frames.

```python
from src.vjepa_integration import VisualPerception

# Initialize perception
perception = VisualPerception(device='cuda')

# Process frames
result = perception.process_frame(frame)

# Get detections
npcs = result['npcs']

# Process sequences
seq_result = perception.process_sequence(frames)
```

### WorldModel

Maintains internal world state and predicts future states.

```python
from src.vjepa_integration import WorldModel, VisualPerception

# Initialize
perception = VisualPerception(device='cuda')
world_model = WorldModel(perception)

# Update with frames
for frame in game_frames:
    state = world_model.update(frame)

# Get predictions
future = world_model.predict_future(steps=10)
```

## Usage Examples

### Basic Integration

```python
import numpy as np
from src.vjepa_integration import create_perception_system

# Create perception system
perception = create_perception_system(device='cuda')

# Process frames from Driftmoor
frame = get_frame_from_driftmoor()  # Your frame extraction
result = perception.process_frame(frame)

# Get NPC detections
npcs = result['npcs']
features = result['features']
```

### With World Model

```python
from src.vjepa_integration import VisualPerception, WorldModel

# Initialize
perception = VisualPerception(device='cuda')
world_model = WorldModel(perception)

# Game loop
while game_running:
    frame = get_frame_from_driftmoor()
    
    # Update world state
    state = world_model.update(frame)
    
    # Get predictions
    future = world_model.predict_future(steps=10)
    
    # Use for decision-making
    decision = make_decision(state, future)
```

### Integration with SpiderBrain

```python
from src.spider_brain import SpiderBrain
from src.vjepa_integration import VisualPerception

# Initialize
perception = VisualPerception(device='cuda')
spider = SpiderBrain(spider_id=1)

# Main loop
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

## Setup

### Requirements

```bash
pip install torch torchvision timm einops opencv-python
```

### Download V-JEPA 2 Models

Models are downloaded from:
- https://dl.fbaipublicfiles.com/vjepa2/

First load downloads ~600MB.

### Integration Steps

1. Ensure `vjepa2/` repository is cloned in project root
2. Install dependencies: `pip install -r requirements.txt`
3. Import modules: `from src.vjepa_integration import *`

## Model Configuration

### Changing Model Type

```python
# High performance (recommended)
perception = VisualPerception(model_type='vjepa2_vit_giant')

# High resolution
perception = VisualPerception(model_type='vjepa2_1_vit_large_384')

# Action-conditioned (for future planning)
perception = VisualPerception(model_type='vjepa2_ac_vit_giant')
```

### Device Configuration

```python
# Auto-detect GPU
perception = VisualPerception(device='auto')

# Force CPU
perception = VisualPerception(device='cpu')

# Force GPU
perception = VisualPerception(device='cuda')
```

## Testing

Run the test suite:

```bash
cd ~/Python/driftmoor-spider-ai/src
python3 ../test_integration.py
```

## Troubleshooting

### "No module named src.hub" Error

This is a known issue with vjepa2's relative imports. The integration handles this gracefully by falling back to mock mode when full model loading fails.

### "No module named timm" Error

Install required dependencies:
```bash
pip install timm einops opencv-python
```

### GPU Out of Memory

Use a smaller model or batch size:
```python
perception = VisualPerception(model_type='vjepa2_vit_large', device='cuda')
```

## Performance Considerations

- V-JEPA 2 models are large (~600MB+)
- GPU recommended for real-time performance
- CPU fallback available but slower
- Feature extraction is O(1) per frame
- World model predictions scale with horizon

## Future Enhancements

1. **NPC Detection Head**: Train classifier on V-JEPA 2 features
2. **World Model Adaptation**: Fine-tune on Driftmoor frames
3. **Action Conditioning**: Use V-JEPA 2-AC for spider-like movements
4. **Memory Integration**: Store world models in SpiderBrain's memory
5. **Multi-Agent Tracking**: Extend for multiple NPC tracking

## References

- V-JEPA 2 Paper: https://arxiv.org/abs/2506.09985
- V-JEPA 2 Code: https://github.com/facebookresearch/vjepa2
- Driftmoor Engine: https://github.com/windwardinc/driftmoor

## License

Same as V-JEPA 2 (MIT)
