# Driftmoor Giant Spider AI

A research project to implement intelligent giant spiders in the Driftmoor world using Yann LeCun's V-JEPA 2 world model technology.

## Overview

This project explores using V-JEPA 2 (Video Joint Embedding Predictive Architecture 2) to create intelligent NPC AI that can:
- Perceive and understand the Driftmoor environment
- Predict NPC movements and behaviors
- Make autonomous decisions about hunting and reproduction
- Adapt to new environments without retraining

## Why V-JEPA 2?

- **Zero-shot robot control**: Can adapt to different control schemes
- **World model capabilities**: Understands physics and predicts outcomes
- **Visual understanding**: State-of-the-art video understanding from Meta FAIR

## Project Structure

```
driftmoor-spider-ai/
├── src/              # Main source code
├── notebooks/        # Analysis and experimentation notebooks
├── data/             # Training data and assets
├── vjepa2/           # V-JEPA 2 repository (submodule)
├── README.md         # This file
└── requirements.txt  # Python dependencies
```

## Setup

```bash
cd ~/Python/driftmoor-spider-ai
pip install -r requirements.txt
```

## Project Goals

1. **Phase 1**: Integrate V-JEPA 2 for visual perception of NPCs
2. **Phase 2**: Add prediction layer for NPC movements
3. **Phase 3**: Implement "spider brain" decision-making
4. **Phase 4**: Connect to Driftmoor game engine

## Research Questions

- How well can V-JEPA 2 understand game environments vs real-world video?
- Can zero-shot robot control be adapted to game NPC control?
- What modifications are needed for spider-like movement?
- How to implement reproduction mechanics?

## Status

**Current Phase**: Project initialization

- [x] Project repository created
- [x] V-JEPA 2 repository cloned
- [ ] Set up project structure
- [ ] Create integration layer
- [ ] Test with sample Driftmoor environment
