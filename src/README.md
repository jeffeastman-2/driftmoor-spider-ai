# Driftmoor Giant Spider AI - Core Modules

## Project Structure

```
src/
├── __init__.py          # Package initialization
├── spider_brain.py      # Main spider AI logic
├── perception.py        # V-JEPA 2 visual perception wrapper
├── prediction.py        # NPC movement prediction
├── action_executor.py   # Game control interface
└── memory.py            # Memory system for tracking NPCs
```

## Setup

```bash
cd driftmoor-spider-ai
pip install -r requirements.txt
```

## Quick Start

```bash
python -m src.spider_brain
```

## Project Goals

1. **Phase 1**: Integrate V-JEPA 2 for visual perception
2. **Phase 2**: Add prediction layer for NPC movements  
3. **Phase 3**: Implement "spider brain" decision-making
4. **Phase 4**: Connect to Driftmoor game engine

## Status

**Current Phase**: Project initialization

- [x] Project repository created
- [x] V-JEPA 2 repository cloned
- [x] Create README
- [x] Create requirements.txt
- [ ] Create src module structure
- [ ] Create integration layer
- [ ] Test with sample Driftmoor environment
