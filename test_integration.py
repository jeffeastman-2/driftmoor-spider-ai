#!/usr/bin/env python3
"""
Test script for Driftmoor Giant Spider AI with V-JEPA 2 integration

Tests the integration of V-JEPA 2 world model for visual perception.
"""

import sys
import os
import numpy as np

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, 'src')

# Add src directory to Python path
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


def test_vjepa_integration():
    """Test V-JEPA 2 integration"""
    print("=" * 60)
    print("Testing V-JEPA 2 Integration")
    print("=" * 60)
    
    from vjepa_integration import VJEPAModel, VisualPerception
    
    # Test model loading (without pretrained weights for speed)
    print("\n1. Loading V-JEPA 2 model...")
    model = VJEPAModel(
        model_type='vjepa2_vit_large',
        device='cpu',
        pretrained=False  # Use False for faster testing
    )
    model.load_model()
    
    # Test feature extraction
    print("\n2. Testing feature extraction...")
    dummy_frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    features = model.extract_features([dummy_frame])
    print(f"   Features shape: {features.shape}")
    print(f"   Features dtype: {features.dtype}")
    print(f"   Features device: {features.device}")
    
    # Test visual perception system
    print("\n3. Testing visual perception system...")
    perception = VisualPerception(
        model_type='vjepa2_vit_large',
        device='cpu'
    )
    
    result = perception.process_frame(dummy_frame)
    print(f"   Detection results: {len(result['npcs'])} NPCs")
    print(f"   NPC IDs: {[npc['id'] for npc in result['npcs']]}")
    
    # Test world model
    print("\n4. Testing world model...")
    from vjepa_integration import WorldModel
    
    world_model = WorldModel(perception)
    
    # Update with multiple frames
    for i in range(5):
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        state = world_model.update(frame)
    
    print(f"   World state updated {state['timestamp']} times")
    
    # Test sequence processing
    print("\n5. Testing sequence processing...")
    dummy_frames = [
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        for _ in range(8)
    ]
    
    seq_result = perception.process_sequence(dummy_frames)
    print(f"   Sequence processed, length: {seq_result['length']}")
    print(f"   Features shape: {seq_result['features'].shape}")
    
    print("\n" + "=" * 60)
    print("V-JEPA Integration Tests Complete!")
    print("=" * 60)


def test_spider_brain():
    """Test the spider brain integration"""
    print("\n" + "=" * 60)
    print("Testing Spider Brain with V-JEPA Integration")
    print("=" * 60)
    
    from spider_brain import SpiderBrain, SpiderState
    from vjepa_integration import VisualPerception
    
    # Initialize spider brain
    print("\n1. Initializing spider brain...")
    spider = SpiderBrain(spider_id=1)
    print(f"   Spider {spider.spider_id} initialized")
    
    # Simulate game state with V-JEPA features
    print("\n2. Processing game state...")
    frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    perception = VisualPerception(device='cpu')
    result = perception.process_frame(frame)
    
    # Create game state with detections
    game_state = {
        'position': (100.0, 200.0),
        'health': 80.0,
        'hunger': 60.0,
        'energy': 90.0,
        'detected_npcs': result['npcs']
    }
    
    # Update spider
    actions = spider.update(game_state)
    
    print(f"\n3. Spider actions generated: {len(actions)}")
    for i, action in enumerate(actions):
        print(f"   Action {i+1}: {action.action_type} -> {action.target_position}")
    
    # Get spider status
    status = spider.get_status()
    print(f"\n4. Spider status:")
    print(f"   Position: {status['position']}")
    print(f"   Health: {status['health']}")
    print(f"   Hunger: {status['hunger']}")
    print(f"   Energy: {status['energy']}")
    
    print("\n" + "=" * 60)
    print("Spider Brain Tests Complete!")
    print("=" * 60)


def test_full_integration():
    """Test full integration pipeline"""
    print("\n" + "=" * 60)
    print("Testing Full Integration Pipeline")
    print("=" * 60)
    
    from spider_brain import SpiderBrain
    from vjepa_integration import VisualPerception, WorldModel
    
    # Initialize all components
    print("\n1. Initializing components...")
    perception = VisualPerception(device='cpu')
    world_model = WorldModel(perception)
    spider = SpiderBrain(spider_id=1)
    
    print("   ✓ Perception system initialized")
    print("   ✓ World model initialized")
    print("   ✓ Spider brain initialized")
    
    # Simulate game loop
    print("\n2. Running simulation loop...")
    for step in range(10):
        # Get frame from game
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Process frame through V-JEPA
        result = perception.process_frame(frame)
        
        # Update world model
        world_state = world_model.update(frame)
        
        # Get actions for spider
        game_state = {
            'position': spider.state.position,
            'health': spider.state.health,
            'hunger': spider.state.hunger,
            'energy': spider.state.energy,
            'detected_npcs': result['npcs']
        }
        
        actions = spider.update(game_state)
        
        print(f"   Step {step+1}: {len(actions)} actions, {len(result['npcs'])} NPCs detected")
    
    print("\n" + "=" * 60)
    print("Full Integration Tests Complete!")
    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Driftmoor Giant Spider AI - Full Test Suite")
    print("V-JEPA 2 Integration Tests")
    print("=" * 60)
    
    try:
        test_vjepa_integration()
        test_spider_brain()
        test_full_integration()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Connect to Driftmoor game engine for real frames")
        print("  2. Train NPC detection head on game frames")
        print("  3. Implement world model adaptation")
        print("  4. Run full simulations with V-JEPA 2 features")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
