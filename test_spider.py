#!/usr/bin/env python3
"""
Simplified test script for Driftmoor Giant Spider AI

Tests the basic functionality without torch dependencies.
"""

import sys
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, 'src')

# Add src directory to Python path
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


def test_action_executor():
    """Test the action executor"""
    print("=" * 50)
    print("Testing Action Executor")
    print("=" * 50)
    
    # Import action_executor
    from action_executor import ActionExecutor, SpiderAction
    
    executor = ActionExecutor()
    print(f"Available actions: {executor.available_actions}")
    
    # Test movement planning
    start_pos = (0.0, 0.0)
    target_pos = (100.0, 100.0)
    
    print(f"\nPlanning movement from {start_pos} to {target_pos}")
    actions = executor.plan_movement(start_pos, target_pos)
    print(f"Number of actions: {len(actions)}")
    if actions:
        print(f"First action: {actions[0].action_type}")
        print(f"First action target: {actions[0].target_position}")


def test_memory_system():
    """Test the memory system"""
    print("\n" + "=" * 50)
    print("Testing Memory System")
    print("=" * 50)
    
    from memory import MemorySystem, NPCInteraction
    
    memory = MemorySystem(max_memory_size=100)
    print(f"Memory system initialized, max_size: {memory.max_memory_size}")
    
    # Simulate some interactions
    interaction = NPCInteraction(
        npc_id=101,
        timestamp=1234567890.0,
        action="chased",
        outcome="success",
        location=(100.0, 200.0)
    )
    memory.record_interaction(interaction)
    print("Recorded interaction: NPC 101 chased successfully")
    
    # Get NPC pattern
    pattern = memory.get_npc_pattern(101)
    print(f"Pattern for NPC 101: {pattern}")


def test_spider_state():
    """Test the spider state dataclass"""
    print("\n" + "=" * 50)
    print("Testing Spider State")
    print("=" * 50)
    
    from spider_brain import SpiderState, SpiderBrain
    
    state = SpiderState(
        position=(100.0, 200.0),
        health=80.0,
        hunger=50.0,
        energy=90.0
    )
    print(f"Spider state: {state}")
    
    # Create a spider
    spider = SpiderBrain(spider_id=1)
    print(f"\nSpider {spider.spider_id} initialized")
    print(f"Status: {spider.get_status()}")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Driftmoor Giant Spider AI - Test Suite")
    print("Running tests WITHOUT torch (simplified)")
    print("=" * 50)
    
    try:
        test_action_executor()
        test_memory_system()
        test_spider_state()
        
        print("\n" + "=" * 50)
        print("All tests completed!")
        print("=" * 50)
        print("\nNote: Full integration with V-JEPA 2 requires torch.")
        print("Install with: pip install torch torchvision")
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
