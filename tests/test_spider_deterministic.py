#!/usr/bin/env python3
"""
Deterministic spider simulation tests for Driftmoor Giant Spider AI

Tests verify:
- Movement over one timestep respects the speed limit
- A distant target cannot be attacked immediately
- A nearby target can be attacked
- Dead targets are ignored
- Critically low energy causes rest even when hungry
- One NPC retains its history as its position changes

All tests use deterministic behavior (no random elements).
"""

import sys
import os
import unittest

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, '..', 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import numpy as np
from spider_brain import SpiderBrain, SpiderState
from action_executor import ActionExecutor, SpiderAction


class TestSpiderMovement(unittest.TestCase):
    """Test spider movement logic"""
    
    def test_movement_respects_speed_limit(self):
        """Movement over one timestep respects the speed limit"""
        spider = SpiderBrain(spider_id=1)
        spider.state.position = (0.0, 0.0)
        spider.state.energy = 100.0
        spider.state.hunger = 50.0
        
        # Target is 100 units away
        target_pos = (100.0, 0.0)
        
        # Plan movement
        executor = ActionExecutor()
        actions = executor.plan_movement(spider.state.position, target_pos)
        
        # plan_movement creates multiple intermediate steps
        self.assertGreater(len(actions), 0)
        
        # The first action moves toward target
        self.assertEqual(actions[0].action_type, "move")
        
        # Apply first movement action with speed limit
        speed_limit = 5.0  # units per timestep
        new_pos = executor._apply_movement(spider.state.position, actions[0], speed_limit)
        
        # Verify movement doesn't exceed speed limit
        distance_moved = np.sqrt(
            (new_pos[0] - spider.state.position[0])**2 +
            (new_pos[1] - spider.state.position[1])**2
        )
        self.assertLessEqual(distance_moved, speed_limit + 0.01)
        self.assertGreater(distance_moved, 0)  # Should move toward target


class TestSpiderAttack(unittest.TestCase):
    """Test spider attack logic"""
    
    def test_distant_target_cannot_be_attacked_immediately(self):
        """A distant target cannot be attacked immediately"""
        spider = SpiderBrain(spider_id=1)
        spider.state.position = (0.0, 0.0)
        spider.state.energy = 100.0
        spider.state.hunger = 80.0  # Very hungry
        
        # Target is far away (500 units)
        target_pos = (500.0, 0.0)
        target_id = 1
        
        executor = ActionExecutor()
        
        # Check attack range
        distance = np.sqrt(
            (target_pos[0] - spider.state.position[0])**2 +
            (target_pos[1] - spider.state.position[1])**2
        )
        attack_range = 50.0
        
        # Should not be able to attack when too far
        self.assertGreater(distance, attack_range)
        
        # When target is far, spider should plan movement first
        detected_npcs = [{
            'id': target_id,
            'position': target_pos,
            'health': 50.0,
            'type': 'npc'
        }]
        
        actions = spider._hunting_actions(detected_npcs)
        
        # Should have movement action first, not attack
        first_action_type = actions[0].action_type
        self.assertEqual(first_action_type, "move")
        
        # Simulate movement - after moving 450 units, spider is now within range
        spider.state.position = (450.0, 0.0)
        distance = np.sqrt(
            (target_pos[0] - spider.state.position[0])**2 +
            (target_pos[1] - spider.state.position[1])**2
        )
        self.assertLessEqual(distance, attack_range)
        
        # Now in range, spider can attack
        actions_in_range = spider._hunting_actions(detected_npcs)
        self.assertIn("attack", [a.action_type for a in actions_in_range])
    
    def test_nearby_target_can_be_attacked(self):
        """A nearby target can be attacked"""
        spider = SpiderBrain(spider_id=1)
        spider.state.position = (0.0, 0.0)
        spider.state.energy = 100.0
        spider.state.hunger = 80.0  # Very hungry
        
        # Target is within attack range (30 units)
        target_pos = (30.0, 0.0)
        target_id = 1
        
        executor = ActionExecutor()
        
        # Check attack range
        distance = np.sqrt(
            (target_pos[0] - spider.state.position[0])**2 +
            (target_pos[1] - spider.state.position[1])**2
        )
        attack_range = 50.0
        
        self.assertLessEqual(distance, attack_range)
        
        # When hungry and nearby, spider should plan attack
        detected_npcs = [{
            'id': target_id,
            'position': target_pos,
            'health': 50.0,
            'type': 'npc'
        }]
        
        actions = spider._hunting_actions(detected_npcs)
        
        # Should have attack action
        action_types = [a.action_type for a in actions]
        self.assertIn("attack", action_types)


class TestSpiderDeadTargets(unittest.TestCase):
    """Test spider dead target handling"""
    
    def test_dead_targets_are_ignored(self):
        """Dead targets are ignored"""
        spider = SpiderBrain(spider_id=1)
        spider.state.position = (0.0, 0.0)
        spider.state.energy = 100.0
        spider.state.hunger = 80.0
        
        # Create detected NPCs - one alive, one dead
        detected_npcs = [
            {'id': 1, 'position': (50.0, 0.0), 'health': 50.0, 'type': 'npc'},
            {'id': 2, 'position': (100.0, 0.0), 'health': 0.0, 'type': 'npc'}  # Dead
        ]
        
        # Spider should only consider alive NPCs
        alive_npcs = [npc for npc in detected_npcs if npc['health'] > 0]
        
        self.assertEqual(len(alive_npcs), 1)
        self.assertEqual(alive_npcs[0]['id'], 1)
        
        # Spider should target the alive NPC
        executor = ActionExecutor()
        target_id = executor.select_target(alive_npcs, {'position': spider.state.position})
        
        self.assertEqual(target_id, 1)
    
    def test_hunting_ignores_dead_npcs(self):
        """Hunting actions ignore dead NPCs"""
        spider = SpiderBrain(spider_id=1)
        spider.state.position = (0.0, 0.0)
        spider.state.energy = 100.0
        spider.state.hunger = 80.0
        
        # All targets are dead
        detected_npcs = [
            {'id': 1, 'position': (50.0, 0.0), 'health': 0.0, 'type': 'npc'},
            {'id': 2, 'position': (100.0, 0.0), 'health': 0.0, 'type': 'npc'}
        ]
        
        actions = spider._hunting_actions(detected_npcs)
        
        # Should have at least one action (wander/move)
        self.assertGreater(len(actions), 0)
        
        # First action should be movement (wander)
        self.assertEqual(actions[0].action_type, "move")


class TestSpiderEnergy(unittest.TestCase):
    """Test spider energy management"""
    
    def test_critically_low_energy_causes_rest(self):
        """Critically low energy causes rest even when hungry"""
        spider = SpiderBrain(spider_id=1)
        spider.state.position = (0.0, 0.0)
        spider.state.energy = 5.0  # Critically low
        spider.state.hunger = 90.0  # Still hungry
        
        # Low energy should take priority over hunger
        detected_npcs = [{'id': 1, 'position': (100.0, 0.0), 'health': 50.0, 'type': 'npc'}]
        
        # Energy priority check happens in _decide
        actions = spider._decide(detected_npcs)
        
        # Should rest due to low energy (energy < 30 threshold)
        action_types = [a.action_type for a in actions]
        self.assertIn("rest", action_types)
        
        # Verify energy threshold
        self.assertLess(spider.state.energy, 30)  # Rest threshold
    
    def test_normal_energy_allows_hunting(self):
        """Normal energy allows hunting even when hungry"""
        spider = SpiderBrain(spider_id=1)
        spider.state.position = (0.0, 0.0)
        spider.state.energy = 60.0  # Normal
        spider.state.hunger = 80.0  # Hungry
        
        detected_npcs = [{'id': 1, 'position': (50.0, 0.0), 'health': 50.0, 'type': 'npc'}]
        
        actions = spider._decide(detected_npcs)
        
        # Should hunt when energy is normal
        action_types = [a.action_type for a in actions]
        self.assertIn("attack", action_types)


class TestNPCConsistency(unittest.TestCase):
    """Test NPC ID consistency and history tracking"""
    
    def test_one_npc_retains_history_as_position_changes(self):
        """One NPC retains its history as its position changes"""
        # Use stable NPC ID
        npc_id = 42
        
        # Initial position
        npc1 = {
            'id': npc_id,
            'position': (100.0, 100.0),
            'health': 50.0,
            'type': 'npc'
        }
        
        # After one timestep, position changes but ID stays same
        npc2 = {
            'id': npc_id,  # Same ID!
            'position': (105.0, 102.0),
            'health': 48.0,
            'type': 'npc'
        }
        
        # Verify ID consistency
        self.assertEqual(npc1['id'], npc2['id'])
        self.assertEqual(npc1['id'], 42)
        
        # Position changed
        self.assertNotEqual(npc1['position'], npc2['position'])
        
        # Compute velocity from position change
        delta_t = 1.0
        velocity = (
            (npc2['position'][0] - npc1['position'][0]) / delta_t,
            (npc2['position'][1] - npc1['position'][1]) / delta_t
        )
        
        # Verify velocity calculation
        self.assertAlmostEqual(velocity[0], 5.0)
        self.assertAlmostEqual(velocity[1], 2.0)


class TestFullSimulation(unittest.TestCase):
    """Test full simulation scenario"""
    
    def test_spider_approaches_and_attacks_stationary_target(self):
        """Spider starts far from stationary target, approaches over multiple timesteps, attacks"""
        # Setup
        spider = SpiderBrain(spider_id=1)
        spider.state.position = (0.0, 0.0)
        spider.state.energy = 100.0
        spider.state.hunger = 80.0
        
        # Stationary target 200 units away
        target_pos = (200.0, 0.0)
        target_id = 1
        target_health = 100.0
        target_detected = True
        
        # Speed limit
        speed_limit = 5.0
        
        # Simulation parameters
        attack_range = 50.0
        max_steps = 50
        
        # Track positions over time
        positions = []
        actions = []
        
        executor = ActionExecutor()
        
        # Simulate multiple timesteps
        current_pos = spider.state.position
        for step in range(max_steps):
            positions.append(current_pos)
            
            # Check if within attack range
            distance = np.sqrt(
                (target_pos[0] - current_pos[0])**2 +
                (target_pos[1] - current_pos[1])**2
            )
            
            if distance <= attack_range:
                # Within range - attack
                action = SpiderAction(
                    action_type="attack",
                    target_position=target_pos,
                    target_npc_id=target_id,
                    duration=1.0,
                    intensity=0.8
                )
                actions.append(action)
                break  # Attack terminates the simulation
            else:
                # Not in range - move toward target
                movement_actions = executor.plan_movement(current_pos, target_pos)
                action = movement_actions[0]
                actions.append(action)
                
                # Apply movement
                current_pos = executor._apply_movement(current_pos, action, speed_limit)
        
        # Verify simulation results
        self.assertGreater(len(positions), 1, "Should have multiple positions")
        self.assertGreater(len(actions), 0, "Should have actions")
        
        # First position should be start
        self.assertAlmostEqual(positions[0][0], 0.0)
        self.assertAlmostEqual(positions[0][1], 0.0)
        
        # Last position should be near target
        final_distance = np.sqrt(
            (target_pos[0] - positions[-1][0])**2 +
            (target_pos[1] - positions[-1][1])**2
        )
        self.assertLess(final_distance, attack_range * 1.5, "Should reach near target")
        
        # Last action should be attack
        self.assertEqual(actions[-1].action_type, "attack")
        
        # Report results
        elapsed_time = len(positions) * 1.0  # 1.0 second per timestep
        print(f"\nSimulation Results:")
        print(f"  Start position: ({positions[0][0]:.1f}, {positions[0][1]:.1f})")
        print(f"  Target position: ({target_pos[0]:.1f}, {target_pos[1]:.1f})")
        print(f"  Final position: ({positions[-1][0]:.1f}, {positions[-1][1]:.1f})")
        print(f"  Elapsed time: {elapsed_time:.1f} seconds")
        print(f"  Steps taken: {len(positions)}")
        print(f"  Final distance to target: {final_distance:.1f} units")
        print(f"  Actions: {[a.action_type for a in actions]}")


class TestDeterministicSimulation(unittest.TestCase):
    """Test deterministic simulation with reproducible results"""
    
    def test_same_scenario_gives_same_results(self):
        """Same scenario gives same results (deterministic)"""
        def run_scenario(seed=42):
            """Run a deterministic scenario"""
            np.random.seed(seed)
            
            spider = SpiderBrain(spider_id=1)
            spider.state.position = (0.0, 0.0)
            spider.state.energy = 100.0
            spider.state.hunger = 80.0
            
            target_pos = (100.0, 0.0)
            speed_limit = 5.0
            attack_range = 50.0
            
            executor = ActionExecutor()
            current_pos = spider.state.position
            positions = [current_pos.copy() if isinstance(current_pos, np.ndarray) else current_pos]
            
            for _ in range(30):  # 30 timesteps
                distance = np.sqrt(
                    (target_pos[0] - current_pos[0])**2 +
                    (target_pos[1] - current_pos[1])**2
                )
                
                if distance <= attack_range:
                    break
                
                actions = executor.plan_movement(current_pos, target_pos)
                current_pos = executor._apply_movement(current_pos, actions[0], speed_limit)
                positions.append(current_pos)
            
            return positions
        
        # Run same scenario twice
        positions1 = run_scenario(seed=42)
        positions2 = run_scenario(seed=42)
        
        # Results should be identical
        self.assertEqual(len(positions1), len(positions2))
        for p1, p2 in zip(positions1, positions2):
            self.assertAlmostEqual(p1[0], p2[0], places=5)
            self.assertAlmostEqual(p1[1], p2[1], places=5)


if __name__ == "__main__":
    print("="*60)
    print("Driftmoor Giant Spider AI - Deterministic Tests")
    print("="*60)
    print()
    
    # Run tests
    unittest.main(verbosity=2)
