#!/usr/bin/env python3
"""
Main simulation script for Driftmoor Giant Spider AI

Demonstrates spider behavior including hunting, movement, and reproduction.
"""

import sys
import os
import time
import random

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, '..')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import numpy as np
from spider_brain import SpiderBrain, SpiderState
from perception import PerceptionSystem
from prediction import MovementPredictor, NPCState
from action_executor import ActionExecutor, SpiderAction
from memory import MemorySystem, NPCInteraction


class SimpleGameWorld:
    """Simple game world with NPCs for testing"""
    
    def __init__(self, width=800, height=600, num_npcs=5):
        self.width = width
        self.height = height
        self.npcs = []
        
        # Create NPCs
        for i in range(num_npcs):
            npc = {
                'id': i,
                'type': 'npc',
                'position': (
                    random.uniform(50, width - 50),
                    random.uniform(50, height - 50)
                ),
                'velocity': (
                    random.uniform(-2, 2),
                    random.uniform(-2, 2)
                ),
                'health': random.uniform(50, 100),
                'size': random.uniform(20, 40)
            }
            self.npcs.append(npc)
    
    def update(self):
        """Update game world state"""
        for npc in self.npcs:
            # Update position
            npc['position'] = (
                npc['position'][0] + npc['velocity'][0],
                npc['position'][1] + npc['velocity'][1]
            )
            
            # Bounce off walls
            if npc['position'][0] <= 0 or npc['position'][0] >= self.width:
                npc['velocity'] = (-npc['velocity'][0], npc['velocity'][1])
            if npc['position'][1] <= 0 or npc['position'][1] >= self.height:
                npc['velocity'] = (npc['velocity'][0], -npc['velocity'][1])
    
    def get_detected_npcs(self, spider_position):
        """Get NPCs visible to spider"""
        detected = []
        for npc in self.npcs:
            distance = np.sqrt(
                (npc['position'][0] - spider_position[0])**2 +
                (npc['position'][1] - spider_position[1])**2
            )
            if distance < 300:  # Detection radius
                detected.append(npc)
        return detected
    
    def render(self, spiders):
        """Render the game world"""
        print(f"\n{'='*50}")
        print("GAME WORLD STATE")
        print(f"{'='*50}")
        print(f"NPCs: {len(self.npcs)}")
        for npc in self.npcs:
            print(f"  NPC {npc['id']}: pos=({npc['position'][0]:.0f}, {npc['position'][1]:.0f}), "
                  f"health={npc['health']:.1f}")
        print(f"\nSpiders: {len(spiders)}")
        for spider in spiders:
            state = spider.get_status()
            print(f"  Spider {state['spider_id']}: pos=({state['position'][0]:.0f}, {state['position'][1]:.0f}), "
                  f"hunger={state['hunger']:.1f}, energy={state['energy']:.1f}, "
                  f"action={state['current_action']}")
        print(f"{'='*50}")


def create_spider_pair():
    """Create a pair of spiders for reproduction testing"""
    spider1 = SpiderBrain(spider_id=1)
    spider2 = SpiderBrain(spider_id=2)
    
    # Position them near each other
    spider1.state.position = (100.0, 100.0)
    spider2.state.position = (150.0, 100.0)
    
    return [spider1, spider2]


def run_simulation(steps=50, num_spiders=2, create_pair=False):
    """Run the spider simulation"""
    
    # Initialize world
    world = SimpleGameWorld(num_npcs=5)
    
    # Initialize spiders
    if create_pair:
        spiders = create_spider_pair()
    else:
        spiders = [SpiderBrain(spider_id=i) for i in range(num_spiders)]
        # Position spiders
        for i, spider in enumerate(spiders):
            spider.state.position = (100.0 + i * 50, 100.0 + i * 50)
    
    # Simulation loop
    for step in range(steps):
        print(f"\n[Step {step}]")
        
        # Update world
        world.update()
        
        # Get detected NPCs for each spider
        for spider in spiders:
            detected = world.get_detected_npcs(spider.state.position)
            
            # Update spider
            actions = spider.update({
                'position': spider.state.position,
                'health': spider.state.health,
                'hunger': spider.state.hunger,
                'energy': spider.state.energy,
                'detected_npcs': detected
            })
            
            # Update spider hunger/energy
            spider.state.hunger = max(0, spider.state.hunger - 0.5)  # Hunger increases over time
            spider.state.energy = max(0, spider.state.energy - 0.3)  # Energy decreases over time
            
            # Check for successful hunts
            if actions and any(a.action_type == 'attack' for a in actions):
                # Spider is attacking - reduce hunger
                spider.state.hunger = max(0, spider.state.hunger - 10)
                spider.state.health = min(100, spider.state.health + 2)  # Heal from eating
                spider.state.energy = min(100, spider.state.energy + 5)
        
        # Render world state every 5 steps
        if step % 5 == 0:
            world.render(spiders)
        
        # Check for reproduction opportunity
        if create_pair and step > 10:
            spider1, spider2 = spiders
            distance = np.sqrt(
                (spider1.state.position[0] - spider2.state.position[0])**2 +
                (spider1.state.position[1] - spider2.state.position[1])**2
            )
            
            # If spiders are close and both hungry, they might mate
            if distance < 30 and spider1.state.hunger < 60 and spider2.state.hunger < 60:
                print(f"\n*** REPRODUCTION EVENT! ***")
                print(f"Spiders {spider1.spider_id} and {spider2.spider_id} are mating!")
                # Create offspring
                offspring = SpiderBrain(spider_id=3)
                offspring.state.position = ((spider1.state.position[0] + spider2.state.position[0]) / 2,
                                           (spider1.state.position[1] + spider2.state.position[1]) / 2)
                offspring.state.health = 100.0
                offspring.state.hunger = 0.0  # Born full
                offspring.state.energy = 100.0
                spiders.append(offspring)
                print(f"New offspring born! Spider ID: {offspring.spider_id}")
        
        time.sleep(0.1)  # Slow down simulation
    
    # Final render
    print("\n" + "="*50)
    print("SIMULATION COMPLETE")
    print("="*50)
    world.render(spiders)
    print(f"Final spider count: {len(spiders)}")
    
    return spiders


if __name__ == "__main__":
    print("="*50)
    print("Driftmoor Giant Spider AI - Simulation")
    print("="*50)
    
    # Run a single spider hunting
    print("\n--- Single Spider Hunt ---")
    run_simulation(steps=30, num_spiders=1, create_pair=False)
    
    # Run spider pair for reproduction
    print("\n\n--- Spider Pair with Reproduction ---")
    run_simulation(steps=30, num_spiders=2, create_pair=True)
