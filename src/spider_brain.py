"""
Main Spider Brain module for Driftmoor Giant Spider AI

Coordinates perception, prediction, memory, and action execution
to create intelligent spider NPCs.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time
import numpy as np

from perception import PerceptionSystem
from prediction import MovementPredictor, NPCState
from action_executor import ActionExecutor, SpiderAction
from memory import MemorySystem, NPCInteraction


@dataclass
class SpiderState:
    """Current state of the spider NPC"""
    position: Tuple[float, float]
    health: float
    hunger: float  # 0-100
    energy: float  # 0-100
    target_npc_id: Optional[int] = None
    current_action: Optional[SpiderAction] = None


class SpiderBrain:
    """Main AI brain for Driftmoor giant spiders"""
    
    def __init__(self, spider_id: int):
        """Initialize the spider brain"""
        self.spider_id = spider_id
        self.state = SpiderState(
            position=(0.0, 0.0),
            health=100.0,
            hunger=50.0,
            energy=100.0
        )
        self.perception = PerceptionSystem()
        self.predictor = MovementPredictor(self.perception)
        self.executor = ActionExecutor()
        self.memory = MemorySystem()
        self.frame_count = 0
        
    def update(self, game_state: Dict) -> List[SpiderAction]:
        """Update spider state and return actions to take"""
        self.frame_count += 1
        self.state.position = game_state.get('position', self.state.position)
        self.state.health = game_state.get('health', self.state.health)
        self.state.hunger = game_state.get('hunger', self.state.hunger)
        self.state.energy = game_state.get('energy', self.state.energy)
        
        detected_npcs = game_state.get('detected_npcs', [])
        actions = self._decide(detected_npcs)
        
        for action in actions:
            self.executor.execute_action(action)
            
        self.state.current_action = actions[-1] if actions else None
        return actions
    
    def _decide(self, detected_npcs: List[Dict]) -> List[SpiderAction]:
        """Decide what actions to take"""
        # Priority 1: Hunt if hungry
        if self.state.hunger > 40:
            return self._hunting_actions(detected_npcs)
        
        # Priority 2: Rest if low energy
        if self.state.energy < 30:
            return [SpiderAction(action_type="rest", duration=30.0)]
        
        # Default: Wander
        return self._wander_actions()
    
    def _hunting_actions(self, detected_npcs: List[Dict]) -> List[SpiderAction]:
        """Generate hunting actions"""
        if not detected_npcs:
            return self._wander_actions()
        
        target_id = self.executor.select_target(detected_npcs, {
            'position': self.state.position
        })
        
        if target_id is None:
            return self._wander_actions()
            
        target_pos = None
        for npc in detected_npcs:
            if npc.get('id') == target_id:
                target_pos = npc.get('position')
                break
                
        if target_pos is None:
            return self._wander_actions()
            
        actions = self.executor.plan_movement(
            self.state.position,
            target_pos
        )
        
        actions.append(SpiderAction(
            action_type="attack",
            target_position=target_pos,
            target_npc_id=target_id,
            duration=1.0,
            intensity=0.8
        ))
            
        return actions
    
    def _mate_actions(self, detected_npcs: List[Dict]) -> List[SpiderAction]:
        """Generate mating actions"""
        spider_partners = [npc for npc in detected_npcs 
                         if npc.get('type') == 'spider' and npc.get('id') != self.spider_id]
        
        if not spider_partners:
            return self._wander_actions()
        
        partner = min(spider_partners, 
                     key=lambda p: np.sqrt(
                         (p.get('position', (0,0))[0] - self.state.position[0])**2 +
                         (p.get('position', (0,0))[1] - self.state.position[1])**2
                     ))
        
        partner_pos = partner.get('position', self.state.position)
        actions = self.executor.plan_movement(self.state.position, partner_pos)
        
        actions.append(SpiderAction(
            action_type="mate",
            target_position=partner_pos,
            target_npc_id=partner.get('id'),
            duration=5.0,
            intensity=1.0
        ))
        
        return actions
    
    def _wander_actions(self) -> List[SpiderAction]:
        """Generate wandering behavior"""
        import random
        angle = random.uniform(0, 2 * np.pi)
        distance = random.uniform(50, 200)
        target_pos = (
            self.state.position[0] + np.cos(angle) * distance,
            self.state.position[1] + np.sin(angle) * distance
        )
        return self.executor.plan_movement(self.state.position, target_pos)
    
    def get_status(self) -> Dict:
        """Get spider brain status"""
        return {
            'spider_id': self.spider_id,
            'position': self.state.position,
            'health': self.state.health,
            'hunger': self.state.hunger,
            'energy': self.state.energy,
            'frame_count': self.frame_count,
            'current_action': self.state.current_action.action_type if self.state.current_action else None
        }


if __name__ == "__main__":
    print("SpiderBrain initialized (mock mode)")
