"""
Action executor module for Driftmoor Giant Spider AI

Handles translating AI decisions into game actions for spider NPCs.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class SpiderAction:
    """Action that a spider can perform"""
    action_type: str  # "move", "attack", "retreat", "mate", "rest"
    target_position: Optional[Tuple[float, float]] = None
    target_npc_id: Optional[int] = None
    duration: float = 1.0  # seconds
    intensity: float = 1.0  # 0-1


class ActionExecutor:
    """Executes actions for spider NPCs in Driftmoor"""
    
    def __init__(self):
        """Initialize the action executor"""
        self.available_actions = [
            "move", "attack", "retreat", "mate", "rest",
            "climb", "web_build", "wait"
        ]
        self.action_history: List[SpiderAction] = []
        
    def execute_action(self, action: SpiderAction) -> bool:
        """Execute a spider action
        
        Args:
            action: The action to execute
            
        Returns:
            True if action was successful
        """
        # TODO: Interface with Driftmoor game engine
        # This will be implemented in Phase 4
        self.action_history.append(action)
        print(f"Executing action: {action.action_type}")
        return True
    
    def plan_movement(self, current_pos: Tuple[float, float],
                     target_pos: Tuple[float, float]) -> List[SpiderAction]:
        """Plan movement to target position
        
        Args:
            current_pos: Current spider position
            target_pos: Target position
            
        Returns:
            List of movement actions
        """
        # Simple direct path
        distance = np.sqrt((target_pos[0] - current_pos[0])**2 + 
                          (target_pos[1] - current_pos[1])**2)
        
        actions = []
        steps = max(1, int(distance / 10))  # One action every 10 units
        
        for i in range(steps):
            t = (i + 1) / steps
            intermediate_pos = (
                current_pos[0] + (target_pos[0] - current_pos[0]) * t,
                current_pos[1] + (target_pos[1] - current_pos[1]) * t
            )
            actions.append(SpiderAction(
                action_type="move",
                target_position=intermediate_pos,
                duration=1.0
            ))
            
        return actions
    
    def select_target(self, detected_npcs: List[Dict],
                     spider_state: Dict) -> Optional[int]:
        """Select an NPC target for hunting
        
        Args:
            detected_npcs: List of detected NPC information
            spider_state: Current spider state
            
        Returns:
            NPC ID to target, or None if no valid target
        """
        # TODO: Implement intelligent target selection
        # For now, select closest vulnerable NPC
        if not detected_npcs:
            return None
            
        closest_target = None
        min_distance = float('inf')
        
        for npc in detected_npcs:
            if npc.get('health', 0) <= 0:
                continue  # Dead NPC
                
            # Calculate distance
            distance = np.sqrt((npc['position'][0] - spider_state['position'][0])**2 +
                             (npc['position'][1] - spider_state['position'][1])**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_target = npc.get('id')
                
        return closest_target
    
    def get_avoidance_actions(self, threats: List[Dict],
                             current_pos: Tuple[float, float]) -> List[SpiderAction]:
        """Get actions to avoid threats
        
        Args:
            threats: List of threat information
            current_pos: Current spider position
            
        Returns:
            List of avoidance actions
        """
        if not threats:
            return []
            
        # Simple avoidance: move away from nearest threat
        nearest = min(threats, key=lambda t: np.sqrt(
            (t['position'][0] - current_pos[0])**2 +
            (t['position'][1] - current_pos[1])**2
        ))
        
        # Calculate escape direction
        escape_direction = (
            current_pos[0] - nearest['position'][0],
            current_pos[1] - nearest['position'][1]
        )
        
        # Normalize
        norm = np.sqrt(escape_direction[0]**2 + escape_direction[1]**2)
        if norm > 0:
            escape_direction = (
                escape_direction[0] / norm,
                escape_direction[1] / norm
            )
            
        # Move away
        escape_pos = (
            current_pos[0] + escape_direction[0] * 50,
            current_pos[1] + escape_direction[1] * 50
        )
        
        return self.plan_movement(current_pos, escape_pos)


if __name__ == "__main__":
    # Test the action executor
    executor = ActionExecutor()
    print("ActionExecutor initialized")
    print("Available actions:", executor.available_actions)
