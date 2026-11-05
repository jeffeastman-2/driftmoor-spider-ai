"""
NPC movement prediction module using V-JEPA 2

Predicts future NPC positions and behaviors based on
visual perception and historical patterns.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

# Mock PerceptionSystem for testing without torch
class PerceptionSystem:
    def __init__(self, model_path=None):
        self.device = 'cpu'
        self.model = None


@dataclass
class NPCState:
    """Current state of an NPC"""
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    heading: float
    behavior: str  # "wandering", "fleeing", "attacking", "idle"
    health: float
    perception_radius: float


class MovementPredictor:
    """Predicts NPC movements using V-JEPA 2 features"""
    
    def __init__(self, perception: PerceptionSystem):
        """Initialize the movement predictor
        
        Args:
            perception: Perception system for visual input
        """
        self.perception = perception
        self.npc_history: Dict[int, List[NPCState]] = {}
        self.velocity_history: Dict[int, List[Tuple[float, float]]] = {}
        
    def update_history(self, frame_id: int, npc_states: List[NPCState]):
        """Update history with current NPC states"""
        for state in npc_states:
            npc_id = hash(state.position)
            if npc_id not in self.npc_history:
                self.npc_history[npc_id] = []
                self.velocity_history[npc_id] = []
            self.npc_history[npc_id].append(state)
            
            # Track velocity
            if len(self.npc_history[npc_id]) >= 2:
                prev_pos = self.npc_history[npc_id][-2].position
                curr_pos = state.position
                velocity = (
                    curr_pos[0] - prev_pos[0],
                    curr_pos[1] - prev_pos[1]
                )
                self.velocity_history[npc_id].append(velocity)
                
    def predict_trajectory(self, npc_id: int, 
                          horizon: int = 30) -> List[Tuple[float, float]]:
        """Predict NPC trajectory for future frames
        
        Args:
            npc_id: NPC identifier
            horizon: Number of frames to predict
            
        Returns:
            List of predicted positions
        """
        if npc_id not in self.npc_history:
            return []
            
        history = self.npc_history[npc_id]
        if len(history) < 3:
            return [history[-1].position] * horizon if history else []
            
        recent = history[-3:]
        velocities = [(s.velocity[0], s.velocity[1]) for s in recent]
        
        # Calculate average velocity
        avg_vx = np.mean([v[0] for v in velocities])
        avg_vy = np.mean([v[1] for v in velocities])
        
        # Predict trajectory
        predicted = []
        current_pos = history[-1].position
        for t in range(1, horizon + 1):
            new_x = current_pos[0] + avg_vx * t
            new_y = current_pos[1] + avg_vy * t
            predicted.append((new_x, new_y))
            
        return predicted
    
    def predict_behavior(self, npc_id: int) -> str:
        """Predict NPC behavior based on recent history
        
        Args:
            npc_id: NPC identifier
            
        Returns:
            Predicted behavior
        """
        if npc_id not in self.npc_history:
            return "idle"
            
        history = self.npc_history[npc_id]
        if len(history) < 5:
            return history[-1].behavior if history else "idle"
            
        behaviors = [h.behavior for h in history[-10:]]
        return max(set(behaviors), key=behaviors.count)
    
    def get_velocity(self, npc_id: int) -> Tuple[float, float]:
        """Get current velocity estimate for NPC
        
        Args:
            npc_id: NPC identifier
            
        Returns:
            Velocity tuple (vx, vy)
        """
        if npc_id not in self.velocity_history:
            return (0.0, 0.0)
            
        velocities = self.velocity_history[npc_id]
        if not velocities:
            return (0.0, 0.0)
            
        # Return average velocity
        return (
            np.mean([v[0] for v in velocities]),
            np.mean([v[1] for v in velocities])
        )
    
    def reset_history(self):
        """Reset all history"""
        self.npc_history.clear()
        self.velocity_history.clear()


if __name__ == "__main__":
    print("Movement Predictor initialized (mock mode)")
