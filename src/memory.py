"""
Memory system for Driftmoor Giant Spider AI

Tracks NPC patterns, history, and learned behaviors over time.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import time


@dataclass
class NPCInteraction:
    """Record of spider-NPC interaction"""
    npc_id: int
    timestamp: float
    action: str  # "seen", "chased", "attacked", "evaded"
    outcome: str = "unknown"
    location: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))


@dataclass 
class SpiderMemory:
    """Memory entry for spider's experiences"""
    frame_id: int
    timestamp: float
    detected_npcs: List[int] = field(default_factory=list)
    actions_taken: List[str] = field(default_factory=list)
    observations: Dict[str, float] = field(default_factory=dict)


class MemorySystem:
    """Manages spider's memory of the environment"""
    
    def __init__(self, max_memory_size: int = 1000):
        """Initialize the memory system
        
        Args:
            max_memory_size: Maximum number of memory entries to keep
        """
        self.max_memory_size = max_memory_size
        self.interactions: List[NPCInteraction] = []
        self.memory_log: List[SpiderMemory] = []
        self.npc_patterns: Dict[int, Dict] = defaultdict(dict)
        self.learned_behaviors: Dict[str, float] = {}
        
    def record_interaction(self, interaction: NPCInteraction):
        """Record a spider-NPC interaction
        
        Args:
            interaction: The interaction to record
        """
        self.interactions.append(interaction)
        
        # Update NPC pattern
        npc_id = interaction.npc_id
        if interaction.action == "chased" or interaction.action == "attacked":
            self.npc_patterns[npc_id]['vulnerable'] = True
        elif interaction.action == "evaded":
            self.npc_patterns[npc_id]['threat_level'] = self.npc_patterns[npc_id].get('threat_level', 0) + 1
            
    def update_memory(self, frame_id: int, detected_npcs: List[int],
                     actions: List[str], observations: Dict[str, float]):
        """Update memory for current frame
        
        Args:
            frame_id: Current frame number
            detected_npcs: IDs of NPCs detected this frame
            actions: Actions taken this frame
            observations: Key observations and their values
        """
        memory = SpiderMemory(
            frame_id=frame_id,
            timestamp=time.time(),
            detected_npcs=detected_npcs,
            actions_taken=actions,
            observations=observations
        )
        self.memory_log.append(memory)
        
        # Trim old memory
        if len(self.memory_log) > self.max_memory_size:
            self.memory_log = self.memory_log[-self.max_memory_size:]
            
    def get_npc_pattern(self, npc_id: int) -> Dict:
        """Get learned pattern for an NPC
        
        Args:
            npc_id: NPC ID
            
        Returns:
            Dictionary of learned patterns
        """
        return self.npc_patterns.get(npc_id, {})
    
    def get_successful_strategies(self) -> List[Dict]:
        """Get strategies that led to successful hunts
        
        Returns:
            List of successful strategy patterns
        """
        successful = []
        for i, interaction in enumerate(self.interactions):
            if interaction.outcome == "success":
                # Find the memory context for this interaction
                if i < len(self.memory_log):
                    memory = self.memory_log[i]
                    successful.append({
                        'context': memory.observations,
                        'actions': memory.actions_taken
                    })
        return successful
    
    def learn_from_success(self, success_context: Dict, actions: List[str]):
        """Learn from successful hunts
        
        Args:
            success_context: Context of successful hunt
            actions: Actions taken
        """
        for action in actions:
            self.learned_behaviors[action] = self.learned_behaviors.get(action, 0) + 1
            
    def clear_old_memories(self, days: int = 7):
        """Clear memories older than specified days
        
        Args:
            days: Number of days to keep
        """
        cutoff = time.time() - (days * 24 * 3600)
        self.interactions = [i for i in self.interactions if i.timestamp > cutoff]
        self.memory_log = [m for m in self.memory_log if m.timestamp > cutoff]


if __name__ == "__main__":
    # Test the memory system
    memory = MemorySystem()
    print("MemorySystem initialized")
    print(f"Max memory size: {memory.max_memory_size}")
