"""Shared bootstrap for module and direct-script launchers."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spider_sim.runtime import game_root, run_game
