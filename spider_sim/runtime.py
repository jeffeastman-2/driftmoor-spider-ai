"""Locate the game dependency without copying it into this project."""
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_root = None

def game_root(root=None):
    """Configure once, before importing game modules; default to sibling worktree."""
    global _root
    if root is None and _root is not None:
        return _root
    selected = Path(root or os.environ.get("DRIFTMOOR_ROOT", PROJECT_ROOT.parent / "Driftmoor-spider-sim")).expanduser().resolve()
    required = ("engine/md_loader.py", "engine/sensory.py", "models/body_geometry.py", "scenes/echoing_caverns.md")
    if not all((selected / name).is_file() for name in required):
        raise ValueError(f"{selected} is not the experimental Driftmoor worktree; set DRIFTMOOR_ROOT or --driftmoor-root")
    if _root is not None and selected != _root and "engine.md_loader" in sys.modules:
        raise RuntimeError("Game modules already loaded; choose a checkout in a fresh process")
    _root = selected
    if str(selected) not in sys.path:
        sys.path.insert(0, str(selected))
    return selected

def run_game(root, code):
    """Run project experiments with game dependency in an isolated interpreter."""
    root = game_root(root)
    prefix = (
        f"import sys; sys.path.insert(0, {str(PROJECT_ROOT)!r}); "
        f"from spider_sim.runtime import game_root; game_root({str(root)!r}); "
    )
    return subprocess.run([sys.executable, "-c", prefix + code],
                          cwd=root, text=True, capture_output=True)
