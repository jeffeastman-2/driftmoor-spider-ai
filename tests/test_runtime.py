"""Check the project/game dependency boundary."""
import json
import pytest
from spider_sim.runtime import game_root, run_game

def test_invalid_dependency_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="experimental Driftmoor"):
        game_root(tmp_path)

def test_subprocess_imports_project_experiments_and_game():
    result = run_game(game_root(),
        "import json; from spider_sim.spider_experiment import run_approach; "
        "print(json.dumps(run_approach()))")
    assert result.returncode == 0, result.stderr
    rows = json.loads(result.stdout)
    assert len(rows) == 7
    assert not rows[0]["baseline"]["spider_active"]
    assert rows[-1]["baseline"]["spider_active"]
