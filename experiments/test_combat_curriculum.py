from experiments.combat_curriculum import curriculum_world,CASES
from experiments.combat_learning import collect_experience
from experiments.recurrent_agents import Agent
from spider_sim.dual_encounter import SPIDER,PLAYER


def test_case_and_action_coverage_are_independent_and_balanced():
    seen=set()
    for seed in range(400256,400320):
        world=curriculum_world(SPIDER,seed)
        seen.add((world.curriculum_case,seed%8))
        for actor in (SPIDER,PLAYER):
            assert world.positions[actor][2]
    assert seen=={(case,action) for case in CASES for action in range(8)}


def test_focused_first_action_is_executed_and_private_horizon_is_preserved():
    episodes=collect_experience(Agent().eval(),SPIDER,range(400256,400264),rounds=2,
        world_factory=curriculum_world,first_action=lambda seed:seed%8)
    for episode in episodes:
        rows=episode["rows"]
        assert rows[0]["action"]==episode["seed"]%8
        assert len(rows[0]["observation"])==41
        if len(rows)>1:
            assert rows[0]["next_observation"]==rows[1]["observation"]
