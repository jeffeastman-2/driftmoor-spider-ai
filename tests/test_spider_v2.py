"""V2 planner and matched-pose collection checks."""
import copy
import pytest

from spider_sim.spider_experiment import ResolvedCavernExperiment
from spider_sim.spider_v2 import predict_candidates, collect_pose_pairs, covered_rows


def seen():
    experiment = ResolvedCavernExperiment()
    experiment.step("gilthaness","move",(120,195))
    return experiment.observe()


def test_two_step_credit_selects_preparation_instead_of_wait():
    observation = seen()
    def predictions(rows):
        result = []
        for row in rows:
            action_code = row[34:38]
            is_orient = action_code[1] == 1
            result.append([1,0.2]+[0]*15+[0.01,0.95 if is_orient else 0.01])
        return result
    assert predict_candidates(observation,observation,predictions,horizon=1)["type"]=="wait"
    assert predict_candidates(observation,observation,predictions,horizon=2)["type"]=="orient"


def test_v2_cannot_invent_contacts_even_with_optimistic_predictions():
    observation = ResolvedCavernExperiment().observe()
    prediction = lambda rows:[[1,0]+[0]*15+[1,1] for row in rows]
    assert predict_candidates(observation,observation,prediction)=={"type":"wait"}


def test_pose_variants_and_candidate_branches_keep_scene_group_together():
    rows = collect_pose_pairs(2,40000)
    assert {r["episode_id"] for r in rows}=={40000,40001}
    for group in (40000,40001):
        subset = [r for r in rows if r["episode_id"]==group]
        assert {r["pose_variant"] for r in subset}=={0,1,2}
        assert all(len(r["input"])==44 and len(r["target"])==19 for r in subset)
        # No hidden coordinates or geometry in the learning interface.
        assert all(set(r)<={
            "episode_id","turn","input","target","action_type","pose_variant","source"
        } for r in subset)
        assert all(r["target"][-1] in (0,1) and r["target"][-2] in (0,1) for r in subset)


def test_pose_pairs_repeat_with_same_seeds():
    assert collect_pose_pairs(1,40002)==collect_pose_pairs(1,40002)


def test_two_step_model_outputs_are_not_used_as_direct_engine_permissions():
    from spider_sim.spider_learning import execute
    experiment = ResolvedCavernExperiment()
    experiment.step("gilthaness","move",(120,195))
    observation = experiment.observe()
    def false_prediction(rows):
        return [[1,0]+[0]*15+[float(r[37]),0] for r in rows]
    action = predict_candidates(observation,observation,false_prediction)
    assert action["type"]=="attack"
    result = execute(experiment,action)
    assert not result["outcome"]["combat"]["attempted"]
