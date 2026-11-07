"""Observation boundary and closed-loop controller tests."""
import copy
import pytest

from spider_sim.spider_controller import (
    candidates, choose_rule, choose_learned, features, model_input,
    INPUT_SIZE, FEATURE_SIZE,
)
from spider_sim.spider_learning import episode, execute
from spider_sim.spider_experiment import ResolvedCavernExperiment
from models.body_geometry import Orientation


def ready():
    experiment = ResolvedCavernExperiment(spider_height_ft=0)
    for destination in ((120,195),(90,195),(69,195)):
        experiment.step("gilthaness","move",destination)
    return experiment


def test_no_contact_means_wait_and_no_target_invention():
    observation = ResolvedCavernExperiment().observe()
    assert candidates(observation) == [{"type":"wait"}]
    assert choose_rule(observation) == {"type":"wait"}
    assert choose_learned(observation,observation,lambda rows: [[0]*18]) == {"type":"wait"}


def test_rule_faces_visible_target_before_biting():
    experiment = ready()
    experiment.step("giant_spider","orient",orientation=Orientation(heading=180))
    observation = experiment.observe()
    choice = choose_rule(observation)
    assert choice["type"] == "orient"
    execute(experiment,choice)
    choice = choose_rule(experiment.observe())
    assert choice["type"] == "attack"
    row = execute(experiment,choice)
    assert row["outcome"]["combat"]["attempted"]


def test_rule_moves_toward_distant_observation_in_bounded_steps():
    experiment = ResolvedCavernExperiment(spider_height_ft=0)
    experiment.step("gilthaness","move",(100,195))
    observation = experiment.observe()
    orient = choose_rule(observation)
    if orient["type"] == "orient":
        execute(experiment,orient)
    action = choose_rule(experiment.observe())
    assert action["type"] == "move"
    before = experiment.positions["giant_spider"]
    row = execute(experiment,action)
    assert 0 < row["outcome"]["distance_moved_ft"] <= 8.1


def test_candidate_generation_is_pure_and_uses_only_observation():
    observation = ready().observe()
    frozen = copy.deepcopy(observation)
    first = candidates(observation)
    assert first == candidates(copy.deepcopy(observation))
    assert observation == frozen
    assert all(set(a) <= {"type","target_id","destination","orientation"} for a in first)


def test_prediction_can_choose_action_but_cannot_bypass_engine_range():
    experiment = ResolvedCavernExperiment()
    experiment.step("gilthaness","move",(120,195))
    observation = experiment.observe()
    def faulty(rows):
        return [[1,0]+[0]*15+[float(row[-1])] for row in rows]
    action = choose_learned(observation,observation,faulty)
    assert action["type"] == "attack"
    row = execute(experiment,action)
    assert not row["outcome"]["combat"]["attempted"]


def test_training_rows_contain_no_privileged_fields_and_fixed_dimensions():
    rows,_ = episode(12,policy="explore",steps=5)
    for row in rows:
        assert len(row["input"]) == INPUT_SIZE
        assert len(row["target"]) == FEATURE_SIZE+1
        assert "ground_truth" not in row
        assert "baseline" not in row
        assert "geometry" not in row["feedback"]
        assert "destination_xy" not in row
        assert row["target"][-1] in {0,1}


def test_history_and_actions_affect_model_input():
    observation = ready().observe()
    previous = copy.deepcopy(observation)
    previous["contacts"] = []
    wait = {"type":"wait"}
    attack = {"type":"attack","target_id":"gilthaness"}
    assert model_input(previous,observation,wait) != model_input(observation,observation,wait)
    assert model_input(previous,observation,wait) != model_input(previous,observation,attack)


def test_closed_loop_episode_is_repeatable():
    assert episode(19,policy="rule",steps=8) == episode(19,policy="rule",steps=8)


def test_zone_baseline_does_not_act_before_entry():
    rows,_ = episode(1,policy="zone",steps=4,family="retreat")
    assert all(row["action"]["type"]=="wait" for row in rows[:2])
