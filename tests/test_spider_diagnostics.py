"""Diagnostics must not alter the frozen policy or live episode."""
import copy
import random

import pytest

from engine.game_rng import rng
from spider_sim.spider_learning import setup
from spider_sim.spider_controller import candidates, choose_learned
from spider_sim.spider_diagnostics import counterfactual, score, trace


def test_counterfactual_does_not_mutate_live_state_or_rng():
    experiment,environment_rng = setup(10000)
    original = copy.deepcopy({
        "positions":experiment.positions,"active":experiment.active,
        "observations":experiment.observe(),"time":experiment.elapsed_seconds,
        "hp":{k:v.character.current_hp for k,v in experiment.config.characters.items()},
    })
    env_state = environment_rng.getstate()
    combat_state = experiment.random.getstate()
    global_rng = rng()
    global_state = global_rng.getstate()
    result = counterfactual(experiment,environment_rng,0,"approach",{"type":"wait"})
    assert result["status"]=="waited"
    assert original == {
        "positions":experiment.positions,"active":experiment.active,
        "observations":experiment.observe(),"time":experiment.elapsed_seconds,
        "hp":{k:v.character.current_hp for k,v in experiment.config.characters.items()},
    }
    assert environment_rng.getstate()==env_state
    assert experiment.random.getstate()==combat_state
    assert rng() is global_rng
    assert global_rng.getstate()==global_state


def test_diagnostic_scores_match_frozen_policy_choice():
    experiment,_ = setup(10002)
    experiment.step("gilthaness","move",(120,195))
    observation = experiment.observe()
    options = candidates(observation)
    outputs = [[1,0.3]+[0]*15+[0.3 if a["type"]=="attack" else 0] for a in options]
    chosen = choose_learned(observation,observation,lambda rows:outputs)
    assert chosen == options[max(range(len(options)),key=lambda i:score(options[i],outputs[i]))]


def test_v1_objective_accepts_attack_probability_below_classifier_threshold():
    wait = [1,0.5]+[0]*16
    attack = [1,0.5]+[0]*15+[0.24]
    assert score({"type":"attack"},attack) > score({"type":"wait"},wait)
    attack[-1] = 0.23
    assert score({"type":"attack"},attack) < score({"type":"wait"},wait)


def test_counterfactual_trace_is_repeatable_and_has_candidate_outcomes():
    predictor = lambda rows:[[1,0.5]+[0]*16 for row in rows]
    first = trace(10003,predictor,steps=3)
    assert first == trace(10003,predictor,steps=3)
    assert all("actual_score" in c for row in first for c in row["candidates"])


def test_turn_then_bite_succeeds_where_wait_then_bite_fails():
    from spider_sim.spider_experiment import ResolvedCavernExperiment
    from models.body_geometry import Orientation
    experiment = ResolvedCavernExperiment(spider_height_ft=0)
    for point in ((120,195),(90,195),(69,195)):
        experiment.step("gilthaness","move",point)
    experiment.step("giant_spider","orient",orientation=Orientation(heading=90))
    environment_rng = random.Random(0)
    wait = counterfactual(
        experiment,environment_rng,4,"pause",{"type":"wait"},probe_followup=True,
    )
    turn = counterfactual(
        experiment,environment_rng,4,"pause",
        {"type":"orient","orientation":[0,0,0]},probe_followup=True,
    )
    assert not wait["followup_attack"]["admitted"]
    assert turn["followup_attack"]["admitted"]
    # The v1 one-step objective nevertheless prefers waiting.
    assert wait["actual_score"] > turn["actual_score"]
