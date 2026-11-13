import pytest
import torch
from models.body_geometry import Orientation
from experiments.bite_advice import AdviceEncounter,posed_values,AdvisedPlanner
from experiments.accurate_senses import observation
from experiments.audit_jepa_planning import fixture
from experiments.recurrent_agents import Agent
from experiments.jepa_planner import Planner
from spider_sim.dual_encounter import SPIDER


def test_candidate_pose_transform_matches_new_measurement_without_world_query():
    world=fixture(SPIDER,"healthy_close_idle",970000,AdviceEncounter)
    old=world.orientations[SPIDER];senses=world.observation_for(SPIDER)["body_senses"]
    new=Orientation(15,25,80)
    transformed=posed_values(senses,old,new)
    world.orientations[SPIDER]=new
    actual=world.observation_for(SPIDER)["body_senses"]["values"]
    assert transformed==pytest.approx(actual,abs=1e-10)


def test_advice_changes_only_attack_admission_and_updates_memory_once():
    class Advisor:
        def probability(self,values):return .9
    agent=Agent().eval();world=fixture(SPIDER,"healthy_close_idle",970001,AdviceEncounter)
    obs=world.observation_for(SPIDER);baseline=Planner(agent).session();baseline.choose_observation(SPIDER,obs)
    session=AdvisedPlanner(agent,Advisor()).session();session.choose_observation(SPIDER,obs)
    assert torch.equal(session.base.hidden,baseline.hidden)
    old=torch.tensor(baseline.last_plan["predictions"]);new=torch.tensor(session.last_plan["predictions"])
    old[7,41]=new[7,41]
    assert torch.equal(old,new)
    assert session.last_plan["bite_advice"]==.9
    assert session.last_plan["without_advice_action"]==baseline.last_plan["selected"]
