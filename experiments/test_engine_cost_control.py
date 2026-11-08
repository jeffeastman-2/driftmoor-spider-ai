import copy
import torch
from experiments.engine_cost_control import reference_scores,run_episode
from experiments.audit_jepa_planning import fixture
from experiments.recurrent_agents import Agent
from spider_sim.encounter_evaluation import TeacherAgent
from spider_sim.dual_encounter import SPIDER


def test_reference_costs_do_not_mutate_live_world_rng_or_opponent():
    world=fixture(SPIDER,"healthy_close_teacher",610000)
    teacher=TeacherAgent().session()
    obs=world.observation_for(SPIDER);rng=world.random.getstate()
    opponent=copy.deepcopy(teacher.__dict__)
    costs,futures=reference_scores(world,SPIDER,[0]*5,teacher,"teacher",2,630000)
    assert world.observation_for(SPIDER)==obs and world.random.getstate()==rng
    assert teacher.last_intent is None and teacher.last_orientation is None
    again=reference_scores(world,SPIDER,[0]*5,teacher,"teacher",2,630000)
    assert torch.equal(costs,again[0]) and futures.shape==(8,43)


def test_idle_reference_has_zero_stationary_injury_and_legal_bite():
    world=fixture(SPIDER,"healthy_close_idle",610000)
    costs,futures=reference_scores(world,SPIDER,[0]*5,TeacherAgent().session(),"idle",4,630000)
    assert futures[7,16]==1 and futures[7,41]==1
    assert costs[7]<costs[0]


def test_closed_loop_executes_reference_action_and_records_costs():
    episode=run_episode(Agent().eval(),SPIDER,"healthy_close_idle",610000,"engine_cost_only",4,rounds=1)
    assert episode["stats"]["first_action"]=="attack"
    assert episode["stats"]["admitted"]==1
    assert len(episode["turns"][0]["costs"])==8
