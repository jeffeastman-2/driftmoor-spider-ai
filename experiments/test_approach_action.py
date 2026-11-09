import pytest
from models.body_geometry import Orientation
from experiments.approach_action import approach_distance,ApproachEncounter
from experiments.audit_jepa_planning import fixture
from experiments.engine_cost_control import run_episode
from experiments.recurrent_agents import Agent
from spider_sim.dual_encounter import SPIDER,PLAYER,ACTIONS
from spider_sim.encounter_teacher import predicted_orientation


def test_approach_uses_private_observation_and_stops_short_of_full_stride():
    world=fixture(SPIDER,"healthy_far_idle",610000,ApproachEncounter)
    obs=world.observation_for(SPIDER)
    pose=Orientation(heading=obs["contact"]["bearing_deg"])
    step=approach_distance(obs,pose,world.speed_ft(SPIDER))
    assert 0<step<world.speed_ft(SPIDER)
    result=world.act(SPIDER,ACTIONS.index("advance"),orientation=pose)
    assert result["distance_moved_ft"]==pytest.approx(step)
    assert world.observation_for(SPIDER)["contact"] is not None
    obs=world.observation_for(SPIDER)
    result=world.act(SPIDER,ACTIONS.index("attack"),orientation=predicted_orientation(obs,7,[0]*5))
    assert result["admitted"]


def test_unknown_contact_preserves_motion_and_speed_zero_stays_zero():
    world=fixture(SPIDER,"no_contact_idle",610000,ApproachEncounter)
    obs=world.observation_for(SPIDER)
    assert approach_distance(obs,Orientation(),30)==30
    assert approach_distance(obs,Orientation(),0)==0


def test_reference_control_can_select_approach_with_unchanged_costs():
    result=run_episode(Agent().eval(),SPIDER,"healthy_far_idle",610000,
        "engine_cost_only",draws=2,rounds=2,world_class=ApproachEncounter)
    assert result["stats"]["first_action"]=="advance"
    assert result["stats"]["admitted"]>=1


def test_opponents_motor_semantics_remain_full_stride():
    world=fixture(SPIDER,"healthy_far_idle",610000,ApproachEncounter)
    assert world.approach_actor==SPIDER
    assert world.movement_distance_ft(PLAYER,"advance",Orientation(heading=180))==world.speed_ft(PLAYER)
    assert world.movement_distance_ft(SPIDER,"flee",Orientation())==2*world.speed_ft(SPIDER)
