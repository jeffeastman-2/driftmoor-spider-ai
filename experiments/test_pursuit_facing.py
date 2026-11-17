import math
import torch
import pytest
from spider_sim.ecology import EcologyEncounter,SPIDER
from spider_sim.djepa_ecology import senses
from spider_sim.pursuit import PursuitPolicy
from models.body_geometry import Orientation


def world_at_store():
    world=EcologyEncounter(42)
    world.positions[SPIDER]=(103,140,'Fungal Stores')
    world.positions['cave_rat_1']=(107,140,'Fungal Stores')
    world.orientations[SPIDER]=Orientation(heading=0,pitch=17,roll=11)
    return world


def test_sideways_move_keeps_facing_and_tilt_without_a_turn_request():
    world=world_at_store();original=world.orientations[SPIDER]
    result=world.act(SPIDER,{'action':'move','heading_deg':90,'distance_ft':5,'keep_facing':True})
    assert world.positions[SPIDER][:2]==pytest.approx((103,145))
    assert world.orientations[SPIDER]==original
    assert result['travel_heading_deg']==90 and result['body_heading_deg']==0 and result['turn_reason'] is None


def test_explicit_tracking_faces_measured_prey_after_actual_own_movement():
    world=world_at_store();track=world.tokens[SPIDER]['cave_rat_1']
    result=world.act(SPIDER,{'action':'move','heading_deg':90,'distance_ft':5,'keep_facing':True,'face_track':track})
    assert world.orientations[SPIDER].heading==pytest.approx(math.degrees(math.atan2(-5,4)))
    assert world.orientations[SPIDER].pitch==17 and world.orientations[SPIDER].roll==11
    assert result['turn_reason']=='track_visible_pursuit_contact'


def test_unseen_contact_cannot_rotate_body_using_hidden_world_position():
    world=world_at_store();track=world.tokens[SPIDER]['cave_rat_1']
    world.positions['cave_rat_1']=(20,60,'Cistern')
    assert not any(c['track']==track and c['modality']=='vision' for c in world.observation_for(SPIDER)['contacts'])
    original=world.orientations[SPIDER]
    result=world.act(SPIDER,{'action':'move','heading_deg':90,'distance_ft':5,'keep_facing':True,'face_track':track})
    assert world.orientations[SPIDER]==original and result['turn_reason'] is None


class PreferMoving:
    action_size=21
    def __init__(self):self.updates=0
    def represent(self,obs,hidden):
        self.updates+=1
        return torch.zeros(1,1,48),torch.zeros(1,1,48)
    def values(self,state,choices):
        scores=torch.ones(len(choices));scores[1]=-100;return scores


def test_no_pursuit_task_rests_instead_of_inventing_neural_exploration():
    model=PreferMoving();policy=PursuitPolicy(model,approach=True,facing='track',rest_without_goal=True)
    obs=senses(world_at_store(),SPIDER);obs['contacts']=[]
    assert policy.choose(obs)=={'action':'wait'}
    assert policy.goal=='rest' and policy.last_plan['selection_reason']=='no_pursuit_goal'
    assert model.updates==1


def test_learned_move_requests_facing_for_a_visible_target_explicitly():
    policy=PursuitPolicy(PreferMoving(),approach=True,facing='track',rest_without_goal=True)
    obs=senses(world_at_store(),SPIDER);command=policy.choose(obs)
    assert command['keep_facing'] is True
    assert command['face_track']==policy.task.track
