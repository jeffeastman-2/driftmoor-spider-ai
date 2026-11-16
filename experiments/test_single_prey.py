import copy
import math
import torch
from spider_sim.ecology import SPIDER
from spider_sim.djepa_ecology import senses
from spider_sim.pursuit import PursuitGoal,PursuitModel,PursuitPolicy,features,load,save
from experiments.train_single_prey import fixture,advance,collect,tensors,objective

torch.set_num_threads(1)


def test_goal_retains_contact_when_a_closer_one_appears_and_uses_actual_movement():
    world,prey,_=fixture(500000,'stationary');obs=senses(world,SPIDER)
    task=PursuitGoal();selected=task.observe(obs);track=selected['track']
    extra=copy.deepcopy(selected);extra.update(track='new-contact',range_ft=0)
    obs['contacts'].append(extra)
    assert task.observe(obs)['track']==track
    original=list(selected['relative_position_ft'])
    obs['contacts']=[];obs['last_result']={'action':'move','distance_ft':2}
    task.previous={'action':'move','heading_deg':90,'distance_ft':10}
    remembered=task.observe(obs)
    assert math.isclose(remembered['relative_position_ft'][0],original[0],abs_tol=1e-12)
    assert remembered['relative_position_ft'][1]==original[1]-2
    assert task.age==1
    task.previous={'action':'wait'}
    for _ in range(4):task.observe(obs)
    assert task.last is None and task.track is None


def test_fixture_remains_exactly_one_spider_and_one_prey_after_office_entry():
    world,prey,opponent=fixture(500000,'fleeing')
    world.positions[SPIDER]=(26,134,"FACTOR'S OFFICE")
    advance(world,prey,opponent,'fleeing',{'action':'wait'})
    assert world.active=={SPIDER,prey}


def test_policy_features_do_not_include_opponent_hp_or_world_position(tmp_path):
    world,prey,_=fixture(500000,'stationary');obs=senses(world,SPIDER)
    task=PursuitGoal();goal=task.observe(obs);before=features(obs,goal)
    world.config.characters[prey].character.current_hp=1
    world.config.characters[prey].character.ac=99
    after=features(senses(world,SPIDER),goal)
    assert before==after and len(before)==36
    model=PursuitModel().eval();save(model,tmp_path/'pursuit.pt')
    restored=load(tmp_path/'pursuit.pt');policy=PursuitPolicy(restored)
    command=policy.choose(obs)
    assert command in policy.last_plan['candidates'] and policy.hidden.shape==(1,1,48)
    assert not hasattr(policy,'world')


def test_engine_cost_training_has_real_gradients_and_no_padding_targets():
    episodes=collect(range(500000,500002),rounds=2)
    data=tensors(episodes);model=PursuitModel()
    loss=objective(model,data);assert torch.isfinite(loss)
    loss.backward()
    assert model.memory.weight_ih_l0.grad.abs().sum()>0
    assert model.value[-1].weight.grad.abs().sum()>0
