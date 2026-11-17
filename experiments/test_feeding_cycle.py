import math
from types import SimpleNamespace
import pytest
from spider_sim.ecology import EcologyEncounter,SPIDER,baseline_policies,ecology_round
from spider_sim.feeding import FeedingPolicy,food_value,WRAP_TURNS,HUNGER_PER_ROUND
from spider_sim.locomotion import LocalLocomotion


def meal_world():
    world=EcologyEncounter(42,feeding_cycle=True,replenish_prey=True)
    rat='cave_rat_1'
    # Reachable native geometry, with no fabricated attack outcome.
    world.positions[SPIDER]=(103,140,'Fungal Stores')
    world.positions[rat]=(107,140,'Fungal Stores')
    for _ in range(100):
        result=world.act(SPIDER,{'action':'attack','target':world.tokens[SPIDER][rat]})
        if result.get('target_defeated'):break
    assert result['target_defeated']
    return world,rat


def test_kill_wrap_feed_conserves_food_and_does_not_resurrect_prey():
    world,rat=meal_world();token=world.tokens[SPIDER][rat]
    assert world.food[rat]['remaining']==.5
    assert world.act(SPIDER,{'action':'feed','target':token})['status']=='food_not_wrapped'
    for i in range(WRAP_TURNS):
        result=world.act(SPIDER,{'action':'wrap','target':token})
        assert result['status']==('wrapped' if i==WRAP_TURNS-1 else 'wrapping')
    before=world.hunger[SPIDER]
    for _ in range(5):world.act(SPIDER,{'action':'feed','target':token})
    assert world.hunger[SPIDER]==pytest.approx(before-.5)
    assert world.food[rat]['remaining']==pytest.approx(0,abs=1e-12)
    assert world.config.characters[rat].character.current_hp==0
    assert world.act(SPIDER,{'action':'feed','target':token})['status']=='food_not_reachable'
    assert world.replace_dead_prey()[0]['actor']=='cave_rat_3'
    assert world.food[rat]['remaining']==pytest.approx(0,abs=1e-12)


def test_remote_food_and_live_targets_cannot_be_wrapped_or_eaten():
    world,rat=meal_world();token=world.tokens[SPIDER][rat]
    world.positions[SPIDER]=(20,60,'Cistern')
    assert world.act(SPIDER,{'action':'wrap','target':token})['status']=='food_not_reachable'
    assert world.act(SPIDER,{'action':'feed','target':world.tokens[SPIDER]['cave_rat_2']})['status']=='food_not_reachable'
    assert world.food[rat]['wrap_turns']==0


def test_hunger_increases_once_per_round_even_when_waiting_and_stays_bounded():
    world=EcologyEncounter(42,feeding_cycle=True)
    class Wait:
        def choose(self,obs):return {'action':'wait'}
    policies={cid:Wait() for cid in world.positions}
    before=world.hunger[SPIDER];ecology_round(world,policies,0)
    assert world.hunger[SPIDER]==pytest.approx(before+HUNGER_PER_ROUND)
    world.hunger[SPIDER]=1;ecology_round(world,policies,1)
    assert world.hunger[SPIDER]==1


def test_prey_food_values_and_full_spider_leaves_food_for_later():
    world,rat=meal_world()
    assert food_value('cave_cricket_1',world.bodies['cave_cricket_1']) < food_value(rat,world.bodies[rat]) < food_value('freight_shadowbeast',world.bodies['freight_shadowbeast'])
    world.food[rat]['wrap_turns']=WRAP_TURNS;world.hunger[SPIDER]=.03
    result=world.act(SPIDER,{'action':'feed','target':world.tokens[SPIDER][rat]})
    assert world.hunger[SPIDER]==0
    assert result['consumed']==pytest.approx(.03)
    assert world.food[rat]['remaining']==pytest.approx(.47)


def test_satisfied_spider_searches_without_a_partner_then_resumes_hunting():
    class Hunter:
        def __init__(self):
            self.task=SimpleNamespace(previous=None,track=None,last=None)
            self.locomotion=LocalLocomotion(42);self.last_plan={}
        def choose(self,obs):return {'action':'attack','target':'contact_1'}
    policy=FeedingPolicy(Hunter())
    obs={'self':{'hunger':0,'hp_fraction':1,'heading_deg':0,'local_clearance_ft':[10.]*8},'contacts':[], 'food_in_view':[]}
    command=policy.choose(obs)
    assert policy.goal=='seek_mate' and command['action']=='move'
    obs['self']['hunger']=.4;policy.choose(obs);assert policy.goal=='seek_mate'
    obs['self']['hunger']=.6;command=policy.choose(obs)
    assert policy.goal=='hunt' and command['action']=='attack'


def test_injury_interrupts_feeding_with_private_sensory_retreat():
    class Hunter:
        locomotion=None
        task=SimpleNamespace(previous=None,track=None,last=None)
        last_plan={}
        def choose(self,obs):return {'action':'attack','target':'contact_1'}
    policy=FeedingPolicy(Hunter())
    obs={'self':{'hunger':.8,'hp_fraction':.2},'contacts':[{'track':'contact_1','modality':'vision','range_ft':3,'bearing_deg':20}], 'food_in_view':[]}
    command=policy.choose(obs)
    assert policy.goal=='survive' and command['action']=='move' and command['heading_deg']==200


def test_feeding_viewer_replays_body_state_and_resets_it(tmp_path):
    from experiments.watch_level2 import LiveRun
    from spider_sim.pursuit import PursuitModel,save
    model=PursuitModel(21);model.value_kind='negative_action_logit'
    weights=tmp_path/'model.pt';save(model,weights)
    args=dict(spider_model=weights,spider_local_motion=True,replenish_prey=True,feeding_cycle=True)
    run=LiveRun(tmp_path/'live',42,8,**args)
    for _ in range(3):run.step()
    restored=LiveRun(run.output,42,8,resume=True,**args)
    assert restored.world.hunger==run.world.hunger
    assert restored.world.food==run.world.food
    run.step();restored.step()
    assert restored.world.hunger==run.world.hunger
    assert restored.world.positions==run.world.positions
    restored.restart()
    assert restored.world.hunger[SPIDER]==.8 and restored.world.food=={}


def test_damage_interrupts_meal_targets_attacker_then_resumes_same_meal():
    from types import SimpleNamespace
    class Hunter:
        locomotion=None
        task=SimpleNamespace(previous=None,track=None,last=None)
        last_plan={}
        def choose(self,obs):return {'action':'wait'}
    policy=FeedingPolicy(Hunter());policy.goal='wrap';policy.meal='food'
    obs={'self':{'hunger':.8,'hp_fraction':.9},'contacts':[{'track':'threat','modality':'vision','range_ft':4,'relative_position_ft':[4,0,0]}],
         'food_in_view':[{'track':'food','range_ft':4,'wrapped':False,'remaining':.5}],
         'incoming_damage':[{'damage_hp':1,'attacker':'threat'}]}
    command=policy.choose(obs)
    assert policy.goal=='defend' and command=={'action':'attack','target':'threat'}
    assert policy.meal=='food'
    obs['incoming_damage']=[]
    assert policy.choose(obs)['action']=='attack'  # Stay engaged between hits.
    obs['last_result']={'target_defeated':True,'target':'threat'};obs['contacts']=[]
    command=policy.choose(obs)
    assert policy.goal=='wrap' and command=={'action':'wrap','target':'food'}


def test_unknown_damage_source_does_not_accuse_an_unrelated_visible_contact():
    from types import SimpleNamespace
    class Hunter:
        locomotion=None;task=SimpleNamespace(previous=None,track=None,last=None);last_plan={}
        def choose(self,obs):return {'action':'wait'}
    policy=FeedingPolicy(Hunter());policy.goal='feed'
    obs={'self':{'hunger':.5,'hp_fraction':.9},'contacts':[{'track':'bystander','modality':'vision','range_ft':4}],
         'incoming_damage':[{'damage_hp':1,'attacker':None}]}
    assert policy.choose(obs)=={'action':'wait'} and policy.goal=='defend'
    assert policy.defense_track is None


def test_actual_engine_hit_is_felt_once_and_misses_do_not_trigger_defense():
    from spider_sim.djepa_ecology import senses
    world,rat=meal_world()
    attacker='freight_shadowbeast';world.positions[attacker]=(107,140,'Fungal Stores')
    command={'action':'attack','target':world.tokens[attacker][SPIDER]}
    for _ in range(30):
        result=world.act(attacker,command)
        if result['damage']>0:break
        assert world.observation_for(SPIDER)['incoming_damage']==[]
    assert result['damage']>0
    events=senses(world,SPIDER)['incoming_damage']
    assert events[-1]['damage_hp']==result['damage']
    assert events[-1]['attacker']==world.tokens[SPIDER][attacker]
    assert world.observation_for(SPIDER)['incoming_damage']==events
    world.act(SPIDER,{'action':'wait'})
    assert world.observation_for(SPIDER)['incoming_damage']==[]
