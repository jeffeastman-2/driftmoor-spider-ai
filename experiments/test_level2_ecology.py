import copy
import json
import pytest
from spider_sim.ecology import EcologyEncounter,GoalBaseline,ForagerBaseline,run_ecology,SPIDER,OBSERVATION_SCHEMA
from engine.webs import damage_web


def test_private_contacts_use_opaque_tracks_and_no_opponent_stats():
    world=EcologyEncounter(5)
    obs=world.observation_for(SPIDER)
    assert obs["schema"]==OBSERVATION_SCHEMA and obs["contacts"]
    encoded=json.dumps(obs)
    for forbidden in ("cave_rat","shadowbeast","armor","faction","target_hp"):
        assert forbidden not in encoded
    web=[c for c in obs["contacts"] if c["modality"]=="web"]
    assert web and all("relative_position_ft" not in c and "body_dimensions_ft" not in c for c in web)
    before=copy.deepcopy(obs)
    world.config.characters['cave_rat_1'].character.ac=99
    world.config.characters['cave_rat_1'].character.current_hp=1
    assert world.observation_for(SPIDER)==before


def test_hidden_actor_movement_cannot_change_spider_observation():
    world=EcologyEncounter(5)
    before=world.observation_for(SPIDER)
    world.positions['pen_lizard_1']=(106,54,'BEAST PENS')
    assert world.observation_for(SPIDER)==before


def test_goal_selection_uses_own_state_and_local_evidence():
    world=EcologyEncounter(0);policy=GoalBaseline()
    obs=world.observation_for(SPIDER)
    policy.choose(obs);assert policy.goal=='feed'
    obs['self']['hp_fraction']=.1
    command=policy.choose(obs);assert policy.goal=='survive' and command['action']=='move'
    obs['contacts']=[];policy.choose(obs);assert policy.goal=='rest'
    obs['self']['hp_fraction']=1
    obs['own_webs_in_view']=[{'item':'test','integrity':.2,'relative_position_ft':[2,0]}]
    command=policy.choose(obs);assert policy.goal=='maintain_web' and command['action']=='repair_web'
    with pytest.raises(ValueError):policy.choose({'schema':'withdrawal-cornered-v6'})


def test_repair_requires_local_reach_and_an_intact_owner_relationship():
    world=EcologyEncounter()
    world.positions[SPIDER]=(99,140,'FUNGAL STORES')
    item=world.config.items['store_web_west'];damage_web(item,3)
    assert world.act(SPIDER,{'action':'repair_web','item':item.id})['status']=='web_repaired'
    assert float(item.state['web_hp'])==3
    assert world.act(SPIDER,{'action':'repair_web','item':'cistern_web'})['status']=='web_not_reachable'


def test_native_combat_and_tethered_movement():
    world=EcologyEncounter(42)
    world.positions['cave_rat_1']=(107,140,'FUNGAL STORES')
    token=world.tokens[SPIDER]['cave_rat_1']
    result=world.act(SPIDER,{'action':'attack','target':token})
    assert result['admitted'] and result['status'] in ('hit','miss')
    # Same motor command cannot attack a distant contact whose geometry is unavailable.
    world.positions['cave_rat_1']=(57,110,'RECEIVING FLOOR')
    assert not world.act(SPIDER,{'action':'attack','target':token})['admitted']
    for _ in range(4):world.act('pen_lizard_1',{'action':'move','heading_deg':90,'distance_ft':30})
    assert world.positions['pen_lizard_1'][1]<=60.000001


def test_baseline_is_reproducible_and_exercises_actual_prey_combat():
    first=run_ecology(42,30)
    assert first==run_ecology(42,30)
    rows=[r for r in first['records'] if r['actor']==SPIDER]
    assert any(r['result']['admitted'] for r in rows)
    assert any(first['final_hp'][cid]==0 for cid in ('cave_rat_1','cave_rat_2','cave_cricket_1'))
    assert all('schema' in r['observation'] for r in first['records'])


def test_goblin_wakes_on_living_enemy_entry_and_has_controller():
    from spider_sim.ecology import baseline_policies,ecology_round
    world=EcologyEncounter(42);goblin='freight_master_goblin'
    policies=baseline_policies(world,42)
    assert goblin not in world.active and goblin in policies
    world.positions[SPIDER]=(26,134,"FACTOR'S OFFICE")
    result=world.act(SPIDER,{'action':'wait'})
    assert result['activations']==[{'actor':goblin,'reason':"zone_entry:Factor's Office"}]
    assert goblin in world.active
    assert 'activations' not in world.observation_for(SPIDER)['last_result']
    rows=ecology_round(world,policies,0)
    assert any(r['actor']==goblin for r in rows)
    world.config.characters[goblin].character.current_hp=0
    world.active.discard(goblin)
    world.act(SPIDER,{'action':'wait'})
    assert goblin not in world.active


def test_corpse_cannot_wake_dormant_goblin():
    world=EcologyEncounter(42)
    world.positions['cave_rat_1']=(26,134,"FACTOR'S OFFICE")
    world.config.characters['cave_rat_1'].character.current_hp=0
    world.active.discard('cave_rat_1')
    world.act(SPIDER,{'action':'wait'})
    assert 'freight_master_goblin' not in world.active
