import json
from pathlib import Path
import pytest
from spider_sim.locomotion import LocalLocomotion,difference
from spider_sim.ecology import EcologyEncounter,SPIDER,baseline_policies,ecology_round
from spider_sim.djepa_ecology import senses


def motor_obs(heading=0,last=None,clearance=None):
    return {'self':{'heading_deg':heading,'local_clearance_ft':clearance or [10.]*8},'last_result':last}


def test_roaming_drifts_gently_and_turns_largely_after_actual_blockage():
    motor=LocalLocomotion(42);previous=0
    for _ in range(12):
        command,reason=motor.roam(motor_obs(previous,{'action':'move','distance_ft':5}))
        assert difference(command['heading_deg'],previous)<=4.000001
        assert command['face_travel'] and reason=='roam_drift'
        previous=command['heading_deg']
    command,reason=motor.roam(motor_obs(previous,{'action':'move','distance_ft':0},[0,10,10,10,10,10,10,10]))
    assert difference(command['heading_deg'],previous)>=30 and reason=='blocked_roam_turn'


def test_recovery_escapes_the_reported_baseline_wall_pose():
    world=EcologyEncounter(42)
    # The actual zero-progress wall point from the archived episode.
    # Stable fixture coordinates are supplied below, not required from local traces.
    world.positions[SPIDER]=(46.56746986383745,80.96074102647557,'Receiving Floor')
    from models.body_geometry import Orientation
    world.orientations[SPIDER]=Orientation(heading=-104.7488816378474)
    failed={'action':'move','heading_deg':-104.7488816378474,'distance_ft':10}
    assert world.act(SPIDER,failed)['distance_ft']==0
    world.last_results[SPIDER]={'action':'move','distance_ft':0,'status':'blocked'}
    motor=LocalLocomotion(42);motor.last_move=failed
    # Force the forward sensor to reflect the wall; all other measurements are native.
    obs=senses(world,SPIDER)
    command,reason=motor.pursue(obs,failed)
    assert reason=='blocked_recovery' and difference(command['heading_deg'],failed['heading_deg'])>=30
    assert world.act(SPIDER,command)['distance_ft']>1


def test_each_death_creates_one_distinct_newborn_keeps_corpse_and_contact_tokens():
    world=EcologyEncounter(42,replenish_prey=True)
    old_tokens=dict(world.tokens[SPIDER]);old_pos=world.positions['cave_rat_1']
    for cid in ('cave_rat_1','cave_cricket_2'):
        world.config.characters[cid].character.current_hp=0;world.active.discard(cid)
    births=world.replace_dead_prey()
    assert {b['actor'] for b in births}=={'cave_rat_3','cave_cricket_3'}
    assert world.positions['cave_rat_1']==old_pos and world.config.characters['cave_rat_1'].character.current_hp==0
    assert world.config.characters['cave_rat_3'].character.id=='cave_rat_3'
    assert world.config.characters['cave_rat_3'].character.current_hp==world.max_hp['cave_rat_1']
    assert all(world.tokens[SPIDER][cid]==token for cid,token in old_tokens.items())
    assert world.replace_dead_prey()==[]
    world.config.characters['cave_rat_3'].character.current_hp=0;world.active.discard('cave_rat_3')
    assert world.replace_dead_prey()[0]['actor']=='cave_rat_4'
    assert sum(cid.startswith('cave_rat_') for cid in world.active)==2


def test_birth_at_round_end_gives_a_fresh_policy_and_no_immediate_newborn_turn():
    world=EcologyEncounter(42,replenish_prey=True);policies=baseline_policies(world,42)
    world.config.characters['cave_rat_1'].character.current_hp=0;world.active.discard('cave_rat_1')
    rows=ecology_round(world,policies,0)
    assert 'cave_rat_3' in policies and 'cave_rat_3' in world.active
    assert all(r['actor']!='cave_rat_3' for r in rows)
    assert any(r['actor']=='cave_rat_3' for r in ecology_round(world,policies,1))


def test_new_viewer_replay_and_restart_restore_population_and_roaming_state(tmp_path):
    from experiments.watch_level2 import LiveRun
    from spider_sim.pursuit import PursuitModel,save
    model=PursuitModel(21);model.value_kind='negative_action_logit';path=tmp_path/'model.pt';save(model,path)
    run=LiveRun(tmp_path/'live',42,8,spider_model=path,spider_local_motion=True,replenish_prey=True)
    for _ in range(4):run.step()
    restored=LiveRun(run.output,42,8,spider_model=path,spider_local_motion=True,replenish_prey=True,resume=True)
    assert restored.world.positions==run.world.positions
    assert restored.world.replaced_prey==run.world.replaced_prey
    run.step();restored.step();assert run.world.positions==restored.world.positions
    restored.restart();assert len(restored.world.positions)==10 and not restored.world.replaced_prey
