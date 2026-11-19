from copy import deepcopy
from types import SimpleNamespace
from experiments.lifecycle_experience import Coverage,ExperienceRecorder,private_observation,situations
from experiments.watch_level2 import LiveRun
from spider_sim.ecology import ecology_round
from spider_sim.pursuit import PursuitModel,save


def test_private_inputs_add_only_proprioception_and_do_not_modify_teacher_observation():
    obs={'self':{'hunger':.4,'hp_fraction':.5},'contacts':[{'track':'contact_2','modality':'web'}]}
    original=deepcopy(obs)
    world=SimpleNamespace(spiders={'a':'female'},mate_cooldown={'a':19},elapsed_seconds=6)
    captured=private_observation(world,'a',obs)
    assert obs==original
    assert captured['self']=={'hunger':.4,'hp_fraction':.5,'sex':'female','mate_cooldown_turns':3}
    assert captured['contacts']==obs['contacts']
    assert not {'seeking_mate','goal','courtship_ready'} & captured['self'].keys()


def test_recording_is_observational_and_preserves_actions_and_world_state(tmp_path):
    model=PursuitModel(21);model.value_kind='negative_action_logit';weights=tmp_path/'weights.pt';save(model,weights)
    kwargs=dict(spider_model=weights,spider_local_motion=True,replenish_prey=True,
        feeding_cycle=True,random_prey=True,spider_pair=True,lifecycle=True)
    plain=LiveRun(tmp_path/'plain',42,25,**kwargs);recorded=LiveRun(tmp_path/'recorded',42,25,**kwargs)
    recorder=ExperienceRecorder(recorded.world,recorded.policies)
    for turn in range(25):
        plain.set_hunger_rate(turn);recorded.set_hunger_rate(turn);recorder.prepare()
        expected=ecology_round(plain.world,plain.policies,turn)
        actual=ecology_round(recorded.world,recorded.policies,turn)
        assert actual==expected
        captured=recorder.decisions(42,turn+1,actual)
        assert all('mate_cooldown_turns' in r['observation']['self'] for r in captured)
        assert len(captured)==sum(r['actor'] in recorded.world.spiders for r in actual)
        assert plain.world.positions==recorded.world.positions and plain.world.hunger==recorded.world.hunger


def row(turn,goal='defend',flags=(),alive=True,actor='a'):
    return {'seed':42,'actor':actor,'turn':turn,'teacher_goal':goal,
        'command':{'action':'wait'},'result':{'status':'waited'},'after_round':{'alive':alive},
        'observation':{'self':{'hunger':.4,'hp_fraction':.8,'sex':'male'}},
        'audit':{'situations':list(flags),'previous_goal':'feed','target_kind':None}}


def test_coverage_counts_contiguous_segments_not_every_turn_as_new_encounter():
    c=Coverage()
    for t in (1,2,3):c.add(row(t,flags=['feeding_under_attack']))
    c.add(row(4));c.add(row(5,flags=['feeding_under_attack']))
    report=c.report()['counts']['feeding_under_attack']
    assert report=={'turns':4,'actor_episodes':2,'actors':1,'seeds':[42]}


def test_defense_sequences_distinguish_return_death_and_right_censoring():
    c=Coverage();c.add(row(1,flags=['feeding_under_attack']));c.add(row(2));c.add(row(3,goal='feed'))
    c.add(row(4,flags=['feeding_under_attack']));c.end_actor(42,'a',5)
    c.add(row(6,actor='b',flags=['feeding_under_attack']));c.finish_defense((42,'b'),'right_censored',10)
    assert c.report()['defense_outcomes']=={'returned_to_food':1,'died':1,'right_censored':1}
    assert c.report()['defense_examples'][0]['duration_turns']==2


def test_competing_needs_and_threshold_flags_do_not_require_hidden_identity():
    obs={'self':{'hunger':.19,'hp_fraction':.34,'mate_cooldown_turns':1},
        'contacts':[{'track':'x','modality':'web','conspecific':True,'mate_signal':'receptive','courtship_signal':True}],
        'food_in_view':[{'track':'y'}],'incoming_damage':[{'damage_hp':2}]}
    flags=situations(obs,'wrap')
    assert {'feeding_under_attack','critical_hp','near_hp_35','near_hunger_20',
        'food_and_receptive_partner','cooldown_near_expiry','web_courtship_signal'}<=flags
