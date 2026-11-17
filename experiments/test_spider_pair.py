import pytest
from spider_sim.ecology import EcologyEncounter,SPIDER,FEMALE_SPIDER
from models.space_time import Position


def paired():return EcologyEncounter(42,feeding_cycle=True,replenish_prey=True,random_prey=True,spider_pair=True)


def test_random_replacement_is_valid_reproducible_and_not_at_home():
    a=paired();b=paired();zones=set()
    for _ in range(20):
        for w in (a,b):
            rat=next(c for c in w.active if c.startswith('cave_rat_'))
            w.config.characters[rat].character.current_hp=0;w.active.discard(rat)
            birth=w.replace_dead_prey()[0]
            point=w.positions[birth['actor']];zones.add(point[2])
            assert point!=w.prey_homes[birth['home']]
            assert w.geometry.zones[point[2]].contains(Position(*point[:2]))
            assert not any(o.contains(Position(*point[:2])) for o in w.geometry.impassable if not all(o.contains(p.centroid()) for p in w.geometry.zones.values()))
        assert a.positions==b.positions
    assert len(zones)>2


def test_pair_has_independent_body_state_and_mutual_local_signals():
    w=paired()
    assert w.config.characters[SPIDER] is not w.config.characters[FEMALE_SPIDER]
    w.hunger[SPIDER]=0
    assert w.hunger[FEMALE_SPIDER]==.8
    w.positions[SPIDER]=(103,140,'Fungal Stores');w.positions[FEMALE_SPIDER]=(107,140,'Fungal Stores')
    contact=next(c for c in w.observation_for(SPIDER)['contacts'] if c['track']==w.tokens[SPIDER][FEMALE_SPIDER] and c['modality']=='vision')
    assert contact['conspecific'] and contact['mate_signal']=='unreceptive'
    assert 'hunger' not in contact and 'hp' not in contact
    w.hunger[FEMALE_SPIDER]=.1
    assert next(c for c in w.observation_for(SPIDER)['contacts'] if c.get('conspecific'))['mate_signal']=='receptive'


def test_courtship_requires_visible_close_fed_living_partners_and_cooldown():
    w=paired();w.positions[SPIDER]=(103,140,'Fungal Stores');w.positions[FEMALE_SPIDER]=(107,140,'Fungal Stores')
    cmd={'action':'court','target':w.tokens[SPIDER][FEMALE_SPIDER]}
    w.hunger[SPIDER]=0
    assert w.act(SPIDER,cmd)['status']=='too_hungry_to_mate'
    w.hunger[FEMALE_SPIDER]=.1
    assert w.act(SPIDER,cmd)['status']=='courting'
    assert w.act(SPIDER,cmd)['status']=='courting'
    assert w.act(SPIDER,cmd)['status']=='courtship_accepted'
    assert len(w.mating_events)==1
    assert w.act(SPIDER,cmd)['status']=='not_receptive'
    w.positions[FEMALE_SPIDER]=(20,60,'Cistern')
    assert w.act(SPIDER,cmd)['status']=='mate_not_reachable'


def test_feeding_policy_never_selects_conspecific_as_prey():
    from spider_sim.feeding import FeedingPolicy
    from spider_sim.pursuit import PursuitModel,PursuitPolicy
    from spider_sim.djepa_ecology import senses
    w=paired();w.positions[SPIDER]=(103,140,'Fungal Stores');w.positions[FEMALE_SPIDER]=(107,140,'Fungal Stores')
    policy=FeedingPolicy(PursuitPolicy(PursuitModel(21),approach=True,local_motion=True))
    token=w.tokens[SPIDER][FEMALE_SPIDER]
    for _ in range(4):
        command=policy.choose(senses(w,SPIDER))
        assert policy.hunter.task.track!=token
        assert not (command['action']=='attack' and command['target']==token)


def test_paired_viewer_resume_and_restart_keep_two_separate_memories(tmp_path):
    from experiments.watch_level2 import LiveRun
    from spider_sim.pursuit import PursuitModel,save
    model=PursuitModel(21);model.value_kind='negative_action_logit';weights=tmp_path/'weights.pt';save(model,weights)
    args=dict(spider_model=weights,feeding_cycle=True,spider_local_motion=True,replenish_prey=True,random_prey=True,spider_pair=True)
    run=LiveRun(tmp_path/'live',42,10,**args)
    for _ in range(3):run.step()
    restored=LiveRun(run.output,42,10,resume=True,**args)
    assert restored.world.positions==run.world.positions
    assert restored.world.hunger==run.world.hunger
    assert run.policies[SPIDER].hunter.hidden is not run.policies[FEMALE_SPIDER].hunter.hidden
    restored.restart()
    assert restored.world.hunger=={SPIDER:.8,FEMALE_SPIDER:.8}


def test_courtship_vibration_requires_connected_intact_web_and_expires():
    w=paired();w.positions[SPIDER]=(92,140,'Fungal Stores');w.positions[FEMALE_SPIDER]=(114,140,'Fungal Stores')
    w.hunger[SPIDER]=0
    assert w.act(SPIDER,{'action':'vibrate_web'})['status']=='courtship_vibration'
    signal=lambda:[c for c in w.observation_for(FEMALE_SPIDER)['contacts'] if c.get('courtship_signal')]
    assert signal() and signal()[0]['modality']=='web'
    assert 'relative_position_ft' not in signal()[0]
    w.config.items['store_web_west'].state['web_hp']='0'
    assert not signal()
    w.config.items['store_web_west'].state['web_hp']='10'
    assert signal()
    w.elapsed_seconds=12
    assert not signal()
    w.positions[SPIDER]=(57,110,'Receiving Floor')
    assert w.act(SPIDER,{'action':'vibrate_web'})['status']=='not_touching_web'


def test_interrupted_courtship_does_not_accumulate_progress():
    w=paired();w.positions[SPIDER]=(103,140,'Fungal Stores');w.positions[FEMALE_SPIDER]=(107,140,'Fungal Stores')
    w.hunger[SPIDER]=w.hunger[FEMALE_SPIDER]=0
    cmd={'action':'court','target':w.tokens[SPIDER][FEMALE_SPIDER]}
    assert w.act(SPIDER,cmd)['courtship_turns']==1
    w.act(SPIDER,{'action':'wait'})
    assert w.act(SPIDER,cmd)['courtship_turns']==1
