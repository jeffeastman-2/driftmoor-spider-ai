import pytest
from spider_sim.ecology import EcologyEncounter,SPIDER,FEMALE_SPIDER,baseline_policies,ecology_round
from spider_sim.lifecycle import STARVATION_INTERVAL
from spider_sim.feeding import FeedingPolicy
from spider_sim.pursuit import PursuitModel,PursuitPolicy


def world_and_policies():
    w=EcologyEncounter(42,feeding_cycle=True,replenish_prey=True,random_prey=True,spider_pair=True,lifecycle=True)
    p=baseline_policies(w,42);model=PursuitModel(21);model.value_kind='negative_action_logit'
    for n,c in enumerate(w.spiders):p[c]=FeedingPolicy(PursuitPolicy(model,approach=True,local_motion=True,seed=42+n))
    return w,p


def lay(w):
    w.positions[SPIDER]=(103,140,'Fungal Stores');w.positions[FEMALE_SPIDER]=(107,140,'Fungal Stores')
    w.hunger[SPIDER]=w.hunger[FEMALE_SPIDER]=0
    command={'action':'court','target':w.tokens[SPIDER][FEMALE_SPIDER]}
    for _ in range(3):result=w.act(SPIDER,command)
    assert result['status']=='courtship_accepted'
    return result['egg_sac']


def test_accepted_mating_creates_one_sac_then_one_hungry_pair_with_fresh_memories():
    w,p=world_and_policies();key=lay(w);old=dict(w.tokens[SPIDER])
    assert key in w.config.items and not w.egg_sacs[key]['hatched']
    # Laid at first frame end (6s), then two further rounds to hatch at 18s.
    for t in range(2):ecology_round(w,p,t)
    assert len(w.spiders)==2
    rows=ecology_round(w,p,2)
    assert w.egg_sacs[key]['hatched'] and len(w.spiders)==4
    children=w.egg_sacs[key]['children']
    assert {w.spiders[c] for c in children}=={'male','female'}
    assert all(w.hunger[c]==.8 and p[c].hunter.hidden is None for c in children)
    assert not any(r['actor'] in children for r in rows)
    assert all(w.tokens[SPIDER][c]==token for c,token in old.items())
    ecology_round(w,p,3)
    assert len(w.spiders)==4 and all(p[c].hunter.hidden is not None for c in children)


def test_shadowbeast_is_replaced_once_and_old_corpse_is_preserved():
    w,p=world_and_policies();root='freight_shadowbeast';pos=w.positions[root]
    w.config.characters[root].character.current_hp=0;w.active.discard(root)
    rows=ecology_round(w,p,0);new='freight_shadowbeast_2'
    assert new in w.active and new in p and w.positions[new]!=pos
    assert not any(r['actor']==new for r in rows)
    assert w.positions[root]==pos and w.config.characters[root].character.current_hp==0
    ecology_round(w,p,1)
    assert len([c for c in w.positions if c.startswith(root)])==2
    w.config.characters[new].character.current_hp=0;w.active.discard(new)
    ecology_round(w,p,2)
    assert 'freight_shadowbeast_3' in w.active


def test_starvation_damages_and_kills_without_resurrecting_spider():
    w,p=world_and_policies()
    class Wait:
        def choose(self,obs):return {'action':'wait'}
    p={c:Wait() for c in w.positions};w.active={SPIDER}
    # Leave inactive residents alive so this test doesn't ask to replace them.
    w.hunger[SPIDER]=1;w.config.characters[SPIDER].character.current_hp=2
    for t in range(STARVATION_INTERVAL-1):ecology_round(w,p,t)
    assert w.config.characters[SPIDER].character.current_hp==2
    ecology_round(w,p,STARVATION_INTERVAL-1)
    assert w.config.characters[SPIDER].character.current_hp==1
    for t in range(STARVATION_INTERVAL):ecology_round(w,p,STARVATION_INTERVAL+t)
    assert w.config.characters[SPIDER].character.current_hp==0 and SPIDER not in w.active
    assert any(e['event']=='starvation_death' for e in w.lifecycle_events)
    ecology_round(w,p,40)
    assert w.config.characters[SPIDER].character.current_hp==0


def test_feeding_before_round_end_resets_starvation_clock():
    w,p=world_and_policies();w.hunger[SPIDER]=1;w.starvation_ticks[SPIDER]=9
    w.positions[SPIDER]=(103,140,'Fungal Stores');rat='cave_rat_1';w.positions[rat]=(107,140,'Fungal Stores')
    w.config.characters[rat].character.current_hp=0;w.active.discard(rat)
    w.food[rat]={'wrap_turns':3,'remaining':.5}
    w.act(SPIDER,{'action':'feed','target':w.tokens[SPIDER][rat]})
    from spider_sim.lifecycle import finish_round
    hp=w.config.characters[SPIDER].character.current_hp;finish_round(w,p)
    assert w.starvation_ticks[SPIDER]==0 and w.config.characters[SPIDER].character.current_hp==hp


def test_lifecycle_viewer_replay_restart_and_sac_item_rendering(tmp_path,monkeypatch):
    from experiments.watch_level2 import LiveRun
    from spider_sim.pursuit import save
    import spider_sim.feeding as nutrition
    import spider_sim.lifecycle as life
    monkeypatch.setattr(nutrition,'HUNGER_PER_ROUND',.5)
    monkeypatch.setattr(life,'STARVATION_INTERVAL',2)
    model=PursuitModel(21);model.value_kind='negative_action_logit';weights=tmp_path/'weights.pt';save(model,weights)
    kwargs=dict(spider_model=weights,feeding_cycle=True,spider_pair=True,spider_local_motion=True,replenish_prey=True,random_prey=True,lifecycle=True)
    run=LiveRun(tmp_path/'live',42,20,**kwargs)
    for _ in range(4):run.step()
    restored=LiveRun(run.output,42,20,resume=True,**kwargs)
    assert restored.life_state()==run.life_state()
    assert any(run.world.starvation_ticks.values())
    restored.restart();assert restored.life_state()['egg_sacs']=={} and len(restored.world.spiders)==2
    key=lay(restored.world);restored.publish([])
    page=(restored.output/'echoing_caverns_level2_map.html').read_text()
    assert 'Spider Egg Sac 1' in page
    from experiments.watch_level2 import CONTROLS
    assert "cid.startsWith('freight_spider')" in CONTROLS


def test_crowded_nursery_defers_pair_without_partial_or_duplicate_births(monkeypatch):
    import spider_sim.lifecycle as life
    w,p=world_and_policies();key=lay(w);w.elapsed_seconds=18
    with monkeypatch.context() as patch:
        patch.setattr(life,'nearby_position',lambda *args,**kwargs:None)
        life.finish_round(w,p)
        assert not w.egg_sacs[key]['hatched'] and len(w.spiders)==2
    life.finish_round(w,p)
    assert w.egg_sacs[key]['hatched'] and len(w.spiders)==4
    life.finish_round(w,p)
    assert len(w.spiders)==4
    male=w.egg_sacs[key]['children'][0]
    w.positions[SPIDER]=(103,140,'Fungal Stores');w.positions[male]=(107,140,'Fungal Stores')
    w.hunger[SPIDER]=w.hunger[male]=0
    assert w.act(SPIDER,{'action':'court','target':w.tokens[SPIDER][male]})['status']=='incompatible_mate'


def test_exhausted_carcass_cleanup_preserves_edible_meals():
    from spider_sim.lifecycle import finish_round
    w,p=world_and_policies()
    empty,meal='cave_rat_1','cave_rat_2'
    for cid in (empty,meal):
        w.config.characters[cid].character.current_hp=0
        w.active.discard(cid)
    w.food[empty]={'remaining':0,'wrap_turns':3}
    w.food[meal]={'remaining':.5,'wrap_turns':3}
    finish_round(w,p)
    w.elapsed_seconds=24;finish_round(w,p)
    assert empty not in w.retired_carcasses
    w.elapsed_seconds=30;finish_round(w,p)
    assert empty in w.retired_carcasses and meal not in w.retired_carcasses
    assert empty in w.positions  # IDs remain reserved; no accidental resurrection.


def test_mating_now_has_twenty_turn_cooldown():
    w,p=world_and_policies()
    lay(w)
    assert w.mate_cooldown[SPIDER]-w.elapsed_seconds==120


@pytest.mark.parametrize('hunger,seeking',[(.099,True),(.1,False)])
def test_mate_seeking_starts_strictly_below_ten_percent(hunger,seeking):
    from spider_sim.djepa_ecology import senses
    w,p=world_and_policies()
    w.hunger[SPIDER]=hunger
    p[SPIDER].choose(senses(w,SPIDER))
    assert p[SPIDER].seeking_mate is seeking


def test_zone_webs_sense_local_presence_without_exposing_body_or_other_zones():
    w,p=world_and_policies();w.zone_webs=True
    w.positions[SPIDER]=(50,100,'Receiving Floor')
    w.positions[FEMALE_SPIDER]=(60,100,'Receiving Floor')
    obs=w.observation_for(SPIDER)
    contact=next(c for c in obs['contacts'] if c['track']==w.tokens[SPIDER][FEMALE_SPIDER] and c['modality']=='web')
    assert obs['self']['touching_web']
    assert 'body_dimensions_ft' not in contact and 'relative_position_ft' not in contact
    w.hunger[SPIDER]=0
    assert w.act(SPIDER,{'action':'vibrate_web'})['status']=='courtship_vibration'
    received=w.observation_for(FEMALE_SPIDER)
    signal=next(c for c in received['contacts'] if c['track']==w.tokens[FEMALE_SPIDER][SPIDER] and c['modality']=='web')
    assert signal['courtship_signal'] and signal['mate_signal']=='receptive'
    w.positions[FEMALE_SPIDER]=(60,100,'Fungal Stores')
    assert not any(c['track']==w.tokens[SPIDER][FEMALE_SPIDER] and c['modality']=='web' for c in w.observation_for(SPIDER)['contacts'])


def test_empty_carcasses_retire_immediately_but_edible_ones_remain():
    w,p=world_and_policies();w.carcass_decay_rounds=0
    empty,meal='cave_rat_1','cave_rat_2'
    for cid in (empty,meal):
        w.config.characters[cid].character.current_hp=0;w.active.discard(cid)
    w.food[empty]={'remaining':0,'wrap_turns':3}
    w.food[meal]={'remaining':.5,'wrap_turns':3}
    from spider_sim.lifecycle import finish_round
    finish_round(w,p)
    assert empty in w.retired_carcasses and meal not in w.retired_carcasses
