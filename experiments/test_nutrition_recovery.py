import pytest
from experiments.test_lifecycle import world_and_policies
from spider_sim.ecology import SPIDER,FEMALE_SPIDER
from spider_sim.lifecycle import recover,start_round


def test_food_supported_recovery_is_opt_in_capped_and_not_resurrection():
    w,_=world_and_policies();body=w.config.characters[SPIDER].character;body.current_hp=20;w.hunger[SPIDER]=.05
    recover(w);assert body.current_hp==20
    w.healing_hp_per_round=1;w.healing_hunger_limit=.1
    recover(w);assert body.current_hp==21
    body.current_hp=w.max_hp[SPIDER];recover(w);assert body.current_hp==w.max_hp[SPIDER]
    body.current_hp=0;recover(w);assert body.current_hp==0


def test_recovery_stops_at_ten_percent_unless_actually_feeding_and_resets_each_round():
    w,_=world_and_policies();w.healing_hp_per_round=1;w.healing_hunger_limit=.1
    body=w.config.characters[SPIDER].character;body.current_hp=20
    w.hunger[SPIDER]=.1;recover(w);assert body.current_hp==20
    w.hunger[SPIDER]=.8;w.fed_this_round.add(SPIDER);recover(w);assert body.current_hp==21
    start_round(w);assert not w.fed_this_round
    recover(w);assert body.current_hp==21


def test_experimental_cutoff_controls_both_signals_and_engine_admission():
    w,_=world_and_policies();w.zone_webs=True;w.mate_hunger_limit=.3
    w.positions[SPIDER]=(103,140,'Fungal Stores');w.positions[FEMALE_SPIDER]=(107,140,'Fungal Stores')
    w.hunger[SPIDER]=w.hunger[FEMALE_SPIDER]=.25
    contact=next(c for c in w.observation_for(SPIDER)['contacts'] if c['track']==w.tokens[SPIDER][FEMALE_SPIDER] and c['modality']=='vision')
    assert contact['mate_signal']=='receptive'
    cmd={'action':'court','target':w.tokens[SPIDER][FEMALE_SPIDER]}
    assert w.act(SPIDER,cmd)['status']=='courting'
    w.hunger[SPIDER]=.3
    assert w.act(SPIDER,cmd)['status']=='too_hungry_to_mate'


def test_successful_engine_feed_marks_recovery_even_above_ten_percent():
    w,_=world_and_policies();w.healing_hp_per_round=1;w.healing_hunger_limit=.1
    rat='cave_rat_1';w.positions[SPIDER]=(103,140,'Fungal Stores');w.positions[rat]=(107,140,'Fungal Stores')
    w.config.characters[rat].character.current_hp=0;w.active.discard(rat)
    w.food[rat]={'wrap_turns':3,'remaining':.5};w.hunger[SPIDER]=.8
    body=w.config.characters[SPIDER].character;body.current_hp=20
    result=w.act(SPIDER,{'action':'feed','target':w.tokens[SPIDER][rat]})
    assert result['status']=='fed' and SPIDER in w.fed_this_round
    recover(w);assert body.current_hp==21 and w.hunger[SPIDER]==pytest.approx(.7)
