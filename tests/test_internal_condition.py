import pytest
from models.condition import CharacterCondition,DimensionProfile
from spider_sim.internal_condition import own_condition
from spider_sim.dual_encounter import DualEncounter,ACTIONS,PLAYER,SPIDER


def test_snapshot_is_private_detached_and_honors_absent_dimensions():
    condition=CharacterCondition("actor","Actor")
    condition.profiles["cortisol"]=DimensionProfile(baseline=.7)
    condition.profiles["dopamine"]=DimensionProfile(baseline=0)
    snapshot=own_condition(condition)
    assert snapshot["deltas"]["cortisol"]==-.2
    assert "dopamine" not in snapshot["present"]
    snapshot["values"]["cortisol"]=1
    assert condition.state.cortisol==.5
    world=DualEncounter(2,outages=False,scenario="critical_gil")
    world.config.characters[PLAYER].condition.state.cortisol=.9
    own=world.observation_for(PLAYER)
    other=world.observation_for(SPIDER)
    assert own["self"]["condition"]["values"]["cortisol"]==.9
    assert other["self"]["condition"]["values"]["cortisol"]!=.9
    assert len(own["values"])==41


def ready(hit):
    world=DualEncounter(2,outages=False,scenario="critical_gil")
    for cid in (PLAYER,SPIDER):
        world.config.characters[cid].character.current_hp=world.max_hp[cid]
        world.config.characters[cid].condition=CharacterCondition(cid,cid)
    world.random.randint=(lambda low,high:high) if hit else (lambda low,high:low)
    return world


def test_combat_hit_updates_actor_and_target_conditions_without_llm():
    world=ready(True)
    result=world.act(SPIDER,ACTIONS.index("attack"))
    assert result["admitted"] and result["damage"]>0
    assert world.config.characters[SPIDER].condition.state.dopamine==pytest.approx(.58)
    assert world.config.characters[PLAYER].condition.state.cortisol==pytest.approx(.62)
    assert world.config.characters[PLAYER].condition.state.norepinephrine==pytest.approx(.58)


def test_miss_and_reaction_apply_conditions_once_with_volatility():
    world=ready(False)
    condition=world.config.characters[SPIDER].condition
    condition.profiles["cortisol"]=DimensionProfile(volatility=.5)
    result=world.act(PLAYER,ACTIONS.index("flee"))
    assert len(result["opportunity_attacks"])==1
    assert condition.state.cortisol==pytest.approx(.52)
    assert world.config.characters[PLAYER].condition.state.cortisol==.5


def test_unadmitted_attack_does_not_change_condition():
    world=ready(False)
    world.positions[PLAYER]=(110,195,world.positions[PLAYER][2])
    before={cid:world.config.characters[cid].condition.state.to_dict() for cid in (PLAYER,SPIDER)}
    assert not world.act(SPIDER,ACTIONS.index("attack"))["admitted"]
    assert before=={cid:world.config.characters[cid].condition.state.to_dict() for cid in (PLAYER,SPIDER)}
