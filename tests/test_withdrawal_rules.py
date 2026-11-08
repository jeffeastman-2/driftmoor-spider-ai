"""5e withdrawal accounting against real game movement and combat."""
import pytest
from models.body_geometry import Orientation
from spider_sim.dual_encounter import DualEncounter,ACTIONS,PLAYER,SPIDER


def ready():
    world=DualEncounter(2,outages=False,scenario="critical_gil")
    for cid in (PLAYER,SPIDER):
        world.config.characters[cid].character.current_hp=world.max_hp[cid]
    world.random.randint=lambda low,high:low  # Miss, but spend the reaction.
    return world

def test_disengage_prevents_reaction_and_uses_speed():
    world=ready()
    speed=world.speed_ft(PLAYER)
    start=world.positions[PLAYER][0]
    result=world.act(PLAYER,ACTIONS.index("retreat"))
    assert result["action_cost"]=="Disengage"
    assert result["opportunity_attacks"]==[]
    assert world.positions[PLAYER][0]-start==pytest.approx(speed)
    assert result["movement_budget_ft"]==speed
    assert world.reaction_available[SPIDER]

def test_dash_grants_twice_speed_and_reaction_occurs_before_exit():
    world=ready()
    speed=world.speed_ft(PLAYER)
    original=world.config.characters[PLAYER].character.speed
    start=world.positions[PLAYER][0]
    result=world.act(PLAYER,ACTIONS.index("flee"))
    assert result["action_cost"]=="Dash"
    assert result["distance_moved_ft"]==pytest.approx(2*speed)
    assert world.positions[PLAYER][0]-start==pytest.approx(2*speed)
    event,=result["opportunity_attacks"]
    assert start<event["position_xy"][0]<world.positions[PLAYER][0]
    assert not event["hit"]
    assert not world.reaction_available[SPIDER]
    assert world.config.characters[PLAYER].character.speed==original
    assert world.elapsed_seconds==6

def test_one_reaction_until_start_of_reactors_next_turn():
    world=ready()
    assert len(world.act(PLAYER,ACTIONS.index("flee"))["opportunity_attacks"])==1
    world.positions[PLAYER]=(76,195,"Secret Chamber")
    assert world.act(PLAYER,ACTIONS.index("flee"))["opportunity_attacks"]==[]
    world.begin_turn(SPIDER)
    assert world.reaction_available[SPIDER]
    world.act(SPIDER,ACTIONS.index("wait"))
    world.positions[PLAYER]=(76,195,"Secret Chamber")
    assert len(world.act(PLAYER,ACTIONS.index("flee"))["opportunity_attacks"])==1

def test_lethal_reaction_stops_mover_inside_reach():
    world=ready()
    world.config.characters[PLAYER].character.current_hp=1
    world.random.randint=lambda low,high:high
    result=world.act(PLAYER,ACTIONS.index("flee"))
    assert result["status"]=="defeated_by_opportunity_attack"
    assert world.config.characters[PLAYER].character.current_hp==0
    assert PLAYER not in world.active
    assert world._reach_distance(SPIDER,PLAYER,world.positions[PLAYER])<=5.5+1e-6
    assert result["distance_moved_ft"]<2*world.speed_ft(PLAYER)
    assert len(result["opportunity_attacks"])==1
    assert world.elapsed_seconds==6

def test_move_that_stays_in_reach_does_not_provoke():
    world=ready()
    world.config.characters[PLAYER].character.speed="1 ft"
    result=world.act(PLAYER,ACTIONS.index("flee"))
    assert result["distance_moved_ft"]==2
    assert result["opportunity_attacks"]==[]
    assert world.reaction_available[SPIDER]

def test_entering_and_leaving_reach_in_one_move_is_detected():
    world=ready()
    world.positions[PLAYER]=(60,195,"Secret Chamber")
    world.orientations[PLAYER]=Orientation(heading=0)
    result=world.act(PLAYER,ACTIONS.index("advance"))
    assert len(result["opportunity_attacks"])==1
    assert result["opportunity_attacks"][0]["position_xy"][0]>70

def test_unseen_mover_does_not_trigger_reaction():
    world=ready()
    world.orientations[PLAYER]=Orientation(heading=0)
    assert world.observation_for(PLAYER)["contact"] is None
    result=world.act(SPIDER,ACTIONS.index("flee"))
    assert result["opportunity_attacks"]==[]
    assert world.reaction_available[PLAYER]

def test_orientation_alone_does_not_provoke():
    world=ready()
    result=world.act(PLAYER,ACTIONS.index("face"),orientation=[0,0,0])
    assert result["opportunity_attacks"]==[]
    assert world.reaction_available[SPIDER]

def test_six_second_round_includes_both_actors():
    world=ready()
    world.act(PLAYER,ACTIONS.index("wait"))
    assert world.elapsed_seconds==6
    world.act(SPIDER,ACTIONS.index("wait"))
    assert world.elapsed_seconds==6
    world.act(PLAYER,ACTIONS.index("wait"))
    world.act(SPIDER,ACTIONS.index("wait"))
    assert world.elapsed_seconds==12
    assert world.turn_counts=={PLAYER:2,SPIDER:2}

def test_no_reaction_with_zero_speed_movement():
    world=ready()
    world.config.characters[PLAYER].character.speed="0 ft"
    result=world.act(PLAYER,ACTIONS.index("flee"))
    assert result["distance_moved_ft"]==0
    assert result["opportunity_attacks"]==[]

def test_blocked_movement_does_not_provoke_for_unreached_exit():
    from models.space_time import Polygon
    world=ready()
    world.geometry.impassable.append(Polygon([(77,190),(78,190),(78,200),(77,200)]))
    result=world.act(PLAYER,ACTIONS.index("flee"))
    assert world.positions[PLAYER][0]<77
    assert result["opportunity_attacks"]==[]
    assert world.reaction_available[SPIDER]

def test_reaction_damage_is_included_in_encounter_accounting():
    from spider_sim.encounter_evaluation import TeacherAgent,encounter
    episode=encounter(124000,"critical_gil",{cid:TeacherAgent() for cid in (PLAYER,SPIDER)},rounds=4)
    totals=episode["stats"]
    assert sum(s["damage_dealt"] for s in totals.values())==sum(s["damage_received"] for s in totals.values())
    assert sum(s["opportunity_damage_dealt"] for s in totals.values())==sum(s["opportunity_damage_received"] for s in totals.values())
    for cid in (PLAYER,SPIDER):
        item=totals[cid]
        assert item["opportunity_damage_dealt"]<=item["damage_dealt"]

def test_spider_fights_after_observing_blocked_flight():
    from spider_sim.encounter_teacher import SensoryTeacher
    world=ready()
    world.config.characters[SPIDER].character.current_hp=3
    teacher=SensoryTeacher()
    initial=world.observation_for(SPIDER)
    teacher.update(initial)
    assert teacher.intent(SPIDER,initial)=="flee"  # No hidden knowledge of walls.
    attempt=world.act(SPIDER,ACTIONS.index("flee"))
    assert attempt["escape_blocked"]
    assert world.observation_for(SPIDER)["values"][29]==1
    x,y,_=world.positions[SPIDER]
    world.positions[PLAYER]=(x+6,y,"Secret Chamber")
    obs=world.observation_for(SPIDER)
    teacher.update(obs)
    command=teacher.decision(SPIDER,obs)
    assert command["intent"]=="cornered"
    assert ACTIONS[command["action"]]=="attack"
    result=world.act(SPIDER,command["action"],orientation=command["orientation"])
    assert result["admitted"]
    assert world.observation_for(SPIDER)["self"]["escape_blocked"]  # Survives stationary turns.

def test_flee_only_control_ignores_blocked_flight_feedback():
    from spider_sim.encounter_teacher import SensoryTeacher
    world=ready()
    world.config.characters[SPIDER].character.current_hp=3
    obs=world.observation_for(SPIDER)
    obs["self"]["escape_blocked"]=True
    teacher=SensoryTeacher(fight_when_blocked=False)
    teacher.update(obs)
    assert teacher.intent(SPIDER,obs)=="flee"
    assert ACTIONS[teacher.choose(SPIDER,obs)]=="flee"

def test_successful_flight_clears_own_blocked_escape_feedback():
    world=ready()
    world.escape_blocked[PLAYER]=True
    result=world.act(PLAYER,ACTIONS.index("flee"))
    assert result["distance_moved_ft"]==2*world.speed_ft(PLAYER)
    assert not world.escape_blocked[PLAYER]

def test_cornered_spider_cannot_attack_a_distant_contact():
    from spider_sim.encounter_teacher import SensoryTeacher
    world=ready()
    world.config.characters[SPIDER].character.current_hp=3
    world.escape_blocked[SPIDER]=True
    world.positions[PLAYER]=(80,195,"Secret Chamber")
    obs=world.observation_for(SPIDER)
    teacher=SensoryTeacher()
    teacher.update(obs)
    assert teacher.intent(SPIDER,obs)=="cornered"
    assert ACTIONS[teacher.choose(SPIDER,obs)]!="attack"

def test_blocked_flight_analysis_distinguishes_retrying_control(monkeypatch):
    import spider_sim.encounter_evaluation as scoring
    world=ready()
    world.config.characters[SPIDER].character.current_hp=3
    monkeypatch.setattr(scoring,"DualEncounter",lambda *args,**kwargs:world)
    class Wait:
        def session(self):
            return self
        def choose(self,values):
            return ACTIONS.index("wait"),[0]*5
    episode=scoring.encounter(1,"ground",{
        PLAYER:Wait(),SPIDER:scoring.TeacherAgent(fight_when_blocked=False)},rounds=4)
    summary=scoring.blocked_flight_behavior({"control":[episode]})["control"]
    assert summary["decisions"]>0
    assert summary["flee_retries"]==summary["decisions"]
    assert summary["attack_requests"]==0
