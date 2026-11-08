"""Teacher decisions must depend on private evidence, not resolver admission."""
import copy
import pytest
from models.body_geometry import Orientation
from spider_sim.dual_encounter import DualEncounter, ACTORS, ACTIONS, SPIDER, PLAYER
from spider_sim.encounter_teacher import LegacyTeacher, SensoryTeacher
from spider_sim.encounter_evaluation import TeacherAgent, encounter, summarize

def ready(player_x=70, spider_height=0):
    world = DualEncounter(1,outages=False)
    world.positions[SPIDER] = (64,195,"Secret Chamber")
    world.positions[PLAYER] = (player_x,195,"Secret Chamber")
    world.heights[SPIDER] = spider_height
    world.concealed = False
    world.orientations[SPIDER] = Orientation(roll=180 if spider_height else 0,heading=0)
    world.orientations[PLAYER] = Orientation(heading=180)
    for cid in ACTORS:
        world.config.characters[cid].character.current_hp = world.max_hp[cid]
    return world

def test_far_contact_no_longer_authorizes_attack():
    world = ready(78)
    obs = world.observation_for(SPIDER)
    assert ACTIONS[LegacyTeacher().choose(SPIDER,obs)]=="attack"
    assert ACTIONS[SensoryTeacher().choose(SPIDER,obs)]=="advance"

@pytest.mark.parametrize("actor",ACTORS)
def test_near_target_attack_is_admitted(actor):
    world = ready()
    obs = world.observation_for(actor)
    action = SensoryTeacher().choose(actor,obs)
    assert ACTIONS[action]=="attack"
    assert world.act(actor,action)["admitted"]

def test_rear_contact_requires_heading_change():
    world = ready()
    world.orientations[SPIDER] = Orientation(heading=180)
    assert ACTIONS[SensoryTeacher().choose(SPIDER,world.observation_for(SPIDER))] =="face"

def test_ceiling_bite_pitches_toward_observed_body():
    world = ready(66,10)
    obs = world.observation_for(SPIDER)
    assert ACTIONS[SensoryTeacher().choose(SPIDER,obs)]=="look_down"

def test_vibration_does_not_invent_body_or_authorize_attack():
    world = ready()
    world.tick = 0
    world.seed = 6
    world.outages = True
    world.motion_tick[PLAYER] = 0
    obs = world.observation_for(SPIDER)
    assert obs["contact"]["source"]=="vibration"
    assert obs["contact"]["dimensions_ft"] is None
    assert obs["contact"]["elevation_deg"] is None
    assert ACTIONS[SensoryTeacher().choose(SPIDER,obs)]!="attack"

@pytest.mark.parametrize("heading,expected",[(0,"retreat"),(180,"retreat")])
def test_wounded_escape_uses_heading_and_increases_separation(heading,expected):
    world = ready()
    world.orientations[SPIDER] = Orientation(heading=heading)
    world.config.characters[SPIDER].character.current_hp = 8
    obs = world.observation_for(SPIDER)
    teacher = SensoryTeacher()
    teacher.update(obs)
    choice = teacher.choose(SPIDER,obs)
    assert ACTIONS[choice]==expected
    # This fixture leaves room to escape within the home boundary.
    before = abs(world.positions[PLAYER][0]-world.positions[SPIDER][0])
    decision = teacher.decision(SPIDER,obs)
    world.act(SPIDER,choice,orientation=decision["orientation"])
    assert world.orientations[SPIDER].heading==0
    after = abs(world.positions[PLAYER][0]-world.positions[SPIDER][0])
    assert after>before

def test_teacher_is_pure_and_compensates_own_motion_without_target_truth():
    world = ready(78)
    obs = world.observation_for(SPIDER)
    frozen = copy.deepcopy(obs)
    teacher = SensoryTeacher()
    teacher.update(obs)
    assert teacher.choose(SPIDER,obs)==teacher.choose(SPIDER,copy.deepcopy(obs))
    assert obs==frozen
    hidden = copy.deepcopy(obs)
    hidden["contact"] = None
    hidden["self"]["anchor_xy"][0] += 3
    old_range = teacher.contact["range_ft"]
    teacher.update(hidden)
    assert teacher.contact["range_ft"]<old_range
    assert teacher.contact["range_ft"]>old_range-3.1

def test_scoring_and_reset_are_reproducible():
    agents = {cid:TeacherAgent() for cid in ACTORS}
    first = encounter(93000,"ground",agents,rounds=4)
    assert first==encounter(93000,"ground",agents,rounds=4)
    summary = summarize([first])
    for cid in ACTORS:
        item = summary[cid]
        assert item["admitted"]+item["rejected_attacks"]==item["attacks"]
        assert item["damage_received"]==item["initial_hp"]-item["final_hp"]
        assert item["episodes"]==1

def test_unarmed_actor_does_not_request_an_attack():
    world = ready()
    obs = world.observation_for(SPIDER)
    obs["self"]["attack_profile"]["reach_ft"] = 0
    assert ACTIONS[SensoryTeacher().choose(SPIDER,obs)]!="attack"

def test_damage_scoring_caps_at_actual_hp_loss(monkeypatch):
    import spider_sim.encounter_evaluation as scoring
    world = ready()
    world.config.characters[SPIDER].character.current_hp = 2
    monkeypatch.setattr(scoring,"DualEncounter",lambda *args,**kwargs:world)
    def action(actor,index,**kwargs):
        assert actor==PLAYER
        world.config.characters[SPIDER].character.current_hp = 0
        return {"action":"attack","status":"hit","admitted":True,"damage":10}
    monkeypatch.setattr(world,"act",action)
    result = encounter(1,"ground",{cid:TeacherAgent() for cid in ACTORS},rounds=1)
    assert result["stats"][PLAYER]["rolled_damage"]==10
    assert result["stats"][PLAYER]["damage_dealt"]==2
    assert result["stats"][SPIDER]["damage_received"]==2
    assert result["stats"][SPIDER]["survived"]==0

def test_retreat_and_flee_are_distinct_intentions():
    world = ready()
    teacher = SensoryTeacher()
    world.config.characters[SPIDER].character.current_hp = 8
    obs = world.observation_for(SPIDER)
    teacher.update(obs)
    assert teacher.decision(SPIDER,obs)["intent"]=="retreat"
    assert teacher.decision(SPIDER,obs)["action"]==ACTIONS.index("retreat")
    world.config.characters[SPIDER].character.current_hp = 3
    obs = world.observation_for(SPIDER)
    decision = teacher.decision(SPIDER,obs)
    assert decision["intent"]=="flee"
    assert ACTIONS[decision["action"]]=="flee"
    assert decision["orientation"][2]==180

def test_flee_turns_and_moves_in_one_slot_and_persists_after_contact_expires():
    world = DualEncounter(2,outages=False,scenario="critical_gil")
    world.random.randint=lambda low,high:low
    teacher = SensoryTeacher()
    start = world.positions[PLAYER][:2]
    actions = []
    for _ in range(16):
        obs = world.observation_for(PLAYER)
        teacher.update(obs)
        decision = teacher.decision(PLAYER,obs)
        assert decision["intent"]=="flee"
        action = decision["action"]
        actions.append(ACTIONS[action])
        before_time = world.elapsed_seconds
        before_x = world.positions[PLAYER][0]
        result = world.act(PLAYER,action,orientation=decision["orientation"])
        assert world.elapsed_seconds-before_time==6
        assert world.positions[PLAYER][0]>=before_x  # Eventually reaches the cavern boundary.
        assert result["distance_moved_ft"]<=result["movement_budget_ft"]
    assert all(a=="flee" for a in actions)
    assert world.orientations[PLAYER].heading==0
    assert world.positions[PLAYER][0]>start[0]
    assert world.observation_for(PLAYER)["contact"] is None
    assert teacher.contact is None
    assert teacher.escape_heading==0

def test_flee_does_not_reverse_an_already_away_heading():
    world = ready()
    world.config.characters[SPIDER].character.current_hp = 3
    world.orientations[SPIDER] = Orientation(heading=180)
    teacher = SensoryTeacher()
    obs = world.observation_for(SPIDER)
    teacher.update(obs)
    assert ACTIONS[teacher.choose(SPIDER,obs)]=="flee"

def test_low_health_without_evidence_does_not_invent_an_escape_direction():
    world = ready()
    teacher = SensoryTeacher()
    obs = world.observation_for(SPIDER)
    obs["contact"] = None
    obs["self"]["hp_fraction"] = .1
    teacher.update(obs)
    assert teacher.escape_heading is None
    assert ACTIONS[teacher.choose(SPIDER,obs)]=="wait"

def test_evaluation_separates_teacher_intent_from_motion():
    agents = {cid:TeacherAgent() for cid in ACTORS}
    episode = encounter(104000,"critical_gil",agents,rounds=12)
    gil = episode["stats"][PLAYER]
    assert gil["intents"]["flee"]>0
    assert gil["flee_forward_moves"]>0
    assert gil["retreat_moves"]==0
    assert gil["flee_facing_away"]==gil["flee_forward_moves"]
    summary = summarize([episode])[PLAYER]
    assert summary["flee_away_heading_rate"]==1

@pytest.mark.parametrize("actor,scenario",[(SPIDER,"wounded_spider"),(PLAYER,"wounded_gil")])
def test_wounded_fixtures_exercise_retreat_with_focus(actor,scenario):
    world = DualEncounter(2,outages=False,scenario=scenario)
    teacher = SensoryTeacher()
    obs = world.observation_for(actor)
    assert obs["contact"]["source"]=="vision"
    teacher.update(obs)
    decision = teacher.decision(actor,obs)
    assert decision["intent"]=="retreat"
    assert ACTIONS[decision["action"]]=="retreat"
    heading = world.orientations[actor].heading
    from spider_sim.encounter_evaluation import separation
    before = separation(world)
    world.act(actor,decision["action"])
    assert separation(world)>before
    assert world.orientations[actor].heading==heading

def test_neural_motor_outputs_are_not_given_a_teacher_intent():
    class WaitAgent:
        def session(self):
            return self
        def choose(self, values):
            return ACTIONS.index("wait"),[0]*5
    episode = encounter(105000,"ground",{cid:WaitAgent() for cid in ACTORS},rounds=1)
    assert all(turn["evaluation"]["declared_intent"] is None for turn in episode["turns"])
    assert all(stats["intents"]=={"unreported":1} for stats in episode["stats"].values())

@pytest.mark.parametrize("actor",ACTORS)
@pytest.mark.parametrize("maneuver",["retreat","flee"])
def test_withdrawal_orients_and_translates_in_one_slot_from_any_facing(actor,maneuver):
    from spider_sim.encounter_evaluation import separation
    world = ready()
    teacher = SensoryTeacher()
    teacher.update(world.observation_for(actor))  # Private evidence before looking away.
    world.orientations[actor] = Orientation(heading=90)
    world.config.characters[actor].character.current_hp = round(
        world.max_hp[actor]*(0.1 if maneuver=="flee" else 0.25))
    obs = world.observation_for(actor)
    teacher.update(obs)
    command = teacher.decision(actor,obs)
    assert ACTIONS[command["action"]]==maneuver
    before_time,before_distance = world.elapsed_seconds,separation(world)
    result = world.act(actor,command["action"],orientation=command["orientation"])
    assert world.elapsed_seconds-before_time==6
    assert separation(world)>before_distance
    assert 0<result["distance_moved_ft"]<=result["movement_budget_ft"]+1e-9
    toward = 0 if actor==SPIDER else 180
    assert world.orientations[actor].heading==((toward+180)%360 if maneuver=="flee" else toward)

def test_neural_command_uses_predicted_memory_without_hidden_target_position():
    from spider_sim.encounter_teacher import predicted_orientation
    world = ready()
    obs = world.observation_for(SPIDER)
    obs["contact"] = None
    belief = [1,0.2,0,1,0]  # Network remembers a contact to the north.
    command = predicted_orientation(obs,ACTIONS.index("flee"),belief)
    assert command[2]==270
    world.positions[PLAYER]=(70,190,"Secret Chamber")
    assert predicted_orientation(obs,ACTIONS.index("flee"),belief)==command

def test_flee_exposes_rear_and_incoming_attacks_are_measured(monkeypatch):
    import spider_sim.encounter_evaluation as scoring
    world = ready(66)
    world.random.randint=lambda low,high:low
    world.config.characters[SPIDER].character.current_hp=3
    monkeypatch.setattr(scoring,"DualEncounter",lambda *args,**kwargs:world)
    class Scripted:
        def __init__(self,names):
            self.names=names
        def session(self):
            self.steps=iter(self.names)
            return self
        def choose(self,values):
            return ACTIONS.index(next(self.steps,"wait")),[0]*5
    agents={PLAYER:Scripted(["wait","attack"]),SPIDER:Scripted(["flee","wait"])}
    episode=encounter(1,"ground",agents,rounds=2)
    spider=episode["stats"][SPIDER]
    assert spider["flee_rear_exposed_moves"]==1
    assert spider["incoming_rear_attack_requests"]==2
    assert spider["incoming_rear_attacks_admitted"]==1  # Reaction before exit; later attack is out of reach.
    assert spider["incoming_rear_hp_damage"]==spider["damage_received"]==0

def test_in_range_rear_attack_is_admitted_and_counted(monkeypatch):
    import spider_sim.encounter_evaluation as scoring
    world = ready()
    world.orientations[SPIDER] = Orientation(heading=180)
    monkeypatch.setattr(scoring,"DualEncounter",lambda *args,**kwargs:world)
    class Attack:
        def session(self):
            return self
        def choose(self,values):
            return ACTIONS.index("attack"),[0]*5
    result=encounter(1,"ground",{cid:Attack() for cid in ACTORS},rounds=1)
    spider=result["stats"][SPIDER]
    assert spider["incoming_rear_attack_requests"]==1
    assert spider["incoming_rear_attacks_admitted"]==1
    assert spider["incoming_rear_hp_damage"]==spider["damage_received"]
