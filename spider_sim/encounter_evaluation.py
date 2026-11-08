"""Matched encounter evaluation. Privileged state is used only for scoring."""
import math
from collections import Counter

from spider_sim.dual_encounter import DualEncounter, ACTORS, ACTIONS, PLAYER, SPIDER
from spider_sim.encounter_teacher import LegacyTeacher, SensoryTeacher, angle_difference, action_orientation, predicted_orientation
from engine.sensory import floor_relative_position

SCENARIOS = ("mixed", "ground", "ceiling", "wounded_spider", "wounded_gil", "facing_away", "critical_spider", "critical_gil")

class TeacherAgent:
    def __init__(self, legacy=False,fight_when_blocked=True):
        self.legacy = legacy
        self.fight_when_blocked = fight_when_blocked

    def session(self):
        return TeacherSession(LegacyTeacher() if self.legacy else SensoryTeacher(self.fight_when_blocked))

class TeacherSession:
    def __init__(self, teacher):
        self.teacher = teacher
        self.last_intent = None
        self.last_orientation = None

    def choose_observation(self, actor, observation):
        belief = self.teacher.update(observation)
        self.last_intent = self.teacher.intent(actor,observation) if isinstance(self.teacher,SensoryTeacher) else None
        action = self.teacher.choose(actor,observation)
        self.last_orientation = (self.teacher.decision(actor,observation)["orientation"]
                                 if isinstance(self.teacher,SensoryTeacher)
                                 else action_orientation(observation,action))
        return action,belief

def separation(world):
    points = [floor_relative_position(world.geometry, world.positions[cid], world.heights[cid]) for cid in ACTORS]
    return points[0].distance_to(points[1])


def rear_exposed(world, defender, attacker):
    """Opponent lies within 60 degrees of defender's rear; geometry only."""
    dx = world.positions[attacker][0]-world.positions[defender][0]
    dy = world.positions[attacker][1]-world.positions[defender][1]
    if math.hypot(dx,dy)<1e-9:
        return False
    bearing = math.degrees(math.atan2(dy,dx))
    return abs(angle_difference(bearing,world.orientations[defender].heading))>=120


def encounter(seed, scenario, agents, rounds=18):
    world = DualEncounter(seed, scenario=scenario)
    sessions = {cid:agents[cid].session() for cid in ACTORS}
    initial_hp = {cid:world.config.characters[cid].character.current_hp for cid in ACTORS}
    stats = {cid:{"attacks":0,"admitted":0,"rejected_attacks":0,"damage_dealt":0,"rolled_damage":0,
                  "escape_moves":0,"escape_increased_separation":0,"escape_separation_change_ft":0.,
                  "blocked_moves":0,"actions":Counter(),"intents":Counter(),
                  "retreat_moves":0,"retreat_facing_retained":0,"retreat_sight_retained":0,
                  "retreat_weapon_facing":0,
                  "retreat_separation_and_focus":0,"flee_forward_moves":0,
                  "flee_facing_away":0,"flee_progress_moves":0,"flee_rear_exposed_moves":0,
                  "incoming_rear_attack_requests":0,"incoming_rear_attacks_admitted":0,
                  "incoming_rear_hp_damage":0,"opportunity_attacks":0,
                  "opportunity_damage_dealt":0,"opportunity_damage_received":0,
                  "opportunity_interruptions":0} for cid in ACTORS}
    trace = []
    for _ in range(rounds):
        if any(world.config.characters[cid].character.current_hp<=0 for cid in ACTORS):
            break
        for cid in (PLAYER,SPIDER):
            if any(world.config.characters[k].character.current_hp<=0 for k in ACTORS):
                break
            world.begin_turn(cid)
            observation = world.observation_for(cid)
            session = sessions[cid]
            if hasattr(session,"choose_observation"):
                action,belief = session.choose_observation(cid,observation)
                command_pose = session.last_orientation
            else:
                action,belief = session.choose(observation["values"])
                command_pose = predicted_orientation(observation,action,belief)
            distance_before = separation(world)
            other = PLAYER if cid==SPIDER else SPIDER
            other_hp = world.config.characters[other].character.current_hp
            target_rear_exposed = rear_exposed(world,other,cid)
            result = world.act(cid,action,orientation=command_pose)
            distance_after = separation(world)
            item = stats[cid]
            name = ACTIONS[action]
            item["actions"][name] += 1
            for event in result.get("opportunity_attacks",[]):
                reactor = stats[event["reactor"]]
                reactor["opportunity_attacks"] += 1
                reactor["opportunity_damage_dealt"] += event["damage"]
                reactor["damage_dealt"] += event["damage"]
                reactor["rolled_damage"] += event["rolled_damage"]
                item["opportunity_damage_received"] += event["damage"]
                if event["rear_exposed"]:
                    item["incoming_rear_attack_requests"] += 1
                    item["incoming_rear_attacks_admitted"] += 1
                    item["incoming_rear_hp_damage"] += event["damage"]
            item["opportunity_interruptions"] += int(result["status"]=="defeated_by_opportunity_attack")
            if name=="attack" and target_rear_exposed:
                stats[other]["incoming_rear_attack_requests"] += 1
                stats[other]["incoming_rear_attacks_admitted"] += int(result["admitted"])
                stats[other]["incoming_rear_hp_damage"] += other_hp-world.config.characters[other].character.current_hp
            item["attacks"] += int(name=="attack")
            item["admitted"] += int(result["admitted"])
            item["rejected_attacks"] += int(name=="attack" and not result["admitted"])
            item["rolled_damage"] += result["damage"]
            item["damage_dealt"] += other_hp-world.config.characters[other].character.current_hp
            item["blocked_moves"] += int(name in ("advance","retreat","flee") and result["status"]=="blocked")
            declared_intent = getattr(session,"last_intent",None) or (name if name in ("retreat","flee") else None)
            item["intents"][declared_intent or "unreported"] += 1
            # Motion scoring uses privileged geometry, not the actor's decision.
            # Report explicit retreat/flee commands; other neural motives stay unknown.
            dx = world.positions[other][0]-world.positions[cid][0]
            dy = world.positions[other][1]-world.positions[cid][1]
            bearing = math.degrees(math.atan2(dy,dx))
            heading = world.orientations[cid].heading
            facing = abs(angle_difference(bearing,heading))<=30
            away = abs(angle_difference(bearing+180,heading))<=30
            after_observation = world.observation_for(cid)
            sight_before = observation["contact"] and observation["contact"]["source"]=="vision"
            sight_after = after_observation["contact"] and after_observation["contact"]["source"]=="vision"
            low = observation["self"]["hp_fraction"]<(0.35 if cid==SPIDER else 0.45)
            escaping = low and name in ("advance","retreat","flee")
            if low and name=="retreat":
                source_point = floor_relative_position(world.geometry,world.positions[cid],world.heights[cid])
                target_point = floor_relative_position(world.geometry,world.positions[other],world.heights[other])
                weapon_facing = world.sockets[cid]["bite"].evaluate(
                    (source_point.x,source_point.y,source_point.z),world.orientations[cid],
                    world.bodies[other],(target_point.x,target_point.y,target_point.z),
                    world.orientations[other])["facing"]
                item["retreat_moves"] += 1
                item["retreat_weapon_facing"] += int(weapon_facing)
                item["retreat_facing_retained"] += int(facing)
                item["retreat_sight_retained"] += int(bool(sight_before and sight_after))
                item["retreat_separation_and_focus"] += int(facing and weapon_facing and distance_after>distance_before+0.1)
            if low and name in ("advance","flee"):
                item["flee_forward_moves"] += 1
                item["flee_facing_away"] += int(away)
                item["flee_progress_moves"] += int(away and distance_after>distance_before+0.1)
                item["flee_rear_exposed_moves"] += int(rear_exposed(world,cid,other))
            if escaping:
                item["escape_moves"] += 1
                item["escape_increased_separation"] += int(distance_after>distance_before+0.1)
                item["escape_separation_change_ft"] += distance_after-distance_before
            trace.append({"actor":cid,"observation":observation,"belief":belief,"result":result,"command_orientation":command_pose,
                          "evaluation":{"separation_before_ft":distance_before,"separation_after_ft":distance_after,
                                        "declared_intent":declared_intent,"facing_threat":facing,"facing_away":away,
                                        "target_rear_exposed_before_attack":target_rear_exposed}})
            if getattr(session,"last_plan",None) is not None:
                trace[-1]["plan"] = session.last_plan
    for cid in ACTORS:
        hp = world.config.characters[cid].character.current_hp
        stats[cid].update({"damage_received":initial_hp[cid]-hp,"survived":int(hp>0),
                           "initial_hp":initial_hp[cid],"final_hp":hp,"episodes":1})
        stats[cid]["actions"] = dict(stats[cid]["actions"])
        stats[cid]["intents"] = dict(stats[cid]["intents"])
    return {"seed":seed,"scenario":scenario,"stats":stats,"turns":trace}

def summarize(episodes):
    result = {}
    for cid in ACTORS:
        totals = Counter()
        actions = Counter()
        intents = Counter()
        for episode in episodes:
            for key,value in episode["stats"][cid].items():
                if key=="actions":
                    actions.update(value)
                elif key=="intents":
                    intents.update(value)
                else:
                    totals[key] += value
        totals = dict(totals)
        totals["actions"] = dict(actions)
        totals["intents"] = dict(intents)
        for metric,numerator,denominator in (
            ("retreat_focus_rate","retreat_facing_retained","retreat_moves"),
            ("retreat_weapon_focus_rate","retreat_weapon_facing","retreat_moves"),
            ("retreat_sight_retention_rate","retreat_sight_retained","retreat_moves"),
            ("retreat_progress_with_focus_rate","retreat_separation_and_focus","retreat_moves"),
            ("flee_away_heading_rate","flee_facing_away","flee_forward_moves"),
            ("flee_progress_rate","flee_progress_moves","flee_forward_moves"),
            ("flee_rear_exposure_rate","flee_rear_exposed_moves","flee_forward_moves"),
        ):
            totals[metric] = totals[numerator]/totals[denominator] if totals[denominator] else None
        totals["attack_admission_rate"] = totals["admitted"]/totals["attacks"] if totals["attacks"] else None
        totals["escape_separation_increase_rate"] = (
            totals["escape_increased_separation"]/totals["escape_moves"] if totals["escape_moves"] else None)
        result[cid] = totals
    return result

def evaluate(learned, seeds, rounds=18):
    """Same initial fixtures; fixed teacher opponents also isolate each learner."""
    sensory = {cid:TeacherAgent() for cid in ACTORS}
    groups = {"legacy":{cid:TeacherAgent(True) for cid in ACTORS},"teacher":sensory}
    groups.update(learned)
    groups["flee_only_spider_vs_teacher"] = {SPIDER:TeacherAgent(fight_when_blocked=False),PLAYER:TeacherAgent()}
    groups["legacy_spider_vs_teacher"] = {SPIDER:TeacherAgent(True),PLAYER:TeacherAgent()}
    groups["legacy_gil_vs_teacher"] = {SPIDER:TeacherAgent(),PLAYER:TeacherAgent(True)}
    for kind,agents in learned.items():
        groups[kind+"_spider_vs_teacher"] = {SPIDER:agents[SPIDER],PLAYER:TeacherAgent()}
        groups[kind+"_gil_vs_teacher"] = {SPIDER:TeacherAgent(),PLAYER:agents[PLAYER]}
    reports,traces = {},{}
    for name,agents in groups.items():
        episodes = [encounter(seed,scenario,agents,rounds) for scenario in SCENARIOS for seed in seeds]
        reports[name] = {"total":summarize(episodes),
                         "scenarios":{scenario:summarize([e for e in episodes if e["scenario"]==scenario])
                                      for scenario in SCENARIOS}}
        traces[name] = episodes
    return reports,traces


def blocked_flight_behavior(traces):
    """Summarize choices in observed critically wounded, blocked-spider states."""
    result = {}
    for name,episodes in traces.items():
        counts = Counter()
        actions = Counter()
        for episode in episodes:
            for turn in episode["turns"]:
                own = turn["observation"]["self"]
                if (turn["actor"]!=SPIDER or own["hp_fraction"]>=0.2
                    or not own.get("escape_blocked",False)):
                    continue
                counts["decisions"] += 1
                action = turn["result"]["action"]
                actions[action] += 1
                if action=="attack":
                    counts["attack_requests"] += 1
                    counts["admitted_attacks"] += int(turn["result"]["admitted"])
        result[name] = {"decisions":counts["decisions"],
                        "attack_requests":counts["attack_requests"],
                        "admitted_attacks":counts["admitted_attacks"],
                        "flee_retries":actions["flee"],"actions":dict(actions)}
    return result
