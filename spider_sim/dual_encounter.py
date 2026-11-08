"""Two private sensory agents in an explicitly simplified cavern fixture."""
from spider_sim.runtime import game_root
game_root()
import math
import random

from spider_sim.spider_experiment import ResolvedCavernExperiment, SPIDER, PLAYER
from engine.sensory import floor_relative_position
from engine.loadout import draw
from models.body_geometry import Orientation, ActionSocket
from models.senses import resolve_senses
from models.space_time import Position
from engine.loadout import wielded_weapons
from spider_sim.withdrawal_rules import WithdrawalRules
from spider_sim.internal_condition import own_condition
from spider_sim.encounter_teacher import ContactMemory, LegacyTeacher, ACTIONS, action_orientation, predicted_orientation

ACTORS = (SPIDER, PLAYER)
# Sensory numbers; prior-action one-hot and feedback follow these 30 values.
BASE_SIZE = 30
INPUT_SIZE = BASE_SIZE+len(ACTIONS)+3
MEMORY_SIZE = 5
OBSERVATION_SCHEMA = "withdrawal-cornered-v6"


class DualEncounter(WithdrawalRules,ResolvedCavernExperiment):
    """No shared transcript or opponent state enters the agent observation.

    Fixture additions: a connected web strip x=60..105,y=188..202, an opaque
    entrance web curtain at x=81, occasional explicit sensor outages, and
    concealment broken by spider movement or an admitted attack.
    """
    def __init__(self,seed=0,outages=True,scenario="mixed"):
        super().__init__(seed=seed)
        self.seed = seed
        self.tick = 0
        self.turn_counts = {cid:0 for cid in ACTORS}
        self.turn_prepared = {cid:False for cid in ACTORS}
        self.reaction_available = {cid:True for cid in ACTORS}
        self.escape_blocked = {cid:False for cid in ACTORS}
        self.outages = outages
        self.motion_tick = {cid:-100 for cid in ACTORS}
        self.last_action = {cid:None for cid in ACTORS}
        self.feedback = {cid:[0,0,0] for cid in ACTORS}
        self.max_hp = {cid:self.config.characters[cid].character.current_hp for cid in ACTORS}
        rng = random.Random(seed)
        x,y = rng.uniform(88,118),rng.uniform(192,198)
        self.positions[PLAYER] = (x,y,self.geometry.zone_at(
            Position(x,y)) or "")
        self.heights[SPIDER] = rng.choice((0.0,10.0,10.0))
        self.orientations[SPIDER] = Orientation(
            roll=180 if self.heights[SPIDER] else 0,heading=rng.uniform(-45,45),
        )
        self.orientations[PLAYER] = Orientation(heading=180)
        self.concealed = self.heights[SPIDER]>0
        self.active = set(self.config.characters)
        for cid in ACTORS:
            ratio = rng.choice((0.1,0.25,0.6,1.0,1.0))
            self.config.characters[cid].character.current_hp = max(1,round(self.max_hp[cid]*ratio))
        gil = self.config.characters[PLAYER]
        # Explicit equipment fixture: draw an existing proficient weapon.
        sword = next(w for w in gil.weapons if w.id=="longsword")
        if not draw(gil,sword):
            raise RuntimeError("Could not draw Gil's existing longsword")
        body = self.bodies[PLAYER]
        self.sockets[PLAYER]["bite"] = ActionSocket(
            (body.depth/2,0,body.height*0.6),(1,0,0),
        )  # Compatibility slot, but sword reach originates from the hand.

        self.scenario = scenario
        if scenario not in ("mixed","ground","ceiling","wounded_spider","wounded_gil","facing_away","critical_spider","critical_gil"):
            raise ValueError("Unknown encounter scenario")
        if scenario in ("ground","facing_away"):
            self.heights[SPIDER] = 0
            self.orientations[SPIDER] = Orientation(heading=0)
            self.concealed = False
        if scenario == "ceiling":
            self.heights[SPIDER] = 10
            self.orientations[SPIDER] = Orientation(roll=180,heading=0)
            self.concealed = True
        if scenario == "facing_away":
            self.orientations[PLAYER] = Orientation(heading=0)

        if scenario in ("wounded_spider","wounded_gil","critical_spider","critical_gil"):
            # Close, visible encounter: a threat is sensed before withdrawing.
            self.positions[SPIDER] = (70,195,"Secret Chamber")
            self.positions[PLAYER] = (76,195,"Secret Chamber")
            self.heights[SPIDER] = 0
            self.orientations[SPIDER] = Orientation(heading=0)
            self.orientations[PLAYER] = Orientation(heading=180)
            self.concealed = False
            hurt = SPIDER if scenario.endswith("spider") else PLAYER
            ratio = 0.1 if scenario.startswith("critical") else 0.25
            for cid in ACTORS:
                self.config.characters[cid].character.current_hp = (
                    max(1,round(self.max_hp[cid]*ratio)) if cid==hurt else self.max_hp[cid])

    def attack_profile(self,actor):
        """Own equipment and socket are self-knowledge, fixed per role here."""
        weapons = [w for w in wielded_weapons(self.config.characters[actor]) if not w.is_ranged]
        return {"reach_ft":max((w.reach for w in weapons),default=0),
                "origin_local_ft":list(self.sockets[actor]["bite"].origin)}

    @staticmethod
    def in_web(xy):
        return 60<=xy[0]<=105 and 188<=xy[1]<=202

    def observation_for(self,actor):
        other = PLAYER if actor==SPIDER else SPIDER
        own = self.positions[actor]
        target = self.positions[other]
        ch = self.config.characters[actor].character
        own_point = floor_relative_position(self.geometry,own,self.heights[actor])
        target_point = floor_relative_position(self.geometry,target,self.heights[other])
        distance = own_point.distance_to(target_point)
        dx,dy,dz = target_point.x-own_point.x,target_point.y-own_point.y,target_point.z-own_point.z
        bearing = math.degrees(math.atan2(dy,dx))%360
        elevation = math.degrees(math.atan2(dz,math.hypot(dx,dy)))
        senses = resolve_senses(
            race=ch.race,senses=getattr(ch,"senses",None),features=getattr(ch,"features",None),
        )
        ambient = self.geometry.ambient_sight_range(own[2])
        sight = max(ambient if ambient is not None else 1000,senses.darkvision_ft)
        # Reproducible outages are a memory-training fixture, not a new game rule.
        outage = self.outages and (self.tick+self.seed+(0 if actor==SPIDER else 3))%11 in (6,7)
        visual_available = not outage
        curtain = (own[0]<81<target[0]) or (target[0]<81<own[0])
        visible = (
            visual_available and not curtain and distance<=sight
            and self.geometry.can_see(own_point,target_point)
            and self.config.characters[other].character.current_hp>0
        )
        if actor==PLAYER:
            forward = self.orientations[actor].rotate((1,0,0))
            alignment = sum(a*b for a,b in zip(forward,(dx,dy,dz)))/max(distance,1e-9)
            visible &= alignment>=math.cos(math.radians(65))
            visible &= not self.concealed
        web_available = actor==SPIDER and self.in_web(own)
        vibration = (
            web_available and self.in_web(target)
            and self.tick-self.motion_tick[other]<=2
            and self.config.characters[other].character.current_hp>0
        )
        # Webs are environmental evidence, not proof of a resident or identity.
        web_cue = visual_available and (
            actor==SPIDER or (own[0]<=115 and abs(own[1]-195)<15)
        )
        contact = None
        if visible or vibration:
            measured_distance, measured_bearing, measured_elevation = distance,bearing,elevation
            dimensions = None
            if visible:
                body = self.bodies[other]
                orientation = self.orientations[other]
                offset = orientation.rotate((0,0,body.height/2))
                cx,cy,cz = dx+offset[0],dy+offset[1],dz+offset[2]
                measured_distance = math.sqrt(cx*cx+cy*cy+cz*cz)
                measured_bearing = math.degrees(math.atan2(cy,cx))%360
                measured_elevation = math.degrees(math.atan2(cz,math.hypot(cx,cy)))
                # Visible silhouette's world-axis bounding extents; no identity
                # or hidden anchor/orientation is supplied to the controller.
                axes = [orientation.rotate(v) for v in (
                    (body.depth,0,0),(0,body.width,0),(0,0,body.height))]
                extents = [sum(abs(v[i]) for v in axes) for i in range(3)]
                dimensions = [max(0.5,round(extents[i]*2)/2) for i in (1,0,2)]
            step = 15 if visible else 45
            contact = {
                "source":"vision" if visible else "vibration",
                "range_ft":round(measured_distance) if visible else round(distance/5)*5,
                "bearing_deg":(round(measured_bearing/step)*step)%360,
                "elevation_deg":round(measured_elevation/15)*15 if visible else None,
                "dimensions_ft":dimensions,
            }
        pose = self.orientations[actor]
        own_body = self.bodies[actor]
        hp = ch.current_hp/self.max_hp[actor]
        # Explicit measurement and capability masks distinguish no signal
        # from unavailable sensors; absent measurements use zero.
        c = contact or {}
        sensed = contact is not None
        b = math.radians(c.get("bearing_deg",0))
        e = math.radians(c.get("elevation_deg") or 0)
        size = c.get("dimensions_ft") or (0,0,0)
        pose_values = [
            value for angle in (pose.roll,pose.pitch,pose.heading)
            for value in (math.sin(math.radians(angle)),math.cos(math.radians(angle)))
        ]
        values = [
            float(visual_available),float(visible),float(web_available),float(vibration),
            float(web_cue),float(sensed),c.get("range_ft",0)/60,
            math.cos(b) if sensed else 0,math.sin(b) if sensed else 0,
            float(visible),math.sin(e) if visible else 0,math.cos(e) if visible else 0,
            float(visible),*[v/10 for v in size],
            hp,self.heights[actor]/10,*pose_values,
            own_body.width/5,own_body.depth/5,own_body.height/6,
            own[0]/160,own[1]/240,float(self.escape_blocked[actor]),
        ]
        assert len(values)==BASE_SIZE
        values += [float(self.last_action[actor]==a) for a in ACTIONS]+self.feedback[actor]
        return {
            "values":values,
            "contact":contact,
            "web_evidence":bool(web_cue),
            "self":{
                "condition":own_condition(self.config.characters[actor].condition),
                "hp_fraction":hp,"orientation_roll_pitch_heading_deg":[pose.roll,pose.pitch,pose.heading],
                "anchor_xy":list(own[:2]),
                "attack_profile":self.attack_profile(actor),
                "speed_ft":self.speed_ft(actor),
                "reaction_available":self.reaction_available[actor],
                "escape_blocked":self.escape_blocked[actor],
            },
        }

    def _stationary_attack(self,actor_id,target_id):
        observation = self.observation_for(actor_id)
        c = observation["contact"]
        if c is None or c["source"]!="vision":
            return "target_not_observable",{"attempted":False,"hit":None,"damage":0},[]
        return super()._stationary_attack(actor_id,target_id)

    def begin_turn(self,actor):
        if not self.turn_prepared[actor]:
            self.reaction_available[actor] = True
            self.turn_prepared[actor] = True

    def act(self,actor,action_index,*,orientation=None):
        if not 0<=action_index<len(ACTIONS):
            raise ValueError("Unknown action index")
        if self.config.characters[actor].character.current_hp<=0:
            raise ValueError("A defeated actor cannot act")
        action = ACTIONS[action_index]
        self.begin_turn(actor)
        turn_start = self.turn_counts[actor]*6
        self.elapsed_seconds = turn_start
        pose = self.orientations[actor]
        destination = None
        if orientation is None:
            orientation = action_orientation(self.observation_for(actor),action_index)
        orientation = (orientation if isinstance(orientation,Orientation) else Orientation(*orientation)) if orientation is not None else None
        kind = action
        target_id = None
        before_hp = {cid:self.config.characters[cid].character.current_hp for cid in ACTORS}
        if action in ("advance","retreat","flee"):
            direction = -1 if action=="retreat" else 1
            x,y,_ = self.positions[actor]
            heading = math.radians((orientation or pose).heading)
            distance = self.speed_ft(actor)*(2 if action=="flee" else 1)
            destination = (x+direction*distance*math.cos(heading),y+direction*distance*math.sin(heading))
            kind = "move"
        elif action in ("face","look_up","look_down"):
            heading,pitch = pose.heading,pose.pitch
            if action=="look_up":
                pitch = max(-90,pitch-30)
            if action=="look_down":
                pitch = min(90,pitch+30)
            orientation = orientation if action=="face" else Orientation(pose.roll,pitch,heading%360)
            kind = "orient"
        elif action=="attack":
            target_id = PLAYER if actor==SPIDER else SPIDER
        previous_pose = self.orientations[actor]
        if kind in ("move","attack") and orientation is not None:
            self.orientations[actor] = orientation
        try:
            if kind=="move":
                result = self._move_with_reactions(
                    actor,destination,self.speed_ft(actor)*(2 if action=="flee" else 1),
                    action=="retreat",super().step)
            else:
                result = super().step(
                    actor,kind,destination,target_id=target_id,
                    orientation=orientation if kind=="orient" else None,
                )
        except Exception:
            self.orientations[actor] = previous_pose
            raise
        self.turn_prepared[actor] = False
        self.turn_counts[actor] += 1
        self.elapsed_seconds = 6*max(self.turn_counts.values())
        self.tick += 1
        self.last_action[actor] = action
        combat = result["outcome"]["combat"] or {}
        moved = result["outcome"]["distance_moved_ft"]>0.1
        if action=="flee" and self.config.characters[actor].character.current_hp>0:
            self.escape_blocked[actor] = result["outcome"]["distance_moved_ft"]+0.1 < self.speed_ft(actor)*2
        if moved:
            self.motion_tick[actor] = self.tick
        if actor==SPIDER and (moved or combat.get("attempted")):
            self.concealed = False
        damage = combat.get("damage",0)
        self.feedback[actor] = [float(combat.get("hit") is True),damage/20,0]
        for cid in ACTORS:
            suffered = before_hp[cid]-self.config.characters[cid].character.current_hp
            if suffered:
                self.feedback[cid][2] = suffered/self.max_hp[cid]
        return {"action":action,"status":result["outcome"]["status"],
                "admitted":bool(combat.get("attempted",False)),"damage":damage,
                "duration_seconds":6,"distance_moved_ft":result["outcome"]["distance_moved_ft"],
                "movement_budget_ft":result["outcome"]["movement_budget_ft"],
                "action_cost":"Disengage" if action=="retreat" else "Dash" if action=="flee" else action,
                "opportunity_attacks":result["outcome"].get("opportunity_attacks",[]),
                "elapsed_seconds":self.elapsed_seconds,"escape_blocked":self.escape_blocked[actor]}



def demonstration(seed,rounds=18,exploration=0.15,teacher="sensory",scenario="mixed"):
    if teacher not in ("sensory","legacy"):
        raise ValueError("Unknown teacher")
    teacher_name = teacher
    world = DualEncounter(seed,scenario=scenario)
    memories = {cid:(LegacyTeacher() if teacher=="legacy" else ContactMemory()) for cid in ACTORS}
    randomizer = random.Random(seed+900000)
    records = {cid:[] for cid in ACTORS}
    for turn in range(rounds):
        for cid in (PLAYER,SPIDER):
            if any(world.config.characters[k].character.current_hp<=0 for k in ACTORS):
                break
            world.begin_turn(cid)
            obs = world.observation_for(cid)
            belief = memories[cid].update(obs)
            teacher = memories[cid].choose(cid,obs)
            teacher_pose = (memories[cid].decision(cid,obs)["orientation"]
                            if teacher_name!="legacy" else action_orientation(obs,teacher))
            executed = randomizer.randrange(len(ACTIONS)) if randomizer.random()<exploration else teacher
            feedback = world.act(cid,executed,orientation=teacher_pose if executed==teacher else None)
            records[cid].append({
                "observation":obs["values"],"teacher_action":teacher,"executed_action":executed,
                "memory_target":belief,"contact_missing":obs["contact"] is None,
                "teacher_intent":memories[cid].intent(cid,obs) if teacher_name!="legacy" else None,
                "teacher_orientation":teacher_pose,
                "feedback":[float(feedback["admitted"]),feedback["damage"]/20],
            })
    for cid,rows in records.items():
        for i,row in enumerate(rows):
            row["next_observation"] = (
                rows[i+1]["observation"] if i+1<len(rows) else world.observation_for(cid)["values"]
            )
    return {"episode_id":seed,"teacher":teacher_name,"scenario":scenario,"actors":records}


def collect(count,start,scenarios=("mixed",)):
    if not scenarios:
        raise ValueError("At least one collection scenario is required")
    return [demonstration(seed,scenario=scenarios[i%len(scenarios)])
            for i,seed in enumerate(range(start,start+count))]


def neural_encounters(seeds,agents):
    """Each session owns an independent hidden state for each actor."""
    from collections import Counter
    totals = {cid:{"attacks":0,"admitted":0,"damage":0,"retreats":0,"actions":Counter()} for cid in ACTORS}
    traces = []
    for seed in seeds:
        world = DualEncounter(seed)
        sessions = {cid:agents[cid].session() for cid in ACTORS}
        episode = []
        for turn in range(18):
            for cid in (PLAYER,SPIDER):
                if any(world.config.characters[k].character.current_hp<=0 for k in ACTORS):
                    break
                world.begin_turn(cid)
                observation = world.observation_for(cid)
                action,belief = sessions[cid].choose(observation["values"])
                result = world.act(cid,action,orientation=predicted_orientation(observation,action,belief))
                item = totals[cid]
                item["actions"][ACTIONS[action]] += 1
                item["attacks"] += int(ACTIONS[action]=="attack")
                item["admitted"] += int(result["admitted"])
                item["damage"] += result["damage"]
                item["retreats"] += int(ACTIONS[action]=="retreat")
                episode.append({"actor":cid,"observation":observation,
                                "belief":belief,"result":result})
        traces.append({"episode_id":seed,"turns":episode})
    for stats in totals.values():
        stats["actions"] = dict(stats["actions"])
    return {"totals":totals,"traces":traces}
