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

ACTORS = (SPIDER, PLAYER)
ACTIONS = ("wait","advance","retreat","turn_left","turn_right","look_up","look_down","attack")
# Sensory numbers; prior-action one-hot and feedback follow these 30 values.
BASE_SIZE = 30
INPUT_SIZE = BASE_SIZE+len(ACTIONS)+3
MEMORY_SIZE = 5


def angle_difference(a,b):
    return (a-b+180)%360-180


class DualEncounter(ResolvedCavernExperiment):
    """No shared transcript or opponent state enters the agent observation.

    Fixture additions: a connected web strip x=60..105,y=188..202, an opaque
    entrance web curtain at x=81, occasional explicit sensor outages, and
    concealment broken by spider movement or an admitted attack.
    """
    def __init__(self,seed=0,outages=True):
        super().__init__(seed=seed)
        self.seed = seed
        self.tick = 0
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
            ratio = rng.choice((0.25,0.6,1.0,1.0))
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
            step = 15 if visible else 45
            contact = {
                "source":"vision" if visible else "vibration",
                "range_ft":round(distance/5)*5,
                "bearing_deg":(round(bearing/step)*step)%360,
                "elevation_deg":round(elevation/15)*15 if visible else None,
                # No body estimate from vibration, and no entity/species ID.
                "dimensions_ft": [
                    round(value*2)/2 for value in (
                        self.bodies[other].width,self.bodies[other].depth,self.bodies[other].height
                    )
                ] if visible else None,
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
            own[0]/160,own[1]/240,float(self.feedback[actor][2]>0),
        ]
        assert len(values)==BASE_SIZE
        values += [float(self.last_action[actor]==a) for a in ACTIONS]+self.feedback[actor]
        return {
            "values":values,
            "contact":contact,
            "web_evidence":bool(web_cue),
            "self":{
                "hp_fraction":hp,"orientation_roll_pitch_heading_deg":[pose.roll,pose.pitch,pose.heading],
                "anchor_xy":list(own[:2]),
            },
        }

    def _stationary_attack(self,actor_id,target_id):
        observation = self.observation_for(actor_id)
        c = observation["contact"]
        if c is None or c["source"]!="vision":
            return "target_not_observable",{"attempted":False,"hit":None,"damage":0},[]
        return super()._stationary_attack(actor_id,target_id)

    def act(self,actor,action_index):
        if not 0<=action_index<len(ACTIONS):
            raise ValueError("Unknown action index")
        action = ACTIONS[action_index]
        pose = self.orientations[actor]
        destination = None
        orientation = None
        kind = action
        target_id = None
        before_hp = {cid:self.config.characters[cid].character.current_hp for cid in ACTORS}
        if action in ("advance","retreat"):
            direction = 1 if action=="advance" else -1
            x,y,_ = self.positions[actor]
            heading = math.radians(pose.heading)
            destination = (x+direction*6*math.cos(heading),y+direction*6*math.sin(heading))
            kind = "move"
        elif action in ("turn_left","turn_right","look_up","look_down"):
            heading,pitch = pose.heading,pose.pitch
            if action=="turn_left":
                heading += 30
            if action=="turn_right":
                heading -= 30
            if action=="look_up":
                pitch = max(-90,pitch-30)
            if action=="look_down":
                pitch = min(90,pitch+30)
            orientation = Orientation(pose.roll,pitch,heading%360)
            kind = "orient"
        elif action=="attack":
            target_id = PLAYER if actor==SPIDER else SPIDER
        result = super().step(
            actor,kind,destination,target_id=target_id,orientation=orientation,
        )
        self.tick += 1
        self.last_action[actor] = action
        combat = result["outcome"]["combat"] or {}
        moved = result["outcome"]["distance_moved_ft"]>0.1
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
                "admitted":bool(combat.get("attempted",False)),"damage":damage}


class ContactMemory:
    """Demonstration memory built only from this actor's observations."""
    def __init__(self):
        self.contact = None
        self.age = 100

    def update(self,observation):
        self.age += 1
        if observation["contact"]:
            self.contact = dict(observation["contact"])
            self.age = 0
        if self.age>6:
            self.contact = None
        if self.contact is None:
            return [0,0,0,0,0]
        c = self.contact
        bearing = math.radians(c["bearing_deg"])
        return [1,c["range_ft"]/60,math.cos(bearing),math.sin(bearing),
                (c.get("elevation_deg") or 0)/90]

    def choose(self,actor,observation):
        own = observation["self"]
        # Distinct authored survival thresholds; learned through imitation.
        threshold = 0.35 if actor==SPIDER else 0.45
        if own["hp_fraction"]<threshold:
            return ACTIONS.index("retreat")
        c = observation["contact"] or self.contact
        roll,pitch,heading = own["orientation_roll_pitch_heading_deg"]
        if c:
            delta = angle_difference(c["bearing_deg"],heading)
            if abs(delta)>20:
                return ACTIONS.index("turn_left" if delta>0 else "turn_right")
            desired = -(c.get("elevation_deg") or 0)
            if desired-pitch>20:
                return ACTIONS.index("look_down")
            if desired-pitch<-20:
                return ACTIONS.index("look_up")
            visible = observation["contact"] and observation["contact"]["source"]=="vision"
            if visible and c["range_ft"]<=15:
                return ACTIONS.index("attack")
            return ACTIONS.index("advance" if actor==PLAYER or c["range_ft"]>15 else "wait")
        if actor==PLAYER:
            if observation["web_evidence"] and pitch>-30:
                return ACTIONS.index("look_up")
            return ACTIONS.index("advance")
        return ACTIONS.index("wait")


def demonstration(seed,rounds=18,exploration=0.15):
    world = DualEncounter(seed)
    memories = {cid:ContactMemory() for cid in ACTORS}
    randomizer = random.Random(seed+900000)
    records = {cid:[] for cid in ACTORS}
    for turn in range(rounds):
        for cid in (PLAYER,SPIDER):
            if any(world.config.characters[k].character.current_hp<=0 for k in ACTORS):
                break
            obs = world.observation_for(cid)
            belief = memories[cid].update(obs)
            teacher = memories[cid].choose(cid,obs)
            executed = randomizer.randrange(len(ACTIONS)) if randomizer.random()<exploration else teacher
            feedback = world.act(cid,executed)
            records[cid].append({
                "observation":obs["values"],"teacher_action":teacher,"executed_action":executed,
                "memory_target":belief,"contact_missing":obs["contact"] is None,
                "feedback":[float(feedback["admitted"]),feedback["damage"]/20],
            })
    for cid,rows in records.items():
        for i,row in enumerate(rows):
            row["next_observation"] = (
                rows[i+1]["observation"] if i+1<len(rows) else world.observation_for(cid)["values"]
            )
    return {"episode_id":seed,"actors":records}


def collect(count,start):
    return [demonstration(seed) for seed in range(start,start+count)]


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
                observation = world.observation_for(cid)
                action,belief = sessions[cid].choose(observation["values"])
                result = world.act(cid,action)
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
