"""Multi-resident Level-2 ecology harness using shared Driftmoor mechanics.

This is a scripted baseline, not a neural policy or a new checkpoint schema.
No cavern-fixture coordinates, synthetic web strip or Gil-specific body assumptions.
"""
from spider_sim.runtime import game_root
ROOT = game_root()
import math
import random
from pathlib import Path
from engine.activation import initial_dormant_set, check_activations
from engine.geometry import _check_impassable, _clamp_position, _check_boundary_crossing
from engine.md_loader import configure_from_md
from engine.sensory import observe_contacts, floor_relative_position
from engine.session import _movement_budget_feet, _clamp_to_authored_home
from engine.webs import limit_movement, repair_web, damage_web, live_webs
from models.body_geometry import Body, Orientation, body_from_identity, body_sockets
from models.space_time import Position
from spider_sim.spider_experiment import ResolvedCavernExperiment

SCENE = "scenes/echoing_caverns_level2.md"
SPIDER = "freight_spider"
OBSERVATION_SCHEMA = "ecology-private-contacts-v1"


class EcologyEncounter:
    def __init__(self, seed=0, scene=SCENE):
        self.seed = seed
        self.random = random.Random(seed)
        self.config = configure_from_md(ROOT / scene)
        self.geometry = self.config.scene_geometry
        if self.geometry is None:
            raise ValueError("Ecology requires scene geometry")
        self.positions = {cid:(p.x,p.y,self.geometry.zone_at(p) or "")
                          for cid,p in self.geometry.starting_positions.items()}
        missing = set(self.config.characters)-set(self.positions)
        if missing:
            raise ValueError(f"Residents without positions: {sorted(missing)}")
        self.heights = {cid:p.z for cid,p in self.geometry.starting_positions.items()}
        self.bodies = {cid:body_from_identity(cs.identity,Body(4,4,3,"unauthored fallback"))
                       for cid,cs in self.config.characters.items()}
        # The authored spider species has not yet acquired physical dimensions.
        self.bodies[SPIDER] = body_from_identity(self.config.characters[SPIDER].identity,
                                                Body(4,4,2,"existing spider proxy"))
        self.orientations = {cid:Orientation() for cid in self.positions}
        self.sockets = {cid:body_sockets(body) for cid,body in self.bodies.items()}
        self.active = set(self.config.characters)-initial_dormant_set(self.config.characters)
        self.max_hp = {cid:cs.character.current_hp for cid,cs in self.config.characters.items()}
        self.elapsed_seconds = 0
        self.last_results = {}
        self.tokens = {actor:{cid:f"contact_{n}" for n,cid in enumerate(sorted(self.positions),1) if cid!=actor}
                       for actor in self.positions}

    _stationary_attack = ResolvedCavernExperiment._stationary_attack

    def observation_for(self, actor):
        character = self.config.characters[actor].character
        sensed = observe_contacts(actor,self.geometry,self.positions,self.active,self.config.characters,
                                  heights=self.heights,items=self.config.items,
                                  body_heights={cid:b.height for cid,b in self.bodies.items()})
        own = floor_relative_position(self.geometry,self.positions[actor],self.heights[actor])
        contacts = []
        for entry in sensed["contacts"]:
            cid = entry["track_id"]
            if self.config.characters[cid].character.current_hp <= 0:
                continue
            public = {key:value for key,value in entry.items() if key!="track_id"}
            public["track"] = self.tokens[actor][cid]
            if entry["modality"] == "vision":
                target = floor_relative_position(self.geometry,self.positions[cid],self.heights[cid])
                public["relative_position_ft"] = [target.x-own.x,target.y-own.y,target.z-own.z]
                body = self.bodies[cid]
                public["body_dimensions_ft"] = [body.width,body.depth,body.height]
            contacts.append(public)
        # Web condition is inspected locally; remote damage is not clairvoyantly known.
        visual_range = max(self.geometry.ambient_sight_range(self.positions[actor][2]) or 0,30)
        webs = []
        for key,item in self.config.items.items():
            if "web_bounds" not in item.state or item.state["web_owner"]!=actor:
                continue
            if own.horizontal_distance_to(item.position)>visual_range or not self.geometry.can_see(own,item.position):
                continue
            webs.append({"item":key,"relative_position_ft":[item.position.x-own.x,item.position.y-own.y],
                         "integrity":float(item.state["web_hp"])/float(item.state["web_max_hp"])})
        body = self.bodies[actor]
        return {"schema":OBSERVATION_SCHEMA,"self":{"hp_fraction":character.current_hp/self.max_hp[actor],
                "body_dimensions_ft":[body.width,body.depth,body.height],
                "heading_deg":self.orientations[actor].heading},
                "contacts":contacts,"own_webs_in_view":webs,"last_result":self.last_results.get(actor)}

    def _target(self, actor, track):
        return next((cid for cid,token in self.tokens[actor].items() if token==track),None)

    def act(self, actor, command):
        if actor not in self.active or self.config.characters[actor].character.current_hp<=0:
            raise ValueError("Inactive or defeated actor")
        action = command.get("action")
        if action not in ("wait","move","attack","repair_web","damage_web"):
            raise ValueError("Unknown ecology motor action")
        obs = self.observation_for(actor)
        character = self.config.characters[actor].character
        origin = self.positions[actor]
        result = {"action":action,"status":"waited","admitted":False,"damage":0,"distance_ft":0.}
        if action == "move":
            heading = float(command["heading_deg"]);distance = float(command.get("distance_ft",10))
            if not math.isfinite(heading) or not math.isfinite(distance) or distance<0:
                raise ValueError("Movement must be finite and nonnegative")
            self.orientations[actor] = Orientation(heading=heading)
            angle = math.radians(heading)
            target = (origin[0]+distance*math.cos(angle),origin[1]+distance*math.sin(angle),"")
            budget = _movement_budget_feet(character.speed,in_combat=True)
            target = limit_movement(character,origin,target,budget,self.geometry,self.config.items,height=self.heights[actor])
            target = _clamp_position(target,origin,budget)
            target = _clamp_to_authored_home(actor,target,character,self.geometry)
            allowed,damage,notes,stop = _check_boundary_crossing(actor,origin,target,character,
                                                             self.geometry.boundaries,self.config.game_time)
            character.current_hp = max(0,character.current_hp-damage)
            if not allowed:
                target = (*stop,"") if stop else origin
            clear,notes,stop = _check_impassable(actor,origin,target,self.geometry.impassable,self.geometry.zones)
            if not clear:
                target = (*stop,"") if stop else origin
            zone = self.geometry.zone_at(Position(*target[:2]))
            if zone:
                self.positions[actor] = (target[0],target[1],zone)
            result["distance_ft"] = math.dist(origin[:2],self.positions[actor][:2])
            result["status"] = "moved" if result["distance_ft"]>1e-8 else "blocked"
        elif action == "attack":
            visible = [c for c in obs["contacts"] if c["modality"]=="vision" and c["track"]==command.get("target")]
            if not visible:
                result["status"] = "target_not_visible"
            else:
                contact = visible[0];dx,dy,dz = contact["relative_position_ft"]
                target = self._target(actor,contact["track"])
                own_mouth = self.sockets[actor]["bite"].origin[2]
                target_mid = dz+contact["body_dimensions_ft"][2]/2
                self.orientations[actor] = Orientation(heading=math.degrees(math.atan2(dy,dx)),
                    pitch=math.degrees(math.atan2(own_mouth-target_mid,max(.01,math.hypot(dx,dy)))))
                status,feedback,_ = self._stationary_attack(actor,target)
                result.update(status=status,admitted=feedback["attempted"],damage=feedback["damage"],
                              hit=feedback["hit"],target=contact["track"],
                              target_defeated=self.config.characters[target].character.current_hp<=0)
        elif action in ("repair_web","damage_web"):
            key = command.get("item");item = self.config.items.get(key)
            own = floor_relative_position(self.geometry,origin,self.heights[actor])
            if item is None or "web_bounds" not in item.state or own.horizontal_distance_to(item.position)>5 or not self.geometry.can_see(own,item.position):
                result["status"] = "web_not_reachable"
            elif action == "repair_web":
                if item.state["web_owner"]!=actor:
                    result["status"] = "not_web_owner"
                else:
                    result["repaired_hp"] = repair_web(item,1,actor)
                    result["status"] = "web_repaired"
            else:
                result["web_damage"] = damage_web(item,1)
                result["status"] = "web_damaged"
        if character.current_hp<=0:
            self.active.discard(actor)
        self.last_results[actor] = dict(result)
        # Only living, active residents may wake a dormant resident. A corpse
        # or another dormant marker on the observer map must not trigger it.
        live_positions={cid:p for cid,p in self.positions.items() if cid in self.active
                        and self.config.characters[cid].character.current_hp>0}
        fired=check_activations(characters=self.config.characters,active_characters=self.active,
                                current_positions=live_positions)
        if fired:result['activations']=[{'actor':cid,'reason':reason} for cid,reason in fired]
        return result


class GoalBaseline:
    """Explicit scripted goals for a baseline; does not read the world object.

    Appetite is an episode preference, not new nutritional depletion mechanics.
    Feed currently means pursue/attack, not corpse consumption or metabolism.
    """
    def __init__(self, appetite=1., seed=0):
        self.appetite = appetite
        self.random = random.Random(seed)
        self.goal = "rest"
        self.goal_turns = 0

    def choose(self, obs):
        if obs.get("schema")!=OBSERVATION_SCHEMA:
            raise ValueError("Wrong ecology observation schema")
        # Prefer measured shape over vibrations when both describe one contact.
        contacts = {c["track"]:c for c in obs["contacts"] if c["modality"]!="vision"}
        contacts.update({c["track"]:c for c in obs["contacts"] if c["modality"]=="vision"})
        nearest = min(contacts.values(),key=lambda c:c["range_ft"],default=None)
        previous = self.goal
        if obs["self"]["hp_fraction"]<.35:
            self.goal = "survive" if nearest else "rest"
        elif self.appetite>.5 and nearest:
            self.goal = "feed"
        elif any(w["integrity"]<1 for w in obs["own_webs_in_view"]):
            self.goal = "maintain_web"
        else:
            self.goal = "rest"
        self.goal_turns = self.goal_turns+1 if previous==self.goal else 1
        if self.goal=="survive":
            heading = nearest.get("bearing_deg")
            if heading is None:
                heading = obs["self"]["heading_deg"]
            return {"action":"move","heading_deg":heading+180,"distance_ft":15}
        if self.goal=="feed":
            if nearest["modality"]=="vision":
                dx,dy,_ = nearest["relative_position_ft"]
                distance = math.hypot(dx,dy)
                heading = math.degrees(math.atan2(dy,dx))
                if distance<=6:
                    return {"action":"attack","target":nearest["track"]}
                return {"action":"move","heading_deg":heading,"distance_ft":min(10,max(1,distance-5))}
            heading = nearest.get("bearing_deg")
            return {"action":"move","heading_deg":heading,"distance_ft":5} if heading is not None else {"action":"wait"}
        if self.goal=="maintain_web":
            web = min((w for w in obs["own_webs_in_view"] if w["integrity"]<1),
                      key=lambda w:math.hypot(*w["relative_position_ft"]))
            dx,dy = web["relative_position_ft"]
            if math.hypot(dx,dy)<=5:
                return {"action":"repair_web","item":web["item"]}
            return {"action":"move","heading_deg":math.degrees(math.atan2(dy,dx)),"distance_ft":5}
        return {"action":"wait"}


class ForagerBaseline:
    """Wander or flee visibly larger bodies; no hidden predator coordinates."""
    def __init__(self,seed=0):
        self.random = random.Random(seed)
    def choose(self,obs):
        own = math.prod(obs["self"]["body_dimensions_ft"])
        threats = [c for c in obs["contacts"] if c["modality"]=="vision"
                   and math.prod(c["body_dimensions_ft"])>own*1.5]
        if threats:
            nearest = min(threats,key=lambda c:c["range_ft"])
            dx,dy,_ = nearest["relative_position_ft"]
            return {"action":"move","heading_deg":math.degrees(math.atan2(dy,dx))+180,"distance_ft":10}
        return {"action":"move","heading_deg":self.random.randrange(360),"distance_ft":5}


def baseline_policies(world,seed):
    # Preserve existing active-policy seeds; dormant controllers wait unused
    # until the native activation trigger fires.
    ordered=sorted(world.active)+sorted(set(world.config.characters)-world.active)
    return {cid:(GoalBaseline(seed=seed+n) if cid in (SPIDER,"freight_shadowbeast")
                 else ForagerBaseline(seed+n)) for n,cid in enumerate(ordered)}


def ecology_round(world,policies,turn):
    records=[]
    for cid in sorted(list(world.active)):
        if cid not in world.active:
            continue
        policy=policies[cid]
        if getattr(policy,"neural",False):
            from spider_sim.djepa_ecology import senses
            obs=senses(world,cid)
        else:obs=world.observation_for(cid)
        command=policy.choose(obs)
        result=world.act(cid,command)
        records.append({"round":turn,"actor":cid,"observation":obs,"goal":getattr(policy,"goal","forage"),
                        "command":command,"result":result,
                        **({"controller_kind":"djepa","plan":policy.last_plan} if getattr(policy,"controller_kind",None)=="djepa" else {})})
    world.elapsed_seconds+=6
    return records


def run_ecology(seed=0,rounds=30):
    world=EcologyEncounter(seed)
    policies=baseline_policies(world,seed)
    records=[]
    for turn in range(rounds):
        records.extend(ecology_round(world,policies,turn))
    return {"schema":OBSERVATION_SCHEMA,"seed":seed,"scene":SCENE,"rounds":rounds,
            "method":"scripted private-sensory ecology baseline; no neural training",
            "final_hp":{cid:cs.character.current_hp for cid,cs in world.config.characters.items()},
            "records":records}
