"""Multi-resident Level-2 ecology harness using shared Driftmoor mechanics.

This is a scripted baseline, not a neural policy or a new checkpoint schema.
No cavern-fixture coordinates, synthetic web strip or Gil-specific body assumptions.
"""
from spider_sim.runtime import game_root
ROOT = game_root()
import math
import copy
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
FEMALE_SPIDER = "freight_spider_female"
OBSERVATION_SCHEMA = "ecology-private-contacts-v1"


class EcologyEncounter:
    def __init__(self, seed=0, scene=SCENE,*,replenish_prey=False,feeding_cycle=False,random_prey=False,spider_pair=False):
        if spider_pair and not feeding_cycle:raise ValueError("Spider pair requires feeding cycle")
        self.random_prey=random_prey;self.spider_pair=spider_pair
        self.spawn_random=random.Random(seed+81721)
        self.spiders={SPIDER:"male"}
        if spider_pair:self.spiders[FEMALE_SPIDER]="female"
        self.courtship={};self.mate_cooldown={};self.mating_events=[]
        self.feeding_cycle=feeding_cycle
        from spider_sim.feeding import INITIAL_HUNGER
        self.hunger={cid:INITIAL_HUNGER for cid in self.spiders} if feeding_cycle else {}
        self.food={}
        self.seed = seed
        self.random = random.Random(seed)
        self.config = configure_from_md(ROOT / scene)
        self.geometry = self.config.scene_geometry
        self.web_signals={}
        if self.geometry is None:
            raise ValueError("Ecology requires scene geometry")
        self.positions = {cid:(p.x,p.y,self.geometry.zone_at(p) or "")
                          for cid,p in self.geometry.starting_positions.items()}
        if spider_pair:
            female=copy.deepcopy(self.config.characters[SPIDER])
            female.character.id=FEMALE_SPIDER;female.character.name='Ms Spider'
            self.config.characters[FEMALE_SPIDER]=female
            self.positions[FEMALE_SPIDER]=self.random_floor_position(Body(4,4,2,"female spider proxy"),exclude=list(self.positions.values()))
        missing = set(self.config.characters)-set(self.positions)
        if missing:
            raise ValueError(f"Residents without positions: {sorted(missing)}")
        self.heights = {cid:p.z for cid,p in self.geometry.starting_positions.items()}
        if spider_pair:self.heights[FEMALE_SPIDER]=0.
        self.bodies = {cid:body_from_identity(cs.identity,Body(4,4,3,"unauthored fallback"))
                       for cid,cs in self.config.characters.items()}
        # The authored spider species has not yet acquired physical dimensions.
        self.bodies[SPIDER] = body_from_identity(self.config.characters[SPIDER].identity,
                                                Body(4,4,2,"existing spider proxy"))
        if spider_pair:self.bodies[FEMALE_SPIDER]=self.bodies[SPIDER]
        self.orientations = {cid:Orientation() for cid in self.positions}
        self.sockets = {cid:body_sockets(body) for cid,body in self.bodies.items()}
        self.active = set(self.config.characters)-initial_dormant_set(self.config.characters)
        self.max_hp = {cid:cs.character.current_hp for cid,cs in self.config.characters.items()}
        self.elapsed_seconds = 0
        self.last_results = {}
        self.replenish_prey=replenish_prey;self.replaced_prey=set();self.last_births=[]
        self.prey_roots={cid:cid for cid in self.positions if cid.startswith(('cave_rat_','cave_cricket_'))}
        self.prey_templates={cid:copy.deepcopy(self.config.characters[cid]) for cid in self.prey_roots}
        self.prey_homes={cid:self.positions[cid] for cid in self.prey_roots}
        self.contact_ids={cid:f'contact_{n}' for n,cid in enumerate(sorted(self.positions),1)}
        self.tokens = {actor:{cid:f"contact_{n}" for n,cid in enumerate(sorted(self.positions),1) if cid!=actor}
                       for actor in self.positions}

    def random_floor_position(self,body,exclude=()):
        """Choose a zone uniformly, then reject rock, walls and occupied space.

        Separate seeded spawn RNG keeps combat dice independent of placement.
        This samples authored floor geometry; it does not certify route connectivity.
        """
        zones=list(self.geometry.zones.items());self.spawn_random.shuffle(zones)
        # Whole-scene Solid Rock is a background mask: zones carve walkable
        # rooms out of it, as in the engine movement model.
        obstacles=[o for o in self.geometry.impassable if not all(o.contains(p.centroid()) for p in self.geometry.zones.values())]
        for name,polygon in zones:
            xs=[v[0] for v in polygon.vertices];ys=[v[1] for v in polygon.vertices]
            for _ in range(300):
                x=self.spawn_random.uniform(min(xs),max(xs));y=self.spawn_random.uniform(min(ys),max(ys))
                points=[Position(x+dx,y+dy) for dx in (-body.width/2,0,body.width/2) for dy in (-body.depth/2,0,body.depth/2)]
                if not all(polygon.contains(p) and not any(obstacle.contains(p) for obstacle in obstacles) for p in points):continue
                occupied=list(exclude)
                if hasattr(self,'active'):occupied += [p for cid,p in self.positions.items() if cid in self.active]
                if any(math.dist((x,y),p[:2])<max(2,body.width) for p in occupied):continue
                return (x,y,name)
        raise ValueError('No valid unoccupied floor position for spawn')

    def replace_dead_prey(self):
        """One newborn per dead rat/cricket, at end of round; corpses remain."""
        self.last_births=[]
        if not self.replenish_prey:return []
        dead=[cid for cid in self.prey_roots if cid not in self.replaced_prey
              and self.config.characters[cid].character.current_hp<=0]
        for parent in sorted(dead):
            root=self.prey_roots[parent];prefix='cave_rat_' if root.startswith('cave_rat_') else 'cave_cricket_'
            n=max(int(cid[len(prefix):]) for cid in self.positions if cid.startswith(prefix))+1
            newborn=prefix+str(n);cs=copy.deepcopy(self.prey_templates[root])
            cs.character.current_hp=self.max_hp[root]
            cs.character.id=newborn;cs.character.name=newborn.replace('_',' ').title()
            self.active.discard(parent)
            self.config.characters[newborn]=cs;self.positions[newborn]=self.random_floor_position(self.bodies[root]) if self.random_prey else self.prey_homes[root]
            self.heights[newborn]=0. if self.random_prey else self.heights[root];self.bodies[newborn]=self.bodies[root]
            self.orientations[newborn]=Orientation();self.sockets[newborn]=self.sockets[root]
            self.max_hp[newborn]=self.max_hp[root];self.active.add(newborn)
            self.prey_roots[newborn]=root;self.replaced_prey.add(parent)
            self.contact_ids[newborn]=f'contact_{len(self.contact_ids)+1}'
            self.tokens={actor:{cid:token for cid,token in self.contact_ids.items() if cid!=actor} for actor in self.positions}
            self.last_births.append({'actor':newborn,'replaces':parent,'home':root})
        return self.last_births

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
            if self.spider_pair and actor in self.spiders and cid in self.spiders and entry['modality']=='vision':
                public['conspecific']=True
                public['mate_signal']='receptive' if self.hunger[cid]<=.2 and self.config.characters[cid].character.current_hp/self.max_hp[cid]>=.35 and self.elapsed_seconds>=self.mate_cooldown.get(cid,0) else 'unreceptive'
            if self.spider_pair and actor in self.spiders and entry['modality']=='web' and self.web_signals.get(cid,{}).get('until',-1)>self.elapsed_seconds:
                public.update(conspecific=True,courtship_signal=True,mate_signal=self.web_signals[cid]['readiness'])
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
        from engine.webs import _touching
        touching_web=any(_touching(own,item,body.height) for item in live_webs(self.config.items).values())
        food=[]
        if self.feeding_cycle and actor in self.spiders:
            from spider_sim.feeding import WRAP_TURNS
            for cid,state in self.food.items():
                target=floor_relative_position(self.geometry,self.positions[cid],self.heights[cid])
                vector=[target.x-own.x,target.y-own.y,target.z-own.z]
                distance=math.sqrt(sum(v*v for v in vector))
                if state['remaining']>0 and distance<=visual_range and self.geometry.can_see(own,target):
                    food.append(dict(track=self.tokens[actor][cid],range_ft=distance,
                                     relative_position_ft=vector,wrapped=state['wrap_turns']>=WRAP_TURNS,
                                     remaining=state['remaining']))
        return {**({'food_in_view':food} if self.feeding_cycle and actor in self.spiders else {}),"schema":OBSERVATION_SCHEMA,"self":{"hp_fraction":character.current_hp/self.max_hp[actor],
                "body_dimensions_ft":[body.width,body.depth,body.height],
                "heading_deg":self.orientations[actor].heading,
                **({"hunger":self.hunger[actor],"touching_web":touching_web} if actor in self.hunger else {})},
                "contacts":contacts,"own_webs_in_view":webs,"last_result":self.last_results.get(actor)}

    def _target(self, actor, track):
        return next((cid for cid,token in self.tokens[actor].items() if token==track),None)

    def act(self, actor, command):
        if actor not in self.active or self.config.characters[actor].character.current_hp<=0:
            raise ValueError("Inactive or defeated actor")
        action = command.get("action")
        if action not in ("wait","move","attack","repair_web","damage_web","wrap","feed","court","vibrate_web"):
            raise ValueError("Unknown ecology motor action")
        if action!='court':
            for pair in list(self.courtship):
                if actor in pair:self.courtship.pop(pair,None)
        obs = self.observation_for(actor)
        character = self.config.characters[actor].character
        origin = self.positions[actor]
        result = {"action":action,"status":"waited","admitted":False,"damage":0,"distance_ft":0.}
        if action == "move":
            heading = float(command["heading_deg"]);distance = float(command.get("distance_ft",10))
            if not math.isfinite(heading) or not math.isfinite(distance) or distance<0:
                raise ValueError("Movement must be finite and nonnegative")
            # Legacy commands retain their original semantics for historical
            # comparisons. New pursuit commands separate travel from facing.
            keep_facing=command.get('keep_facing',False)
            face_track=command.get('face_track')
            if not keep_facing and not face_track:
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
            if keep_facing or face_track:
                turn_reason=None
                # Aim only using an already visible measurement and actual own
                # displacement; never fetch the target's hidden world position.
                contact=next((c for c in obs['contacts'] if c['modality']=='vision' and c['track']==face_track),None)
                if command.get('face_travel'):
                    old=self.orientations[actor]
                    self.orientations[actor]=Orientation(heading=heading,pitch=old.pitch,roll=old.roll)
                    turn_reason=command.get('turn_reason','face_travel')
                elif contact:
                    dx,dy,_=contact['relative_position_ft']
                    dx-=self.positions[actor][0]-origin[0];dy-=self.positions[actor][1]-origin[1]
                    if math.hypot(dx,dy)>1e-8:
                        old=self.orientations[actor]
                        self.orientations[actor]=Orientation(heading=math.degrees(math.atan2(dy,dx)),pitch=old.pitch,roll=old.roll)
                        turn_reason='track_visible_pursuit_contact'
                result.update(travel_heading_deg=heading,body_heading_deg=self.orientations[actor].heading,
                              turn_reason=turn_reason)
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
        elif action=='vibrate_web':
            if not self.spider_pair or actor not in self.spiders or not obs['self'].get('touching_web'):
                result['status']='not_touching_web'
            else:
                receptive=self.hunger[actor]<=.2 and character.current_hp/self.max_hp[actor]>=.35 and self.elapsed_seconds>=self.mate_cooldown.get(actor,0)
                self.web_signals[actor]={'until':self.elapsed_seconds+12,'readiness':'receptive' if receptive else 'unreceptive'}
                result['status']='courtship_vibration'
        elif action=='court':
            target=self._target(actor,command.get('target'))
            contact=next((c for c in obs['contacts'] if c['track']==command.get('target') and c['modality']=='vision'),None)
            result['target']=command.get('target')
            if actor not in self.spiders or target not in self.spiders or not contact or contact['range_ft']>6:
                result['status']='mate_not_reachable'
            elif self.hunger[actor]>.2 or self.hunger[target]>.2:
                result['status']='too_hungry_to_mate';self.courtship.pop(tuple(sorted((actor,target))),None)
            elif any(self.config.characters[c].character.current_hp/self.max_hp[c]<.35 or self.elapsed_seconds<self.mate_cooldown.get(c,0) for c in (actor,target)):
                result['status']='not_receptive'
            else:
                pair=tuple(sorted((actor,target)));progress=self.courtship.get(pair,0)+1;self.courtship[pair]=progress
                result.update(status='courtship_accepted' if progress>=3 else 'courting',courtship_turns=progress)
                if progress>=3:
                    for c in pair:self.mate_cooldown[c]=self.elapsed_seconds+3600
                    self.mating_events.append({'actors':list(pair),'time':self.elapsed_seconds,'event':'courtship_accepted'})
                    self.courtship.pop(pair,None)
        elif action in ('wrap','feed'):
            from spider_sim.feeding import WRAP_TURNS,FEED_PER_TURN
            target=self._target(actor,command.get('target'))
            meal=next((c for c in obs.get('food_in_view',[]) if c['track']==command.get('target')),None)
            result['target']=command.get('target')
            if not self.feeding_cycle or actor not in self.spiders or meal is None or meal['range_ft']>6:
                result['status']='food_not_reachable'
            else:
                state=self.food[target]
                if action=='wrap':
                    state['wrap_turns']=min(WRAP_TURNS,state['wrap_turns']+1)
                    result.update(status='wrapped' if state['wrap_turns']==WRAP_TURNS else 'wrapping',wrap_turns=state['wrap_turns'])
                elif state['wrap_turns']<WRAP_TURNS:
                    result['status']='food_not_wrapped'
                else:
                    amount=min(FEED_PER_TURN,state['remaining'],self.hunger[actor])
                    state['remaining']=max(0.,state['remaining']-amount)
                    if state['remaining']<1e-9:state['remaining']=0.
                    self.hunger[actor]=max(0.,self.hunger[actor]-amount)
                    result.update(status='fed' if amount>0 else 'full',consumed=amount,hunger=self.hunger[actor],food_remaining=state['remaining'])
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
        if self.feeding_cycle:
            # A corpse becomes food only after an actual attack defeat. No live
            # targets, remote corpse coordinates or prey names enter the policy.
            if result.get('target_defeated'):
                target=self._target(actor,result['target'])
                if target not in self.spiders and target not in self.food:
                    from spider_sim.feeding import food_value
                    self.food[target]={'wrap_turns':0,'remaining':food_value(target,self.bodies[target])}
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
    if world.feeding_cycle:
        from spider_sim.feeding import HUNGER_PER_ROUND
        for actor in world.hunger:
            world.hunger[actor]=min(1.,world.hunger[actor]+HUNGER_PER_ROUND)
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
                        **({"controller_kind":policy.controller_kind,"plan":policy.last_plan} if getattr(policy,"controller_kind",None) in ("djepa","pursuit") else {})})
    births=world.replace_dead_prey()
    for birth in births:
        newborn=birth['actor'];parent_policy=policies.get(birth['home'])
        if getattr(parent_policy,'controller_kind',None)=='djepa':
            from spider_sim.djepa_ecology import NeuralPolicy
            policies[newborn]=NeuralPolicy(parent_policy.model,parent_policy.species,parent_policy.selection)
        else:policies[newborn]=ForagerBaseline(world.seed+10000+int(world.contact_ids[newborn].rsplit('_',1)[1]))
        for row in records:
            if row['result'].get('target_defeated') and world._target(row['actor'],row['result'].get('target'))==birth['replaces']:
                row.setdefault('births',[]).append(birth)
                break
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
