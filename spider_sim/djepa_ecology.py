"""Shared species architecture for private multi-contact predictive control.

JEPA-inspired, not a joint-embedding JEPA objective: forecasts are explicit
engine outcomes. Goal configuration and costs remain authored and inspectable.
"""
from spider_sim.runtime import game_root
game_root()
import math
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

SCHEMA="ecology-djepa-v1"
SPECIES=("rat","spider","shadowbeast")
SELF_SIZE=24
CONTACT_SIZE=12
CANDIDATE_SIZE=18
OUTCOMES=("injury_fraction","forward_displacement_30ft","left_displacement_30ft",
          "admission_logit","conditional_damage_20hp","contact_logit","range_30ft")


def species_for(actor):
    if actor.startswith("cave_rat_"):return "rat"
    if actor=="freight_spider":return "spider"
    if actor=="freight_shadowbeast":return "shadowbeast"
    return None


def senses(world,actor):
    """Engine-side sensor adapter: ten-foot local floor clearance, not a map.

    Synthetic ideal local surface measurements use the existing collision model.
    They do not expose routes, distant terrain, hidden creatures or future costs.
    """
    from engine.geometry import _check_impassable
    from engine.session import _movement_budget_feet
    from engine.loadout import wielded_weapons
    from models.space_time import Position
    obs=world.observation_for(actor)
    obs["schema"]=SCHEMA
    own=obs["self"];character=world.config.characters[actor].character
    own["max_hp"]=world.max_hp[actor]
    own["speed_ft"]=_movement_budget_feet(character.speed,in_combat=True)
    own["reach_ft"]=max((w.reach for w in wielded_weapons(world.config.characters[actor]) if not w.is_ranged),default=0)
    own["height_above_floor_ft"]=world.heights[actor]
    origin=world.positions[actor];heading=own["heading_deg"]
    clearance=[]
    for offset in range(0,360,45):
        angle=math.radians(heading+offset)
        end=(origin[0]+10*math.cos(angle),origin[1]+10*math.sin(angle),"")
        allowed,_,stop=_check_impassable(actor,origin,end,world.geometry.impassable,world.geometry.zones)
        clearance.append(10. if allowed else math.dist(origin[:2],stop) if stop else 0.)
    own["local_clearance_ft"]=clearance
    return obs


def contacts(obs):
    values={c["track"]:c for c in obs["contacts"] if c["modality"]!="vision"}
    values.update({c["track"]:c for c in obs["contacts"] if c["modality"]=="vision"})
    return values


def goal_for(obs,species):
    if obs["self"]["hp_fraction"]<.4:return "survive" if obs["contacts"] else "rest"
    if species=="rat":return "forage"
    return "feed" if obs["contacts"] else "explore"


def contact_values(contact,obs):
    if contact is None:return [0.]*CONTACT_SIZE
    heading=math.radians(obs["self"]["heading_deg"])
    bearing=math.radians((contact["bearing_deg"] if contact.get("bearing_deg") is not None else obs["self"]["heading_deg"])-obs["self"]["heading_deg"])
    vector=contact.get("relative_position_ft")
    if vector:
        x,y,z=vector
        forward=x*math.cos(heading)+y*math.sin(heading)
        left=-x*math.sin(heading)+y*math.cos(heading)
    else:forward=left=z=0.
    return [float(contact["modality"]==kind) for kind in ("vision","web","nonvisual")]+[
        forward/30,left/30,z/10,*[v/10 for v in contact.get("body_dimensions_ft",[0,0,0])],
        contact["range_ft"]/30,math.cos(bearing),math.sin(bearing)]


def encode(obs,species):
    if obs.get("schema")!=SCHEMA:raise ValueError("dJEPA ecology observation schema mismatch")
    own=obs["self"];last=obs.get("last_result") or {};goal=goal_for(obs,species)
    values=[own["hp_fraction"],*[v/10 for v in own["body_dimensions_ft"]],own["speed_ft"]/30,own["reach_ft"]/10,
            *[float(last.get("action")==a) for a in ("wait","move","attack")],
            last.get("distance_ft",0)/30,float(last.get("admitted",False)),last.get("damage",0)/20,
            *[v/10 for v in own["local_clearance_ft"]],
            *[float(goal==g) for g in ("feed","survive","forage","explore")]]
    assert len(values)==SELF_SIZE
    measured=list(contacts(obs).values())
    rows=[contact_values(c,obs) for c in measured]
    return values,rows


def candidates(obs):
    result=[{"action":"wait"}]
    heading=obs["self"]["heading_deg"]
    for offset in range(0,360,45):
        for distance in (5.,10.):
            result.append({"action":"move","heading_deg":(heading+offset)%360,"distance_ft":distance})
    result.extend({"action":"attack","target":c["track"]} for c in contacts(obs).values() if c["modality"]=="vision")
    return result


def candidate_values(command,obs,candidate_size=CANDIDATE_SIZE):
    angle=math.radians(command.get("heading_deg",obs["self"]["heading_deg"])-obs["self"]["heading_deg"])
    target=contacts(obs).get(command.get("target"))
    values=[float(command["action"]==a) for a in ("wait","move","attack")]+[
        math.cos(angle),math.sin(angle),command.get("distance_ft",0)/30]+contact_values(target,obs)
    if candidate_size==CANDIDATE_SIZE+1:
        offset=round(math.degrees(angle)/45)%8
        values.append(obs['self']['local_clearance_ft'][offset]/10 if command['action']=='move' else 0.)
    elif candidate_size!=CANDIDATE_SIZE:raise ValueError('Unknown candidate feature contract')
    return values


class EcologyModel(nn.Module):
    def __init__(self,candidate_size=CANDIDATE_SIZE):
        super().__init__()
        self.candidate_size=candidate_size
        self.contact_encoder=nn.Sequential(nn.Linear(CONTACT_SIZE,24),nn.ReLU(),nn.Linear(24,24),nn.ReLU())
        self.encoder=nn.Sequential(nn.Linear(SELF_SIZE+48,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU())
        self.memory=nn.GRU(32,48,batch_first=True)
        self.predictor=nn.Sequential(nn.Linear(48+candidate_size,64),nn.ReLU(),nn.Linear(64,len(OUTCOMES)))
        self.proposal=nn.Sequential(nn.Linear(48+candidate_size,32),nn.ReLU(),nn.Linear(32,1))

    def represent(self,own,seen,mask,hidden=None):
        embedded=self.contact_encoder(seen)
        count=mask.sum(-1,keepdim=True).clamp(min=1)
        mean=(embedded*mask.unsqueeze(-1)).sum(-2)/count
        maximum=embedded.masked_fill(~mask.bool().unsqueeze(-1),-1e9).max(-2).values
        maximum=torch.where(mask.sum(-1,keepdim=True)>0,maximum,torch.zeros_like(maximum))
        encoded=self.encoder(torch.cat((own,mean,maximum),dim=-1))
        state,hidden=self.memory(encoded,hidden)
        return state,hidden

    def outcomes(self,state,choices):
        raw=self.predictor(torch.cat((state,choices),dim=-1))
        return torch.cat((raw[...,:1].sigmoid(),raw[...,1:3].tanh(),raw[...,3:4],
                          F.softplus(raw[...,4:5]),raw[...,5:6],F.softplus(raw[...,6:7])),dim=-1)


def observation_tensors(observations,species):
    pairs=[encode(obs,species) for obs in observations]
    count=max(1,max(len(seen) for _,seen in pairs))
    own=torch.tensor([p[0] for p in pairs],dtype=torch.float32)[None]
    seen=torch.zeros(1,len(pairs),count,CONTACT_SIZE);mask=torch.zeros(1,len(pairs),count)
    for t,(_,rows) in enumerate(pairs):
        if rows:
            seen[0,t,:len(rows)]=torch.tensor(rows);mask[0,t,:len(rows)]=1
    return own,seen,mask


def costs(obs,species,commands,predictions):
    """Authored goal preferences in HP-equivalent units, not a learned critic.

    Permission is combined with damage conditional on permission, including misses.
    No separate impossible-attack penalty and no authoritative feasibility mask.
    """
    injury=predictions[:,0]*obs["self"]["max_hp"]
    expected_damage=predictions[:,3].sigmoid()*predictions[:,4]*20
    low=1-obs["self"]["hp_fraction"]
    score=injury*(4+4*low)
    if species!="rat":
        for i,command in enumerate(commands):
            if command["action"]=="attack":score[i]-=expected_damage[i]*(1-low)
    return score


class NeuralPolicy:
    neural=True
    controller_kind="djepa"
    def __init__(self,model,species,selection="predictive"):
        if selection not in ("predictive","proposal"):raise ValueError("Unknown neural selection mode")
        self.model=model;self.species=species;self.selection=selection;self.hidden=None;self.goal="rest";self.last_plan=None

    @torch.no_grad()
    def choose(self,obs):
        own,seen,mask=observation_tensors([obs],self.species)
        state,self.hidden=self.model.represent(own,seen,mask,self.hidden)
        choices=candidates(obs);encoded=torch.tensor([candidate_values(c,obs,self.model.candidate_size) for c in choices])
        latent=state[0,0].expand(len(choices),-1)
        prediction=self.model.outcomes(latent,encoded)
        prior=self.model.proposal(torch.cat((latent,encoded),dim=-1)).squeeze(-1)
        # Proposal is learned from the cautious teacher; predictions can override it.
        score=costs(obs,self.species,choices,prediction)-.25*F.log_softmax(prior,dim=0)
        if self.selection=="proposal":score=-F.log_softmax(prior,dim=0)
        selected=int(score.argmin());self.goal=goal_for(obs,self.species)
        self.last_plan={"selection":self.selection,"selected":selected,"candidates":choices,"scores":score.tolist(),
                        "predicted_injury_hp":float(prediction[selected,0]*obs["self"]["max_hp"]),
                        "predicted_admission":float(prediction[selected,3].sigmoid()),
                        "predicted_damage_hp":float(prediction[selected,3].sigmoid()*prediction[selected,4]*20),
                        "predictions":prediction.tolist()}
        return choices[selected]


def save(model,path,species):
    torch.save({"schema":SCHEMA,"species":species,"self_size":SELF_SIZE,"contact_size":CONTACT_SIZE,
                "candidate_size":model.candidate_size,"outcomes":list(OUTCOMES),"state":model.state_dict()},path)


def load(path,species):
    payload=torch.load(path,map_location="cpu",weights_only=True)
    if (payload.get("schema")!=SCHEMA or payload.get("species")!=species or
        payload.get("outcomes")!=list(OUTCOMES) or payload.get("candidate_size") not in (CANDIDATE_SIZE,CANDIDATE_SIZE+1) or
        payload.get("self_size")!=SELF_SIZE or payload.get("contact_size")!=CONTACT_SIZE):
        raise ValueError("Species/checkpoint/schema mismatch")
    model=EcologyModel(payload["candidate_size"]);model.load_state_dict(payload["state"])
    return model.eval()


def policies(world,directory,seed=0,selection="predictive"):
    from spider_sim.ecology import baseline_policies
    result=baseline_policies(world,seed)
    models={species:load(Path(directory)/(species+".pt"),species) for species in SPECIES}
    for actor in result:
        species=species_for(actor)
        if species:result[actor]=NeuralPolicy(models[species],species,selection)
    return result


class CautiousTeacher:
    """Private-sensory initialization policy; not a neural controller."""
    def __init__(self,species):self.species=species;self.goal="rest";self.turn=0
    def choose(self,obs):
        self.turn+=1;self.goal=goal_for(obs,self.species)
        choices=candidates(obs);own_volume=math.prod(obs["self"]["body_dimensions_ft"])
        nearby=list(contacts(obs).values())
        threats=[c for c in nearby if c["modality"]=="vision" and math.prod(c["body_dimensions_ft"])>own_volume*(1.5 if self.species=="rat" else .4)]
        prey=[c for c in nearby if c["modality"]=="vision" and math.prod(c["body_dimensions_ft"])<own_volume*.4]
        if self.species=="rat":prey=[]
        threat=min(threats,key=lambda c:c["range_ft"],default=None)
        target=min(prey,key=lambda c:c["range_ft"],default=None)
        retreat=threat is not None and (self.species=="rat" or obs["self"]["hp_fraction"]<.75 or threat["range_ft"]<10)
        if retreat:
            desired=(threat.get("bearing_deg") or 0)+180
        elif target:
            desired=math.degrees(math.atan2(target["relative_position_ft"][1],target["relative_position_ft"][0]))
            if target["range_ft"]<=5:
                return next(c for c in choices if c["action"]=="attack" and c["target"]==target["track"])
        else:desired=(obs["self"]["heading_deg"]+45*(self.turn//3))%360
        moves=[c for c in choices if c["action"]=="move"]
        def score(c):
            delta=abs((c["heading_deg"]-desired+180)%360-180)/180
            offset=round((c["heading_deg"]-obs["self"]["heading_deg"])/45)%8
            clearance=obs["self"]["local_clearance_ft"][offset]
            blocked=max(0,c["distance_ft"]-clearance)
            return delta+blocked+(.05 if c["distance_ft"]==5 else 0)
        best=min(moves,key=score)
        return best if min(obs["self"]["local_clearance_ft"])<10 or target or retreat or self.turn%4 else choices[0]
