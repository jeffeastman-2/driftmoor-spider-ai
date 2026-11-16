"""One-goal private-sensory pursuit controller and engine-supervised action values.

Authored target commitment is separate from learned motor choice. No global
position, opponent HP, RNG state or native reach check enters inference.
"""
import math
import torch
from torch import nn
from spider_sim.djepa_ecology import contacts,encode,candidates,candidate_values

SCHEMA='single-prey-pursuit-v1'
INPUT_SIZE=36
ACTION_SIZE=23


class PursuitGoal:
    def __init__(self):self.track=None;self.last=None;self.age=0;self.previous=None
    def observe(self,obs):
        observed=contacts(obs);feedback=obs.get('last_result') or {}
        if feedback.get('target_defeated') and feedback.get('target')==self.track:
            self.track=None;self.last=None
        if self.track is None:
            visible=[c for c in observed.values() if c['modality']=='vision']
            if visible:self.track=min(visible,key=lambda c:c['range_ft'])['track']
        current=observed.get(self.track)
        if current and current['modality']=='vision':self.last=dict(current);self.age=0
        elif self.last:
            self.age+=1
            # Dead reckoning from actual motor feedback, not hidden world state.
            if self.previous and self.previous['action']=='move':
                angle=math.radians(self.previous['heading_deg']);distance=feedback.get('distance_ft',0)
                x,y,z=self.last['relative_position_ft']
                self.last=dict(self.last,relative_position_ft=[x-distance*math.cos(angle),y-distance*math.sin(angle),z])
                self.last['range_ft']=math.sqrt(sum(v*v for v in self.last['relative_position_ft']))
                self.last['bearing_deg']=math.degrees(math.atan2(self.last['relative_position_ft'][1],self.last['relative_position_ft'][0]))
            if self.age>4:self.track=None;self.last=None
        return self.last


def features(obs,goal):
    own,_=encode(obs,'spider')
    if goal is None:return own+[0.]*12
    # Encode the selected contact, preserving only the spider's own measurements.
    from spider_sim.djepa_ecology import contact_values
    return own+contact_values(goal,obs)


def actions(obs,goal):
    # Same discrete motor inventory as the previous neural controller; no mask.
    choices=candidates(obs)
    if goal:choices=[c for c in choices if c['action']!='attack' or c['target']==goal['track']]
    return choices


def action_features(command,obs,goal,size=ACTION_SIZE):
    values=candidate_values(command,obs,19)
    vector=goal.get('relative_position_ft') if goal else None
    if vector:
        direction=math.atan2(vector[1],vector[0]);heading=math.radians(command.get('heading_deg',obs['self']['heading_deg']))
        values.extend((math.cos(heading-direction),math.sin(heading-direction)))
    else:values.extend((0.,0.))
    if size==23:
        # Derived local kinematics, not an engine rollout or a reach mask.
        # The net needn't rediscover Euclidean distance from rotated x/y inputs.
        x,y=(vector[:2] if vector else (0.,0.))
        distance=command.get('distance_ft',0) if command['action']=='move' else 0.
        if distance:
            offset=round((command['heading_deg']-obs['self']['heading_deg'])/45)%8
            distance=min(distance,obs['self']['local_clearance_ft'][offset])
        angle=math.radians(command.get('heading_deg',0))
        values.extend((math.hypot(x,y)/10,math.hypot(x-distance*math.cos(angle),y-distance*math.sin(angle))/10))
    elif size!=21:raise ValueError('Unknown pursuit action contract')
    return values


class PursuitModel(nn.Module):
    def __init__(self,action_size=ACTION_SIZE):
        super().__init__()
        self.action_size=action_size
        self.encoder=nn.Sequential(nn.Linear(INPUT_SIZE,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU())
        self.memory=nn.GRU(32,48,batch_first=True)
        self.value=nn.Sequential(nn.Linear(48+action_size,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    def represent(self,observations,hidden=None):return self.memory(self.encoder(observations),hidden)
    def values(self,state,choices):return self.value(torch.cat((state,choices),dim=-1)).squeeze(-1)


class PursuitPolicy:
    neural=True
    controller_kind='pursuit'
    def __init__(self,model):self.model=model;self.hidden=None;self.task=PursuitGoal();self.goal='pursue';self.last_plan=None
    @torch.no_grad()
    def choose(self,obs):
        goal=self.task.observe(obs);choices=actions(obs,goal)
        own=torch.tensor(features(obs,goal),dtype=torch.float32)[None,None]
        state,self.hidden=self.model.represent(own,self.hidden)
        encoded=torch.tensor([action_features(c,obs,goal,self.model.action_size) for c in choices],dtype=torch.float32)
        values=self.model.values(state[0,0].expand(len(choices),-1),encoded)
        selected=int(values.argmin());command=choices[selected];self.task.previous=command
        self.last_plan={'goal_track':self.task.track,'goal_age':self.task.age,'selected':selected,
                        'candidates':choices,'predicted_goal_costs':values.tolist()}
        return command


def save(model,path):torch.save({'schema':SCHEMA,'inputs':INPUT_SIZE,'actions':model.action_size,'state':model.state_dict()},path)
def load(path):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if (payload.get('schema'),payload.get('inputs'))!=(SCHEMA,INPUT_SIZE) or payload.get('actions') not in (21,23):
        raise ValueError('Pursuit checkpoint contract mismatch')
    model=PursuitModel(payload['actions']);model.load_state_dict(payload['state']);return model.eval()
