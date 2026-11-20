"""Offline learned lifecycle intentions from private sensory observations.

This module does not deploy a controller or replace engine/lifecycle rules.
Global IDs, teacher flags, audit fields and future outcomes are never features.
"""
import math
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

GOALS=('hunt','approach_meal','wrap','feed','seek_mate','approach_mate','court','defend','survive')
ACTIONS=('wait','move','attack','wrap','feed','court','vibrate_web','repair_web','damage_web')


def _bounded(value,scale=1):return max(-2.,min(2.,float(value or 0)/scale))


def _contacts(obs):
    # Prefer visual measurements when the same private track has both modalities.
    result={c['track']:c for c in obs.get('contacts',[]) if c['modality']!='vision'}
    result.update({c['track']:c for c in obs.get('contacts',[]) if c['modality']=='vision'})
    return list(result.values())


def _contact(c,heading):
    if c is None:return [0.]*10
    bearing=c.get('bearing_deg');angle=math.radians(bearing-heading) if bearing is not None else None
    dims=c.get('body_dimensions_ft',[0.,0.,0.])
    return [1.,_bounded(c.get('range_ft'),60),math.sin(angle) if angle is not None else 0.,
        math.cos(angle) if angle is not None else 0.,float(c['modality']=='vision'),float(c['modality']=='web'),
        *[_bounded(v,10) for v in dims],float(angle is not None)]


def features(obs,*,names=False):
    own=obs.get('self',{});last=obs.get('last_result') or {};values=[];labels=[]
    def add(name,value):labels.append(name);values.append(float(value or 0))
    for key in ('hunger','hp_fraction'):add(key,_bounded(own.get(key)))
    add('male',own.get('sex')=='male');add('female',own.get('sex')=='female')
    add('own_cooldown',_bounded(own.get('mate_cooldown_turns'),20));add('touching_web',own.get('touching_web',False))
    for key,scale in (('speed_ft',30),('reach_ft',10),('height_above_floor_ft',10)):add(key,_bounded(own.get(key),scale))
    for i,v in enumerate(own.get('body_dimensions_ft',[0.,0.,0.])):add(f'own_body_{i}',_bounded(v,10))
    for action in ACTIONS:add('last_action_'+action,last.get('action')==action)
    maxhp=max(1,own.get('max_hp',26))
    for key,scale in (('damage',maxhp),('distance_ft',10),('wrap_turns',3),('courtship_turns',3),('consumed',.1)):
        add('last_'+key,_bounded(last.get(key),scale))
    add('last_admitted',last.get('admitted',False));add('last_hit',last.get('hit',False));add('last_blocked',last.get('status')=='blocked')
    incoming=obs.get('incoming_damage',[])
    add('incoming_damage',_bounded(sum(d.get('damage_hp',0) for d in incoming),maxhp))
    add('known_attacker',any(d.get('attacker') for d in incoming))
    for i in range(8):add(f'clearance_{i}',_bounded(own.get('local_clearance_ft',[0.]*8)[i],10))
    contacts=_contacts(obs);food=obs.get('food_in_view',[])
    prey=sorted((c for c in contacts if not c.get('conspecific')),key=lambda c:c['range_ft'])
    mates=sorted((c for c in contacts if c.get('conspecific')),key=lambda c:(c.get('mate_signal')!='receptive',c['range_ft']))
    for key,value,scale in [('contact_count',len(contacts),10),('food_count',len(food),5),('conspecific_count',len(mates),5),('receptive_count',sum(c.get('mate_signal')=='receptive' for c in mates),5)]:add(key,_bounded(value,scale))
    heading=own.get('heading_deg',0)
    for kind,items in [('contact',prey),('mate',mates)]:
        for slot in range(2):
            c=items[slot] if slot<len(items) else None
            for i,value in enumerate(_contact(c,heading)):add(f'{kind}_{slot}_{i}',value)
            if kind=='mate':
                add(f'mate_{slot}_receptive',bool(c and c.get('mate_signal')=='receptive'))
                add(f'mate_{slot}_signal',bool(c and c.get('courtship_signal')))
    for slot in range(2):
        meals=sorted(food,key=lambda m:m['range_ft']);meal=meals[slot] if slot<len(meals) else None
        if meal:
            x,y,*_=meal['relative_position_ft'];angle=math.atan2(y,x)-math.radians(heading)
            vals=[1.,_bounded(meal['range_ft'],60),math.sin(angle),math.cos(angle),float(meal['wrapped']),_bounded(meal['remaining'],3)]
        else:vals=[0.]*6
        for i,value in enumerate(vals):add(f'meal_{slot}_{i}',value)
    return labels if names else values


FEATURE_NAMES=tuple(features({},names=True))


def action_features(obs,command):
    heading=obs['self'].get('heading_deg',0)
    angle=math.radians(command.get('heading_deg',heading)-heading)
    target=next((c for c in _contacts(obs) if c['track']==command.get('target',command.get('face_track'))),None)
    meal=next((m for m in obs.get('food_in_view',[]) if m['track']==command.get('target',command.get('face_track'))),None)
    if target is None and meal:
        x,y,*_=meal['relative_position_ft']
        target={'range_ft':meal['range_ft'],'bearing_deg':math.degrees(math.atan2(y,x)),'modality':'vision'}
    return [float(command['action']==a) for a in ACTIONS]+[_bounded(command.get('distance_ft'),10),math.sin(angle),math.cos(angle)]+_contact(target,heading)+[float(bool(meal and meal['wrapped'])),_bounded(meal['remaining'],3) if meal else 0.]


ACTION_SIZE=len(ACTIONS)+3+10+2


class LifecycleModel(nn.Module):
    def __init__(self,recurrent=True,hidden=64):
        super().__init__();self.recurrent=recurrent;self.hidden=hidden
        self.encoder=nn.Sequential(nn.Linear(len(FEATURE_NAMES),128),nn.ReLU(),nn.Linear(128,hidden),nn.ReLU())
        self.memory=nn.GRU(hidden,hidden,batch_first=True) if recurrent else None
        self.actor=nn.Sequential(nn.Linear(hidden,64),nn.ReLU(),nn.Linear(64,len(GOALS)))

    def represent(self,history,lengths):
        embedded=self.encoder(history)
        if not self.recurrent:return embedded
        packed=pack_padded_sequence(embedded,lengths.cpu(),batch_first=True,enforce_sorted=False)
        states,_=self.memory(packed)
        return pad_packed_sequence(states,batch_first=True,total_length=history.shape[1])[0]

    def forward(self,history,lengths):return self.actor(self.represent(history,lengths))

    def step(self,observation,hidden=None):
        embedded=self.encoder(observation)
        if self.recurrent:
            state,hidden=self.memory(embedded,hidden)
        else:state=embedded
        return self.actor(state),hidden


class LatentPredictor(nn.Module):
    """One-step predictor, trained after freezing perception, memory and actor."""
    def __init__(self,hidden=64,action_conditioned=True):
        super().__init__();self.action_conditioned=action_conditioned
        self.network=nn.Sequential(nn.Linear(hidden+(ACTION_SIZE if action_conditioned else 0),128),nn.ReLU(),nn.Linear(128,hidden+2))

    def forward(self,state,action):
        return self.network(torch.cat((state,action),-1) if self.action_conditioned else state)
