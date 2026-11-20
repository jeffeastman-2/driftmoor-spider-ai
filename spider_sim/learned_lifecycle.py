"""Learned goal selection with authored, sensory-only motor execution.

No call to the teacher's choose(), no teacher goal latch or safety goal override.
Missing targets produce wait commands; engine feasibility remains authoritative.
"""
import math
import torch
from spider_sim.feeding import FeedingPolicy,INTERRUPTIBLE_GOALS
from spider_sim.lifecycle_model import LifecycleModel,FEATURE_NAMES,GOALS,features


def load(path):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if tuple(saved['features'])!=FEATURE_NAMES or tuple(saved['goals'])!=GOALS:
        raise ValueError('Lifecycle feature/goal schema mismatch')
    model=LifecycleModel(saved['recurrent']);model.load_state_dict(saved['state_dict']);return model.eval()


class LearnedLifecyclePolicy(FeedingPolicy):
    def __init__(self,hunter,model,sex):
        super().__init__(hunter);self.model=model;self.sex=sex;self.hidden=None
        self.proprioception=lambda:{'sex':self.sex,'mate_cooldown_turns':0}
        self.last_probabilities=None;self.unexecutable=0

    def spawn(self,hunter,sex):return type(self)(hunter,self.model,sex)

    @torch.no_grad()
    def choose(self,obs):
        private=dict(obs,self={**obs['self'],**self.proprioception()})
        logits,self.hidden=self.model.step(torch.tensor(features(private),dtype=torch.float32)[None,None],self.hidden)
        self.last_probabilities=logits[0,0].softmax(-1).tolist()
        selected=GOALS[int(logits[0,0].argmax())]
        command=self.execute(selected,obs)
        self.goal=selected
        self.hunter.task.previous=command
        if selected!='hunt':self.hunter.task.track=None;self.hunter.task.last=None
        self.last_plan=dict(self.hunter.last_plan,executed_command=dict(command),
            selection_reason='learned_lifecycle_goal',lifecycle_goal=selected,
            lifecycle_probabilities=dict(zip(GOALS,self.last_probabilities)))
        return command

    def execute(self,goal,obs):
        # Advance learned pursuit memory once, including during non-hunt goals.
        proposal=self.hunter.choose(dict(obs,contacts=[c for c in obs['contacts'] if not c.get('conspecific')]))
        incoming=obs.get('incoming_damage',[])
        if incoming and self.goal in INTERRUPTIBLE_GOALS:
            known=[e for e in incoming if e.get('attacker')]
            if known:self.defense_track=known[-1]['attacker'];self.defense_age=0
        feedback=obs.get('last_result') or {}
        if feedback.get('target_defeated') and feedback.get('target')==self.defense_track:self.defense_track=None
        attacker=next((c for c in obs['contacts'] if c['track']==self.defense_track and c['modality']=='vision'),None)
        if self.defense_track:
            self.defense_age=0 if attacker else self.defense_age+1
            if self.defense_age>4:self.defense_track=None
        contacts={c['track']:c for c in obs['contacts'] if c.get('conspecific') and c.get('mate_signal')=='receptive' and c['modality']!='vision'}
        contacts.update({c['track']:c for c in obs['contacts'] if c.get('conspecific') and c.get('mate_signal')=='receptive' and c['modality']=='vision'})
        partner=contacts.get(self.mate_track)
        if partner is None:
            invitations=[c for c in contacts.values() if c.get('courtship_signal')]
            partner=min(invitations or list(contacts.values()),key=lambda c:c['range_ft'],default=None)
        self.mate_track=partner['track'] if partner else None
        meals=obs.get('food_in_view',[])
        meal=next((m for m in meals if m['track']==self.meal),None) or min(meals,key=lambda c:c['range_ft'],default=None)
        def move(contact,speed,stop=4):
            vector=contact.get('relative_position_ft')
            if vector:x,y,*_=vector
            else:
                angle=math.radians(contact['bearing_deg']);x=contact['range_ft']*math.cos(angle);y=contact['range_ft']*math.sin(angle)
            cmd={'action':'move','heading_deg':math.degrees(math.atan2(y,x)),'distance_ft':min(speed,max(.5,math.hypot(x,y)-stop)),'keep_facing':True}
            if goal in ('approach_mate','defend'):cmd['face_track']=contact['track']
            if self.hunter.locomotion:cmd,_=self.hunter.locomotion.pursue(obs,cmd)
            return cmd
        if goal=='hunt':return proposal
        if goal=='defend':
            if attacker:
                return {'action':'attack','target':attacker['track']} if attacker['range_ft']<=6 else move(attacker,10,5)
            return {'action':'wait'}
        if goal=='survive':
            visible=[c for c in obs['contacts'] if c['modality']=='vision']
            if not visible:return {'action':'wait'}
            threat=min(visible,key=lambda c:c['range_ft'])
            cmd={'action':'move','heading_deg':threat['bearing_deg']+180,'distance_ft':10,'keep_facing':True,'face_track':threat['track']}
            return self.hunter.locomotion.pursue(obs,cmd)[0] if self.hunter.locomotion else cmd
        if goal in ('seek_mate','approach_mate','court'):
            self.search_turns+=1
            if goal=='seek_mate':
                if obs['self'].get('touching_web') and self.search_turns%5==1:return {'action':'vibrate_web'}
                return self.hunter.locomotion.roam(obs)[0] if self.hunter.locomotion else {'action':'wait'}
            if partner:
                return {'action':'court','target':partner['track']} if goal=='court' else move(partner,5)
        if goal in ('approach_meal','wrap','feed') and meal:
            self.meal=meal['track']
            return move(meal,5) if goal=='approach_meal' else {'action':goal,'target':meal['track']}
        self.unexecutable+=1
        return {'action':'wait'}
