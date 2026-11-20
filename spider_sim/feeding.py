"""Authored feeding goals; learned pursuit remains the hunting motor policy.

Prototype timing and nutrition are experiment parameters, not spider biology.
"""
import math

INITIAL_HUNGER = .8
HUNGER_PER_ROUND = .01
RESUME_HUNTING = .6
WRAP_TURNS = 3
FEED_PER_TURN = .1
INTERRUPTIBLE_GOALS = {'wrap','feed','approach_meal','seek_mate','approach_mate','court','defend'}


def food_value(actor, body):
    # Explicit experiment values: authored cricket geometry is actually larger
    # than rat geometry, so volume alone would contradict the intended food chain.
    if actor.startswith('cave_cricket_'):return .2
    if actor.startswith('cave_rat_'):return .5
    if actor.startswith('freight_shadowbeast'):return 2.5
    return max(.2,min(3.,body.width*body.depth*body.height/10))


class FeedingPolicy:
    neural = True
    controller_kind = 'pursuit'

    def __init__(self, hunter):
        self.hunter = hunter
        self.goal = 'hunt'
        self.last_plan = None
        self.seeking_mate = False
        self.meal = None
        self.search_turns=0
        self.defense_track=None
        self.defense_age=0
        self.mate_track=None

    def choose(self, obs):
        private=dict(obs,contacts=[c for c in obs['contacts'] if not c.get('conspecific')])
        proposal = self.hunter.choose(private)  # Keep recurrent memory current.
        hunger = obs['self']['hunger']
        threshold=getattr(self,'mate_hunger_threshold',.1)
        if (hunger < threshold if threshold>.001 else hunger<=threshold):
            self.seeking_mate = True
        if hunger >= getattr(self,'resume_hunting',.2):
            self.seeking_mate = False
        # Respond to an observed invitation even when not sated enough to
        # initiate a search. Keep a visible/receptive partner through approach.
        ready=(hunger<getattr(self,'mate_hunger_limit',.2) and obs['self']['hp_fraction']>=.35
               and getattr(self,'courtship_ready',True))
        partners={}
        for c in obs['contacts']:
            if c.get('conspecific') and c.get('mate_signal')=='receptive':
                if c['track'] not in partners or c['modality']=='vision':partners[c['track']]=c
        response=partners.get(self.mate_track) if ready else None
        if response is None and ready:
            invitations=[c for c in obs['contacts'] if c.get('conspecific')
                and c.get('mate_signal')=='receptive' and c.get('courtship_signal')]
            response=min(invitations,key=lambda c:c['range_ft'],default=None)
            if response:response=partners[response['track']]
        self.mate_track=response['track'] if response else None
        incoming=obs.get('incoming_damage',[])
        interrupted=self.goal in INTERRUPTIBLE_GOALS
        if incoming and interrupted:
            known=[event for event in incoming if event.get('attacker')]
            if known:
                self.defense_track=known[-1]['attacker'];self.defense_age=0
        feedback=obs.get('last_result') or {}
        if feedback.get('target_defeated') and feedback.get('target')==self.defense_track:
            self.defense_track=None
        attacker=next((c for c in obs['contacts'] if c['track']==self.defense_track and c['modality']=='vision'),None)
        if self.defense_track:
            self.defense_age=0 if attacker else self.defense_age+1
            if self.defense_age>4:self.defense_track=None
        # A direct attack interrupts feeding or mating before the critical-HP rule.
        # No hidden attacker position is followed. Preserve the unfinished meal.
        if self.defense_track or (incoming and interrupted):
            self.goal='defend'
            if attacker:
                if attacker['range_ft']<=6:
                    proposal={'action':'attack','target':attacker['track']}
                else:
                    x,y,_=attacker['relative_position_ft']
                    proposal={'action':'move','heading_deg':math.degrees(math.atan2(y,x)), 'distance_ft':min(10,max(1,math.hypot(x,y)-5)), 'keep_facing':True,'face_track':attacker['track']}
                    if self.hunter.locomotion:proposal,_=self.hunter.locomotion.pursue(obs,proposal)
            else:
                proposal={'action':'wait'}  # Alert, without inventing an attacker.
        elif obs['self']['hp_fraction'] < .35:
            self.goal = 'survive'
            visible = [c for c in obs['contacts'] if c['modality'] == 'vision']
            if visible:
                threat = min(visible, key=lambda c:c['range_ft'])
                proposal = {'action':'move', 'heading_deg':threat['bearing_deg']+180,
                            'distance_ft':10, 'keep_facing':True, 'face_track':threat['track']}
                if self.hunter.locomotion:
                    proposal, _ = self.hunter.locomotion.pursue(obs, proposal)
            else:
                proposal = {'action':'wait'}
        elif self.seeking_mate or response:
            self.goal = 'seek_mate'
            self.search_turns+=1
            candidates=list(partners.values()) if ready else []
            partner=response or min(candidates,key=lambda c:c['range_ft'],default=None)
            if partner:self.mate_track=partner['track']
            if partner and hunger<=getattr(self,'mate_hunger_limit',.2):
                if partner['modality']=='vision' and partner['range_ft']<=6:
                    self.goal='court';proposal={'action':'court','target':partner['track']}
                else:
                    vector=partner.get('relative_position_ft')
                    if vector:x,y,_=vector
                    else:
                        angle=math.radians(partner['bearing_deg']);x=partner['range_ft']*math.cos(angle);y=partner['range_ft']*math.sin(angle)
                    self.goal='approach_mate'
                    proposal={'action':'move','heading_deg':math.degrees(math.atan2(y,x)), 'distance_ft':min(5,max(.5,math.hypot(x,y)-4)), 'keep_facing':True,'face_track':partner['track']}
                    if self.hunter.locomotion:proposal,_=self.hunter.locomotion.pursue(obs,proposal)
            elif obs['self'].get('touching_web') and self.search_turns%5==1:
                proposal={'action':'vibrate_web'}
            else:
                proposal = self.hunter.locomotion.roam(obs)[0] if self.hunter.locomotion else {'action':'wait'}
        else:
            meals = obs.get('food_in_view', [])
            meal = next((c for c in meals if c['track'] == self.meal), None)
            if meal is None:
                meal = min(meals, key=lambda c:c['range_ft'], default=None)
            self.meal = meal['track'] if meal else None
            if meal:
                if meal['range_ft'] > 6:
                    self.goal = 'approach_meal'
                    x,y,_ = meal['relative_position_ft']
                    proposal = {'action':'move','heading_deg':math.degrees(math.atan2(y,x)),
                                'distance_ft':min(5,max(.5,math.hypot(x,y)-4)), 'keep_facing':True}
                    if self.hunter.locomotion:
                        proposal, _ = self.hunter.locomotion.pursue(obs,proposal)
                else:
                    self.goal = 'feed' if meal['wrapped'] else 'wrap'
                    proposal = {'action':self.goal,'target':meal['track']}
            else:
                self.goal = 'hunt'
        self.hunter.task.previous = proposal
        if self.goal != 'hunt':
            self.hunter.task.track = None
            self.hunter.task.last = None
        self.last_plan = dict(self.hunter.last_plan, executed_command=dict(proposal),
                              hunger=hunger, meal_track=self.meal, defense_track=self.defense_track, incoming_damage=incoming)
        if self.goal != 'hunt':
            self.last_plan.update(selection_reason='feeding_goal_'+self.goal, selected=None)
        return proposal
