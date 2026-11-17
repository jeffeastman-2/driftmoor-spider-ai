"""Authored feeding goals; learned pursuit remains the hunting motor policy.

Prototype timing and nutrition are experiment parameters, not spider biology.
"""
import math

INITIAL_HUNGER = .8
HUNGER_PER_ROUND = .001
RESUME_HUNTING = .6
WRAP_TURNS = 3
FEED_PER_TURN = .1


def food_value(actor, body):
    # Explicit experiment values: authored cricket geometry is actually larger
    # than rat geometry, so volume alone would contradict the intended food chain.
    if actor.startswith('cave_cricket_'):return .2
    if actor.startswith('cave_rat_'):return .5
    if actor=='freight_shadowbeast':return 2.5
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

    def choose(self, obs):
        private=dict(obs,contacts=[c for c in obs['contacts'] if not c.get('conspecific')])
        proposal = self.hunter.choose(private)  # Keep recurrent memory current.
        hunger = obs['self']['hunger']
        if hunger <= .001:
            self.seeking_mate = True
        if hunger >= RESUME_HUNTING:
            self.seeking_mate = False
        # Abandon the meal when injured: existing pursuit is not a defense policy.
        if obs['self']['hp_fraction'] < .35:
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
        elif self.seeking_mate:
            self.goal = 'seek_mate'
            self.search_turns+=1
            partners=[c for c in obs['contacts'] if c.get('conspecific') and c.get('mate_signal')=='receptive']
            partner=min(partners,key=lambda c:c['range_ft'],default=None)
            if partner and hunger<=.2:
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
                              hunger=hunger, meal_track=self.meal)
        if self.goal != 'hunt':
            self.last_plan.update(selection_reason='feeding_goal_'+self.goal, selected=None)
        return proposal
