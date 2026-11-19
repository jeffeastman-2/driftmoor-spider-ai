"""Private-observation lifecycle demonstrations and contiguous-event coverage.

Actor IDs, audit fields and teacher state are metadata/labels, never model inputs.
No controller or world rule is changed by recording an episode.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import math

SCHEMA='lifecycle-experience-v1'
GOALS=('hunt','approach_meal','wrap','feed','seek_mate','approach_mate','court','defend','survive')
FOOD_GOALS={'approach_meal','wrap','feed'}


def private_observation(world,actor,observation):
    result=deepcopy(observation)
    own=result['self']
    # These are proprioception, not another creature's hidden world state.
    remaining=max(0,world.mate_cooldown.get(actor,0)-world.elapsed_seconds)
    own.update(sex=world.spiders[actor],mate_cooldown_turns=math.ceil(remaining/6))
    return result


class ExperienceRecorder:
    def __init__(self,world,policies):
        self.world=world;self.policies=policies;self.wrapped=set();self.captured={}
        self.previous_goal={};self.previous_contacts={};self.previous_food={}
        self.history_contacts=defaultdict(set)

    def prepare(self):
        self.captured={}
        for actor,policy in self.policies.items():
            if actor not in self.world.spiders or actor in self.wrapped:continue
            choose=policy.choose
            def capture(observation,actor=actor,choose=choose):
                self.captured[actor]=private_observation(self.world,actor,observation)
                return choose(observation)
            policy.choose=capture
            self.wrapped.add(actor)

    def decisions(self,seed,turn,records):
        world=self.world
        output=[]
        for row in records:
            actor=row['actor']
            if actor not in self.captured:continue
            obs=self.captured[actor]
            previous=self.previous_goal.get(actor)
            tracks={c['track'] for c in obs['contacts']}
            food={m['track'] for m in obs.get('food_in_view',[])}
            flags=situations(obs,previous)
            if self.previous_contacts.get(actor,set())-tracks:flags.add('previous_contact_missing')
            if (tracks-self.previous_contacts.get(actor,set())) & self.history_contacts[actor]:flags.add('contact_reappeared')
            if self.previous_food.get(actor,set())-food:flags.add('previous_meal_missing')
            self.history_contacts[actor].update(tracks)
            self.previous_contacts[actor]=tracks;self.previous_food[actor]=food
            self.previous_goal[actor]=row['goal']
            target=world._target(actor,row['command'].get('target'))
            kind=('rat' if target and target.startswith('cave_rat_') else
                  'cricket' if target and target.startswith('cave_cricket_') else
                  'shadowbeast' if target and target.startswith('freight_shadowbeast') else
                  'spider' if target in world.spiders else 'other' if target else None)
            output.append({'schema':SCHEMA,'seed':seed,'turn':turn,'actor':actor,
                'observation':obs,'teacher_goal':row['goal'],'command':deepcopy(row['command']),
                'result':deepcopy(row['result']),
                'after_round':{'hp_fraction':world.config.characters[actor].character.current_hp/world.max_hp[actor],
                    'hunger':world.hunger[actor],'alive':actor in world.active},
                'audit':{'previous_goal':previous,'target_kind':kind,'situations':sorted(flags)}})
        return output


def situations(obs,previous_goal=None):
    own=obs['self'];hunger=own['hunger'];hp=own['hp_fraction']
    flags=set()
    food=obs.get('food_in_view',[])
    receptive=[c for c in obs['contacts'] if c.get('conspecific') and c.get('mate_signal')=='receptive']
    damage=bool(obs.get('incoming_damage'))
    if hunger<.1:flags.add('hunger_below_10')
    elif hunger<.2:flags.add('hunger_10_to_20')
    elif hunger<.6:flags.add('hunger_20_to_60')
    elif hunger<1:flags.add('hunger_60_to_100')
    else:flags.add('maximum_hunger')
    if hp<.35:flags.add('critical_hp')
    if hp<.35 and hunger>=.6:flags.add('hungry_and_critical_hp')
    if food:flags.add('food_visible')
    if food and hunger>=.6:flags.add('hungry_with_food')
    if food and receptive:flags.add('food_and_receptive_partner')
    if receptive and hunger<.1:flags.add('sated_with_receptive_partner')
    if any(c.get('courtship_signal') and c['modality']=='web' for c in obs['contacts']):flags.add('web_courtship_signal')
    if damage:flags.add('incoming_damage')
    if damage and previous_goal in FOOD_GOALS:flags.add('feeding_under_attack')
    if previous_goal=='defend' and not damage:flags.add('defense_without_new_damage')
    if own.get('mate_cooldown_turns') in (1,2):flags.add('cooldown_near_expiry')
    for threshold,label in ((.1,'10'),(.2,'20'),(.6,'60')):
        if abs(hunger-threshold)<=.02+1e-9:flags.add('near_hunger_'+label)
    if abs(hp-.35)<=.05:flags.add('near_hp_35')
    return flags


class Coverage:
    def __init__(self):
        self.counts=Counter();self.episodes=Counter();self.actors=defaultdict(set)
        self.seeds=defaultdict(set);self.last={};self.defenses={};self.defense_outcomes=Counter()
        self.defense_examples=[]

    def add(self,row):
        seed,actor,turn=row['seed'],row['actor'],row['turn'];key=(seed,actor)
        flags=set(row['audit']['situations'])|{'goal:'+row['teacher_goal'],
            'action:'+row['command']['action'],'result:'+row['result']['status']}
        if row['audit']['target_kind']:flags.add('target:'+row['audit']['target_kind'])
        old_turn,old_flags=self.last.get(key,(-10,set()))
        for flag in flags:
            self.counts[flag]+=1;self.actors[flag].add(key);self.seeds[flag].add(seed)
            if turn!=old_turn+1 or flag not in old_flags:self.episodes[flag]+=1
        self.last[key]=(turn,flags)
        if 'feeding_under_attack' in flags and key not in self.defenses:
            own=row['observation']['self']
            self.defenses[key]={'seed':seed,'actor':actor,'started':turn,
                'previous_goal':row['audit']['previous_goal'],'response_goal':row['teacher_goal'],
                'hunger':own['hunger'],'hp_fraction':own['hp_fraction'],'sex':own['sex']}
        if key in self.defenses and row['teacher_goal'] in FOOD_GOALS:
            self.finish_defense(key,'returned_to_food',turn)
        if not row['after_round']['alive']:self.finish_defense(key,'died',turn)

    def finish_defense(self,key,outcome,turn):
        event=self.defenses.pop(key,None)
        if event is None:return
        self.defense_outcomes[outcome]+=1
        self.defense_examples.append({**event,'outcome':outcome,'ended':turn,'duration_turns':turn-event['started']})

    def end_actor(self,seed,actor,turn):self.finish_defense((seed,actor),'died',turn)

    def report(self):
        return {'counts':{flag:{'turns':n,'actor_episodes':self.episodes[flag],
            'actors':len(self.actors[flag]),'seeds':sorted(self.seeds[flag])}
            for flag,n in sorted(self.counts.items())},
            'defense_outcomes':dict(self.defense_outcomes),
            'defense_still_open':len(self.defenses),'defense_examples':self.defense_examples,
            'episode_definition':'Contiguous per-actor situation or goal segments, not independent samples.'}
