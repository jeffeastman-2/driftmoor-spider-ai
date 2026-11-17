"""Paired engine trials of authored meal-defense choices, without UI pacing.

No training or claim of learned behavior. Private observations drive controllers;
world state is used only to construct scenarios and score actual outcomes.
"""
import argparse
import gzip
import hashlib
import itertools
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path

from spider_sim.ecology import EcologyEncounter,SPIDER,GoalBaseline,ecology_round
from spider_sim.feeding import FeedingPolicy
from spider_sim.pursuit import PursuitPolicy,load

BEAST='freight_shadowbeast'
MEAL='cave_rat_1'
MODES=('pursue','guard','resume')


class ComparisonPolicy(FeedingPolicy):
    def __init__(self,hunter,mode):
        super().__init__(hunter)
        if mode not in MODES:raise ValueError(mode)
        self.mode=mode;self.last_sighting=None;self.releases=0

    def choose(self,obs):
        previous=self.hunter.task.previous or {}
        feedback=obs.get('last_result') or {}
        current=next((c for c in obs['contacts'] if c['track']==self.defense_track and c['modality']=='vision'),None)
        release=None
        # Infer the opponent's radial movement after subtracting actual own travel.
        # Heading and hidden flee intent of the opponent are deliberately unavailable.
        if self.defense_track and not obs.get('incoming_damage'):
            if self.mode=='resume' and feedback.get('target')==self.defense_track and feedback.get('damage',0)>0:
                release='successful_retaliation'
            if self.mode=='guard' and current and self.last_sighting and self.last_sighting['track']==self.defense_track:
                old=self.last_sighting['relative_position_ft'];new=current['relative_position_ft']
                angle=math.radians(previous.get('heading_deg',0))
                moved=feedback.get('distance_ft',0) if previous.get('action')=='move' else 0
                dx=new[0]-old[0]+moved*math.cos(angle);dy=new[1]-old[1]+moved*math.sin(angle)
                outward=(dx*old[0]+dy*old[1])/max(.001,math.hypot(*old[:2]))
                if outward>.5 and current['range_ft']>6:release='observed_withdrawal'
        if release:
            self.defense_track=None;self.defense_age=0;self.releases+=1
        command=super().choose(obs)
        measured=next((c for c in obs['contacts'] if c['track']==self.defense_track and c['modality']=='vision'),None)
        self.last_sighting=dict(measured) if measured else None
        self.last_plan.update(experiment_mode=self.mode,defense_release=release)
        return command


class Attacker(GoalBaseline):
    def __init__(self,behavior,max_hp,return_turns=3):
        super().__init__();self.behavior=behavior;self.max_hp=max_hp
        self.flee_turns=0;self.last_hp=max_hp;self.returns=0;self.return_turns=return_turns

    def choose(self,obs):
        if self.behavior=='current':return super().choose(obs)
        hp=obs['self']['hp_fraction']*self.max_hp
        if hp<self.last_hp:
            self.flee_turns=9999 if self.behavior=='first_hit' else self.return_turns
        self.last_hp=hp
        visible=[c for c in obs['contacts'] if c['modality']=='vision']
        nearest=min(visible,key=lambda c:c['range_ft'],default=None)
        if self.flee_turns:
            self.flee_turns-=1;self.goal='survive'
            if self.flee_turns==0:self.returns+=1
            return {'action':'move','heading_deg':nearest['bearing_deg']+180,'distance_ft':10} if nearest else {'action':'wait'}
        # Returner intentionally re-engages even if wounded; this is an experimental
        # stress condition, not a silent modification of the live Shadowbeast.
        if nearest:
            self.goal='feed'
            if nearest['range_ft']<=6:return {'action':'attack','target':nearest['track']}
            x,y,_=nearest['relative_position_ft']
            return {'action':'move','heading_deg':math.degrees(math.atan2(y,x)),'distance_ft':min(10,max(1,math.hypot(x,y)-5))}
        self.goal='rest';return {'action':'wait'}


def scenarios(seeds):
    for seed,hunger,health,beast_hp,layout,behavior in itertools.product(seeds,(.4,.9),(.5,1.),(12,24),('stores','guardroom'),('current','first_hit','returner')):
        yield dict(seed=seed,hunger=hunger,health=health,beast_hp=beast_hp,layout=layout,behavior=behavior)


def setup(case,model,mode):
    world=EcologyEncounter(case['seed'],feeding_cycle=True)
    # Isolate one meal interruption without unrelated births, courtship, or prey
    # switching. The normal game engine still resolves sight, movement and dice.
    world.active={SPIDER,BEAST}
    for cid,session in world.config.characters.items():
        if cid not in world.active:session.character.current_hp=0
    if case['layout']=='stores':x,y,zone=100.,140.,'Fungal Stores'
    else:x,y,zone=25.,95.,'Freight Guardroom'
    world.positions[SPIDER]=(x,y,zone);world.positions[MEAL]=(x+4,y,zone)
    world.positions[BEAST]=(x-4,y,zone)
    world.config.characters[MEAL].character.current_hp=0
    world.food[MEAL]={'wrap_turns':0,'remaining':1.2}
    # A sufficiently large pre-existing meal lets us measure attaining fullness;
    # it is a standardized food portion, not a newly enlarged live rat.
    world.hunger[SPIDER]=case['hunger']
    world.config.characters[SPIDER].character.current_hp=round(world.max_hp[SPIDER]*case['health'])
    world.max_hp[BEAST]=case['beast_hp'];world.config.characters[BEAST].character.current_hp=case['beast_hp']
    hunter=PursuitPolicy(model,approach=True,facing='track',rest_without_goal=True,local_motion=True,seed=case['seed'])
    spider=ComparisonPolicy(hunter,mode);spider.goal='wrap';spider.meal=world.tokens[SPIDER][MEAL]
    # Begin after the first wrap action, so a first-round hit genuinely interrupts
    # an in-progress meal rather than requiring a fabricated damage observation.
    world.act(SPIDER,{'action':'wrap','target':spider.meal})
    beast=Attacker(case['behavior'],case['beast_hp'])
    return world,{SPIDER:spider,BEAST:beast}


def episode(case,model,mode,horizon,transitions=None,case_id=0):
    world,policies=setup(case,model,mode)
    totals=Counter();first_full=None;first_defense=None;last_defense=None
    interruptions=0;was_defending=False;recorded_hit_rounds=set();last_food_round=None
    for turn in range(horizon):
        rows=ecology_round(world,policies,turn)
        for row in rows:
            if row['actor']==BEAST and row['result']['damage']>0:
                totals['damage_taken']+=row['result']['damage'];recorded_hit_rounds.add(turn)
            if row['actor']!=SPIDER:continue
            goal=row['goal'];result=row['result'];command=row['command']
            if goal=='defend':
                totals['defense_turns']+=1
                if command['action']=='move':totals['chase_turns']+=1
                if not was_defending:interruptions+=1
                if first_defense is None:first_defense=turn
                last_defense=turn
            if result.get('consumed',0)>0:
                totals['food_consumed']+=result['consumed'];last_food_round=turn
            if command['action']=='attack' and not result['admitted']:totals['rejected_attacks']+=1
            if command['action']=='move' and result['distance_ft']<.2:totals['blocked_moves']+=1
            was_defending=goal=='defend'
            if transitions is not None:
                from spider_sim.djepa_ecology import senses
                transitions.write(json.dumps({'case':case_id,'mode':mode,'turn':turn,'observation':row['observation'],
                    'goal':goal,'command':command,'result':result,'next_observation':senses(world,SPIDER),
                    'timing':'next observation is after the whole engine round'})+'\n')
        if world.hunger[SPIDER]<=.001:
            first_full=turn+1;break
        if world.config.characters[SPIDER].character.current_hp<=0:break
    return dict(case_id=case_id,mode=mode,**case,rounds=turn+1,
                survived=world.config.characters[SPIDER].character.current_hp>0,
                final_hp=world.config.characters[SPIDER].character.current_hp,
                final_hunger=world.hunger[SPIDER],became_full=first_full is not None,
                turns_to_full=first_full,interruptions=interruptions,
                repeat_interruptions=max(0,interruptions-1),
                beast_returns=policies[BEAST].returns,beast_killed=world.config.characters[BEAST].character.current_hp<=0,
                last_food_round=last_food_round,defense_releases=policies[SPIDER].releases,
                **{key:totals[key] for key in ('damage_taken','food_consumed','defense_turns','chase_turns','rejected_attacks','blocked_moves')})


def aggregate(rows):
    n=len(rows);full=[r['turns_to_full'] for r in rows if r['became_full']]
    return dict(episodes=n,survival_rate=sum(r['survived'] for r in rows)/n,
                fullness_rate=sum(r['became_full'] for r in rows)/n,
                mean_turns_to_full_successes=statistics.mean(full) if full else None,
                **{'mean_'+key:statistics.mean(r[key] for r in rows) for key in ('rounds','damage_taken','food_consumed','defense_turns','chase_turns','repeat_interruptions','rejected_attacks','blocked_moves')})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,default=6);parser.add_argument('--horizon',type=int,default=72)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    model=load(args.model);model.eval()
    cases=list(scenarios(range(73000,73000+args.seeds)));rows=[];started=time.monotonic()
    with (args.output/'episodes.jsonl').open('w') as summary,gzip.open(args.output/'transitions.jsonl.gz','wt') as transitions:
        for i,case in enumerate(cases):
            for mode in MODES:
                row=episode(case,model,mode,args.horizon,transitions,i);rows.append(row)
                summary.write(json.dumps(row)+'\n');summary.flush()
            if (i+1)%24==0:print(f'{i+1}/{len(cases)} paired scenarios; {time.monotonic()-started:.1f}s',flush=True)
    report={'kind':'paired authored policy experiment; no training','cases':len(cases),'episodes':len(rows),'horizon':args.horizon,
            'elapsed_seconds':time.monotonic()-started,'seeds':list(range(73000,73000+args.seeds)),
            'model_sha256':hashlib.sha256(args.model.read_bytes()).hexdigest(),
            'overall':{mode:aggregate([r for r in rows if r['mode']==mode]) for mode in MODES},
            'by_attacker':{behavior:{mode:aggregate([r for r in rows if r['mode']==mode and r['behavior']==behavior]) for mode in MODES} for behavior in ('current','first_hit','returner')},
            'limits':['Isolated spider/attacker/standardized 1.2-unit corpse, not full ecology',
                      'Episode stops at fullness, death or horizon; damage exposure durations differ',
                      'Paired starting seeds; divergent actions consume different subsequent dice',
                      'Guard infers retreat from observed relative motion corrected for own travel',
                      'No learned policy or predictor has been trained from these data']}
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['overall'],indent=2),flush=True)

if __name__=='__main__':main()
