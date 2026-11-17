"""Counterfactual short-horizon meal-defense outcomes from cloned engine states.

Copies and outcome labels are researcher-side only. No opponent HP, behavior,
world coordinate, RNG state or scenario id enters predictor inputs.
"""
import argparse
import copy
import gzip
import hashlib
import json
import math
import time
from pathlib import Path
import torch
from experiments.compare_meal_defense import setup,scenarios,BEAST,MODES
from spider_sim.ecology import SPIDER
from spider_sim.djepa_ecology import senses
from spider_sim.feeding import HUNGER_PER_ROUND
from spider_sim.pursuit import load

HORIZONS=(3,6,12)
CONTEXT_NAMES=('hunger','hp_fraction','incoming_damage_fraction','meal_visible','meal_range_30ft','food_units_3','wrapped','previously_handling_food')
OUTCOME_NAMES=('damage_fraction','hunger_reduction','damaging_attack','death')


def private_context(obs,meal_track,previous_goal):
    meal=next((c for c in obs.get('food_in_view',[]) if c['track']==meal_track),None)
    return [obs['self']['hunger'],obs['self']['hp_fraction'],
            sum(e['damage_hp'] for e in obs.get('incoming_damage',[]))/obs['self']['max_hp'],
            float(meal is not None),meal['range_ft']/30 if meal else 0.,
            meal['remaining']/3 if meal else 0.,float(meal['wrapped']) if meal else 0.,
            float(previous_goal in ('wrap','feed','approach_meal'))]


def begin_round(world,policies):
    world.hunger[SPIDER]=min(1.,world.hunger[SPIDER]+HUNGER_PER_ROUND)
    if BEAST in world.active:
        obs=world.observation_for(BEAST)
        world.act(BEAST,policies[BEAST].choose(obs))
    return senses(world,SPIDER) if SPIDER in world.active else None


def fork_outcome(world,spider,beast,mode,draw_seed):
    # Policy carries the fixed neural module; share that immutable module rather
    # than duplicating its tensors for every branch. Hidden states remain copied.
    w=copy.deepcopy(world)
    controller=copy.deepcopy(spider,{id(spider.hunter.model):spider.hunter.model})
    controller.mode=mode
    opponent=copy.deepcopy(beast)
    policies={SPIDER:controller,BEAST:opponent}
    w.random.seed(draw_seed)
    initial_hp=w.config.characters[SPIDER].character.current_hp
    initial_hunger=w.hunger[SPIDER];maximum=w.max_hp[SPIDER]
    cumulative_damage=0.;outcomes=[]
    for step in range(1,max(HORIZONS)+1):
        # The first decision is already after the opponent's current-round action.
        # Subsequent steps include a new opponent response before our decision.
        obs=senses(w,SPIDER) if step==1 else begin_round(w,policies)
        if SPIDER in w.active:
            w.act(SPIDER,controller.choose(obs))
        w.elapsed_seconds+=6
        hp=w.config.characters[SPIDER].character.current_hp
        cumulative_damage=max(cumulative_damage,initial_hp-hp)
        if step in HORIZONS:
            outcomes.append([cumulative_damage/maximum,initial_hunger-w.hunger[SPIDER],float(cumulative_damage>0),float(hp<=0)])
        # Death is absorbing; no hunger increase on a corpse is a label of interest.
        if hp<=0:
            for horizon in HORIZONS:
                if horizon>step:outcomes.append([cumulative_damage/maximum,initial_hunger-w.hunger[SPIDER],1.,1.])
            break
    return outcomes


def collect(case,model,split,draws,return_delay):
    world,policies=setup(case,model,'guard')
    spider=policies[SPIDER];beast=policies[BEAST];beast.return_turns=return_delay
    origins=[];seen_retreat=False;seen_return=False;first_hit=False;previous_contact=None
    for turn in range(48):
        obs=begin_round(world,policies)
        if obs is None:break
        prior_command=spider.hunter.task.previous or {};feedback=obs.get('last_result') or {}
        contact=next((c for c in obs['contacts'] if c['modality']=='vision' and c['track']==world.tokens[SPIDER][BEAST]),None)
        retreat=False
        if contact and previous_contact:
            old=previous_contact['relative_position_ft'];new=contact['relative_position_ft']
            moved=feedback.get('distance_ft',0) if prior_command.get('action')=='move' else 0.
            angle=math.radians(prior_command.get('heading_deg',0))
            dx=new[0]-old[0]+moved*math.cos(angle);dy=new[1]-old[1]+moved*math.sin(angle)
            retreat=(dx*old[0]+dy*old[1])/max(.001,math.hypot(*old[:2]))>.5 and contact['range_ft']>6
        hit=bool(obs.get('incoming_damage'))
        reason=None
        if hit and not first_hit:first_hit=True;reason='first_injury'
        elif retreat and not seen_retreat:seen_retreat=True;reason='first_observed_retreat'
        elif hit and seen_retreat and not seen_return:seen_return=True;reason='renewed_injury'
        if reason:
            before=copy.deepcopy(spider,{id(model):model})
            # Encode this observation exactly once in the actual fixed-memory path.
            previous_goal=spider.goal;command=spider.choose(obs)
            latent=spider.hunter.hidden[0,0].detach().tolist()
            context=private_context(obs,before.meal,previous_goal)
            values=[]
            origin_index=len(origins)
            for action,mode in enumerate(MODES):
                for draw in range(draws):
                    seed=case['seed']*100000+turn*100+draw+17
                    labels=fork_outcome(world,before,beast,mode,seed)
                    values.append(dict(action=action,draw=draw,targets=labels))
            origins.append(dict(split=split,seed=case['seed'],condition={**case,'return_delay':return_delay},
                                turn=turn,reason=reason,latent=latent,context=context,branches=values))
        else:command=spider.choose(obs)
        previous_contact=dict(contact) if contact else None
        world.act(SPIDER,command);world.elapsed_seconds+=6
        if world.hunger[SPIDER]<=.001:break
    return origins


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--train-seeds',type=int,default=8);parser.add_argument('--val-seeds',type=int,default=2);parser.add_argument('--test-seeds',type=int,default=4)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);model=load(args.model);model.requires_grad_(False)
    started=time.monotonic();counts={};reason_counts={}
    with gzip.open(args.output/'forks.jsonl.gz','wt') as out:
        for split,start,n,draws in (('train',81000,args.train_seeds,2),('val',82000,args.val_seeds,2),('test',83000,args.test_seeds,4)):
            cases=list(scenarios(range(start,start+n)));count=0;reasons={}
            for i,case in enumerate(cases):
                # Training sees 2/4-turn returns, validation 3, and test 5: delay
                # is hidden from inputs and the test return schedule is unseen.
                delay=(2 if case['seed']%2==0 else 4) if split=='train' else 3 if split=='val' else 5
                records=collect(case,model,split,draws,delay)
                for record in records:
                    out.write(json.dumps(record)+'\n');count+=1
                    reasons[record['reason']]=reasons.get(record['reason'],0)+1
                if (i+1)%48==0:print(f'{split}: {i+1}/{len(cases)} conditions, {count} origins; {time.monotonic()-started:.1f}s',flush=True)
            counts[split]=count;reason_counts[split]=reasons
    metadata=dict(schema='meal-forecast-forks-v1',horizons=HORIZONS,context_names=CONTEXT_NAMES,outcome_names=OUTCOME_NAMES,
                  modes=MODES,origin_counts=counts,reason_counts=reason_counts,elapsed_seconds=time.monotonic()-started,
                  frozen_model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                  split_seeds={'train':list(range(81000,81000+args.train_seeds)),'val':list(range(82000,82000+args.val_seeds)),'test':list(range(83000,83000+args.test_seeds))},
                  limits=['Same two authored rooms; disjoint seeds and held-out return delay, not held-out geography',
                          'Each choice commits to a scripted continuation, not one isolated physical action',
                          'Decision triggers select injury/retreat histories and are not ordinary-turn prevalence',
                          'Opponent condition/return delay are metadata only, never predictor inputs'])
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))

if __name__=='__main__':main()
