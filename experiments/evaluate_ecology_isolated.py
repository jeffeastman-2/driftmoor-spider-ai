"""Held-out species tests with unchanged scripted opponents."""
import argparse
import json
import math
from pathlib import Path
import torch
from experiments.train_ecology_djepa import training_world,FOCUS
from spider_sim.ecology import EcologyEncounter,baseline_policies,ecology_round
from spider_sim.djepa_ecology import SPECIES,CautiousTeacher,NeuralPolicy,load,senses


def facing_wall(world,actor):
    """Native ten-foot sensor confirms a wall within five feet straight ahead."""
    return senses(world,actor)['self']['local_clearance_ft'][0]<5


def evaluate(directory,seeds=range(350000,350016),rounds=24,species_list=SPECIES,report_name="isolated-report.json"):
    result={}
    for species in species_list:
        model=load(directory/(species+'.pt'),species);actor=FOCUS[species]
        groups={}
        for suite in ('focused','authored'):
            groups[suite]={}
            for mode in ('scripted','teacher','proposal','predictive'):
                total={'episodes':0,'survived':0,'hp_lost':0,'attacks':0,'admitted':0,'damage':0,
                       'moves':0,'blocked':0,'distance_ft':0.,'wall_starts':0,'wall_escaped':0,'max_blocked_streak':0}
                traces=[]
                for seed in seeds:
                    world=training_world(species,seed) if suite=='focused' else EcologyEncounter(seed)
                    policies=baseline_policies(world,seed)
                    if mode in ('proposal','predictive'):policies[actor]=NeuralPolicy(model,species,mode)
                    elif mode=='teacher':
                        policies[actor]=CautiousTeacher(species);policies[actor].neural=True
                    origin=world.positions[actor][:2];streak=0;escaped=False;rows=[]
                    wall=suite=='focused' and facing_wall(world,actor)
                    total['wall_starts']+=int(wall)
                    initial_hp=world.config.characters[actor].character.current_hp
                    for turn in range(rounds):
                        records=ecology_round(world,policies,turn)
                        row=next((r for r in records if r['actor']==actor),None)
                        if row is None:break
                        rows.append(row);r=row['result'];action=row['command']['action']
                        if action=='attack':
                            total['attacks']+=1;total['admitted']+=int(r['admitted']);total['damage']+=r['damage']
                        if action=='move':
                            total['moves']+=1;total['blocked']+=int(r['status']=='blocked');total['distance_ft']+=r['distance_ft']
                        streak=streak+1 if action=='move' and r['status']=='blocked' else 0
                        total['max_blocked_streak']=max(total['max_blocked_streak'],streak)
                        if wall and turn<6 and math.dist(origin,world.positions[actor][:2])>=5:escaped=True
                    hp=world.config.characters[actor].character.current_hp
                    total['episodes']+=1;total['survived']+=int(hp>0);total['hp_lost']+=initial_hp-hp
                    total['wall_escaped']+=int(escaped)
                    traces.append({'seed':seed,'records':rows})
                groups[suite][mode]=total
                (directory/(f'isolated-{species}-{suite}-{mode}.json')).write_text(json.dumps(traces))
                print(species,suite,mode,total,flush=True)
        result[species]=groups
    report={'seeds':list(seeds),'rounds':rounds,'wall_escape_definition':'At least five feet displacement within six rounds from a start with native measured forward clearance under five feet',
            'opponents':'Existing scripted opponents; only focal actor controller changes; focused starts can have reduced HP',
            'results':result}
    (directory/report_name).write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--models',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1);evaluate(args.models)
