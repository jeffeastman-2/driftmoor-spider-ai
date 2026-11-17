"""Small disjoint-seed full-ecology check of the controlled meal-defense comparison."""
import argparse
import json
from collections import Counter
from pathlib import Path
from experiments.compare_meal_defense import ComparisonPolicy,MODES
from spider_sim.ecology import EcologyEncounter,baseline_policies,ecology_round
from spider_sim.pursuit import load,PursuitPolicy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,default=8);parser.add_argument('--rounds',type=int,default=120)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False);model=load(args.model);episodes=[]
    for seed in range(74000,74000+args.seeds):
        for mode in MODES:
            w=EcologyEncounter(seed,feeding_cycle=True,replenish_prey=True,random_prey=True,spider_pair=True)
            policies=baseline_policies(w,seed)
            for n,cid in enumerate(w.spiders):
                policies[cid]=ComparisonPolicy(PursuitPolicy(model,approach=True,facing='track',rest_without_goal=True,local_motion=True,seed=seed+n),mode)
            full={};food=0.;damage=0;chases=0;repeats=Counter();previous={};births=0
            for turn in range(args.rounds):
                for row in ecology_round(w,policies,turn):
                    target=w._target(row['actor'],row['result'].get('target'))
                    if target in w.spiders:damage+=row['result']['damage']
                    if row['actor'] not in w.spiders:continue
                    cid=row['actor'];food+=row['result'].get('consumed',0)
                    defending=row['goal']=='defend'
                    if defending and not previous.get(cid,False):repeats[cid]+=1
                    previous[cid]=defending
                    chases+=int(defending and row['command']['action']=='move')
                births+=len(w.last_births)
                for cid in w.spiders:
                    if w.hunger[cid]<=.001 and cid not in full:full[cid]=turn+1
            episodes.append(dict(seed=seed,mode=mode,surviving_spiders=sum(w.config.characters[c].character.current_hp>0 for c in w.spiders),
                fed_spiders=len(full),food_consumed=food,damage_taken=damage,defense_chase_turns=chases,
                repeat_interruptions=sum(max(0,n-1) for n in repeats.values()),courtships=len(w.mating_events),births=births))
        print(f'Full ecology seed {seed} complete',flush=True)
    summary={mode:{'episodes':len(rr:=[r for r in episodes if r['mode']==mode]),
                   **{key:sum(r[key] for r in rr)/len(rr) for key in ('surviving_spiders','fed_spiders','food_consumed','damage_taken','defense_chase_turns','repeat_interruptions','courtships')}} for mode in MODES}
    report={'rounds':args.rounds,'seeds':list(range(74000,74000+args.seeds)),'summary':summary,'episodes':episodes,
            'limits':['Small ecology sanity check, not sufficient to select a general controller','Two spiders per run, all other residents scripted','Normal authored initial population; random replacement prey; no standardized meal']}
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
