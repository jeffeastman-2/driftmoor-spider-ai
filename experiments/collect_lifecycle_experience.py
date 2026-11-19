"""Collect whole-seed lifecycle experience without changing the live viewer."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import gzip
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
import torch

from experiments.watch_level2 import LiveRun
from experiments.run_lifecycle_long import cache_static_geometry
from experiments.lifecycle_experience import ExperienceRecorder,Coverage,SCHEMA
from spider_sim.ecology import ecology_round

SPLITS={'train':[11,23,37,42,53,67,79,83],'validation':[97,109],'test':[127,139]}


def collect_one(output,weights,seed,turns,split):
    output=Path(output);weights=Path(weights);summary_path=output/'summary.json'
    model_hash=hashlib.sha256(weights.read_bytes()).hexdigest()
    if summary_path.exists():
        saved=json.loads(summary_path.read_text())
        if saved['completed'] and saved['seed']==seed and saved['turns']==turns and saved['model_sha256']==model_hash and saved['split']==split:
            return saved
        raise ValueError('Existing corpus parameters differ: '+str(output))
    torch.set_num_threads(1)
    run=LiveRun(output,seed,turns,spider_model=weights,spider_facing='track',
        spider_local_motion=True,replenish_prey=True,feeding_cycle=True,
        random_prey=True,spider_pair=True,lifecycle=True)
    world=run.world;cache_static_geometry(world.geometry)
    recorder=ExperienceRecorder(world,run.policies);coverage=Coverage()
    started=time.monotonic();decisions=0;goals=Counter();events=Counter();actors={}
    with gzip.open(output/'decisions.jsonl.gz','wt',encoding='utf-8') as log, (output/'population.jsonl').open('w') as population:
        for turn in range(turns):
            run.set_hunger_rate(turn);recorder.prepare()
            alive_before=set(world.active)&set(world.spiders)
            records=ecology_round(world,run.policies,turn)
            data=recorder.decisions(seed,turn+1,records)
            for row in data:
                row['split']=split
                log.write(json.dumps(row,separators=(',',':'))+'\n')
                coverage.add(row);goals[row['teacher_goal']]+=1;decisions+=1
            for actor in world.spiders:
                actors.setdefault(actor,{'sex':world.spiders[actor],'birth_turn':0 if actor in ('freight_spider','freight_spider_female') else turn+1,'death_turn':None})
            for actor in alive_before-world.active:
                actors[actor]['death_turn']=turn+1;coverage.end_actor(seed,actor,turn+1)
            for row in records:
                events.update(e['event'] for e in row.get('lifecycle_events',[]))
            alive=[c for c in world.spiders if c in world.active]
            population.write(json.dumps({'turn':turn+1,'spiders':len(alive),
                'males':sum(world.spiders[c]=='male' for c in alive),
                'females':sum(world.spiders[c]=='female' for c in alive),
                'food_consumed':sum(r['result'].get('consumed',0) for r in records)})+'\n')
            if (turn+1)%100==0 or turn+1==turns:
                log.flush();population.flush()
                (output/'progress.json').write_text(json.dumps({'seed':seed,'turn':turn+1,'decisions':decisions,'spiders':len(alive),'elapsed_seconds':round(time.monotonic()-started,2)})+'\n')
        # End-of-run truncation is not a death or an unsuccessful defense.
        for key in list(coverage.defenses):coverage.finish_defense(key,'right_censored',turns)
    (output/'coverage.json').write_text(json.dumps(coverage.report(),indent=2)+'\n')
    (output/'actors.json').write_text(json.dumps(actors,indent=2)+'\n')
    summary={'schema':SCHEMA,'completed':True,'seed':seed,'split':split,'turns':turns,
        'decisions':decisions,'actors':len(actors),'living_spiders':len(alive),
        'goals':dict(goals),'events':dict(events),'model_sha256':model_hash,
        'elapsed_seconds':time.monotonic()-started,
        'rules':{'hunger':run.hunger_schedule,'starvation':run.starvation_schedule,
                 'behavior':run.behavior_schedule,'zone_web':run.zone_web_schedule}}
    summary_path.write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--spider-model',type=Path,required=True)
    parser.add_argument('--turns',type=int,default=1000)
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if args.turns<1 or args.workers<1:parser.error('positive turns and workers required')
    args.output.mkdir(parents=True,exist_ok=True)
    manifest={'schema':SCHEMA,'splits':SPLITS,'turns_per_seed':args.turns,
        'model_sha256':hashlib.sha256(args.spider_model.read_bytes()).hexdigest(),
        'origin':'ordinary teacher-controlled episodes; no scenario interventions',
        'inputs':'private observation plus own sex and own cooldown; no teacher flags, actor IDs or audit labels',
        'teacher':'authored lifecycle goals with fixed learned pursuit motor',
        'sequence_key':['seed','actor'],'holdout_unit':'whole seed',
        'terminal_handling':'actors.json records death separately; living sequence ends are right-censored'}
    path=args.output/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('Corpus manifest mismatch')
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(collect_one,str(args.output/f'seed-{seed}'),str(args.spider_model),seed,args.turns,split):seed for split,seeds in SPLITS.items() for seed in seeds}
        for future in as_completed(futures):
            result=future.result();results.append(result)
            print(json.dumps({'finished_seed':result['seed'],'split':result['split'],'decisions':result['decisions'],'living_spiders':result['living_spiders'],'elapsed_seconds':round(result['elapsed_seconds'],1)}),flush=True)
            (args.output/'collection-progress.json').write_text(json.dumps({'completed_seeds':sorted(r['seed'] for r in results),'total_seeds':sum(map(len,SPLITS.values()))},indent=2)+'\n')
    (args.output/'collection-summary.json').write_text(json.dumps(sorted(results,key=lambda r:r['seed']),indent=2)+'\n')


if __name__=='__main__':main()
