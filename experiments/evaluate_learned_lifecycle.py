"""Prospective matched teacher/student ecology runs; no live viewer changes."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import gzip,json,math,multiprocessing,time,hashlib
from pathlib import Path
import torch
from experiments.watch_level2 import LiveRun
from experiments.run_lifecycle_long import cache_static_geometry
from experiments.lifecycle_experience import ExperienceRecorder,Coverage
from spider_sim.ecology import ecology_round
from spider_sim.learned_lifecycle import LearnedLifecyclePolicy,load

SEEDS=[151,163,179,191,211,223,239,251]


def run_one(output,pursuit,weights,seed,turns,mode,current_profile=False,resume=False):
    torch.set_num_threads(1);output=Path(output);started=time.monotonic()
    if resume and (output/'summary.json').exists():
        saved=json.loads((output/'summary.json').read_text())
        if (saved['seed'],saved['turns'],saved['mode'],saved.get('current_profile',False))!=(seed,turns,mode,current_profile):raise ValueError('Cached run mismatch')
        return {**saved,'cached':True}
    if resume and output.exists():
        index=1
        while output.with_name(output.name+f'.partial-{index}').exists():index+=1
        output.rename(output.with_name(output.name+f'.partial-{index}'))
    run=LiveRun(output,seed,turns,spider_model=Path(pursuit),spider_facing='track',spider_local_motion=True,
        replenish_prey=True,feeding_cycle=True,random_prey=True,spider_pair=True,lifecycle=True)
    world=run.world;cache_static_geometry(world.geometry)
    if mode!='teacher':
        model=load(weights)
        for actor in world.spiders:
            run.policies[actor]=LearnedLifecyclePolicy(run.policies[actor].hunter,model,world.spiders[actor])
    recorder=ExperienceRecorder(world,run.policies);coverage=Coverage();goals=Counter();statuses=Counter();events=Counter()
    food=0.;decisions=0;first_egg=None;peak=2;deaths=0;starvation=0
    with gzip.open(output/'decisions.jsonl.gz','wt',compresslevel=1) as log,(output/'population.jsonl').open('w') as population:
        for turn in range(turns):
            if current_profile:
                from experiments.lifecycle_rules import current_rules
                current_rules(run,turn)
            else:run.set_hunger_rate(turn)
            for actor in world.spiders:
                policy=run.policies[actor]
                if isinstance(policy,LearnedLifecyclePolicy):
                    policy.proprioception=lambda actor=actor:{'sex':world.spiders[actor],
                        'mate_cooldown_turns':math.ceil(max(0,world.mate_cooldown.get(actor,0)-world.elapsed_seconds)/6)}
            recorder.prepare();before=set(world.active)&set(world.spiders)
            records=ecology_round(world,run.policies,turn)
            for row in recorder.decisions(seed,turn+1,records):
                # Recorder field is historically named teacher_goal; for student
                # runs it is the student's executed goal, never a teacher label.
                row['controller']=mode
                if mode!='teacher':row['goal_probabilities']=run.policies[row['actor']].last_probabilities
                coverage.add(row);decisions+=1
                goals[row['teacher_goal']]+=1;statuses[row['result']['status']]+=1
                food+=row['result'].get('consumed',0)
                if row['result'].get('egg_sac') and first_egg is None:first_egg=turn+1
                log.write(json.dumps(row,separators=(',',':'))+'\n')
            for actor in before-world.active:deaths+=1;coverage.end_actor(seed,actor,turn+1)
            for row in records:events.update(e['event'] for e in row.get('lifecycle_events',[]))
            living=len(world.active&set(world.spiders));peak=max(peak,living)
            population.write(json.dumps({'turn':turn+1,'spiders':living})+'\n')
            if (turn+1)%100==0:(output/'progress.json').write_text(json.dumps({'turn':turn+1,'living':living})+'\n')
        for key in list(coverage.defenses):coverage.finish_defense(key,'right_censored',turns)
    result={'current_profile':current_profile,'seed':seed,'mode':mode,'turns':turns,'decisions':decisions,'living_spiders':living,'peak_spiders':peak,
        'first_egg_turn':first_egg,'eggs':events['egg_sac_laid'],'hatches':events['egg_sac_hatched'],
        'deaths':deaths,'starvation_deaths':events['starvation_death'],'food_consumed':food,
        'goals':dict(goals),'statuses':dict(statuses),'events':dict(events),
        'unexecutable_goals':sum(getattr(p,'unexecutable',0) for p in run.policies.values()),
        'all_spider_policies_learned':all(isinstance(run.policies[a],LearnedLifecyclePolicy) for a in world.spiders) if mode!='teacher' else None,
        'interrupted_meals':coverage.report()['defense_outcomes'],'seconds':time.monotonic()-started}
    (output/'coverage.json').write_text(json.dumps(coverage.report(),indent=2)+'\n')
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--pursuit',required=True);p.add_argument('--model');p.add_argument('--turns',type=int,default=1000);p.add_argument('--workers',type=int,default=3);p.add_argument('--seeds',type=int,nargs='+',default=SEEDS);p.add_argument('--purpose',choices=('prospective','diagnostic'),default='prospective');p.add_argument('--current-rules',action='store_true');p.add_argument('--models',nargs='+');p.add_argument('--resume',action='store_true');a=p.parse_args()
    models={'student':a.model} if a.model else {}
    for item in a.models or []:
        name,path=item.split('=',1)
        if name=='teacher' or not name.replace('_','').replace('-','').isalnum() or name in models:raise ValueError('Invalid or duplicate model label')
        models[name]=path
    if not models:p.error('Provide --model or --models LABEL=PATH')
    a.output.mkdir(parents=True,exist_ok=True)
    manifest={'seeds':a.seeds,'turns':a.turns,'models':models,'pursuit':a.pursuit,'current_rules':a.current_rules,
        'selection':'argmax learned lifecycle logits, no teacher goal fallback; identical motor/engine rules',
        'newborns':'same learned weights, separate fresh GRU state','origin':a.purpose+' seed comparison',
        'model_sha256':{name:hashlib.sha256(Path(path).read_bytes()).hexdigest() for name,path in models.items()}}
    manifest_path=a.output/'manifest.json'
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text())
        if not a.resume:raise ValueError('Existing evaluation; use --resume or a new output directory')
        if any(previous.get(key)!=value for key,value in manifest.items()):raise ValueError('Evaluation manifest mismatch')
        manifest['logging_history']=previous.get('logging_history',['gzip compression level9'])
    else:manifest['logging_history']=[]
    if not manifest['logging_history'] or manifest['logging_history'][-1]!='gzip compression level1':manifest['logging_history'].append('gzip compression level1')
    manifest['workers']=a.workers
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    results=[]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(run_one,str(a.output/f'{mode}-{seed}'),a.pursuit,weights,seed,a.turns,mode,a.current_rules,a.resume) for seed in a.seeds for mode,weights in [('teacher',None),*models.items()]]
        for future in as_completed(futures):
            r=future.result();results.append(r);print(json.dumps({k:r[k] for k in ('seed','mode','living_spiders','eggs','food_consumed','seconds')}),flush=True)
            (a.output/'results.json').write_text(json.dumps(sorted(results,key=lambda r:(r['seed'],r['mode'])),indent=2)+'\n')

if __name__=='__main__':main()
