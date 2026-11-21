"""Student-only rollouts with non-executed teacher corrections on training seeds."""
import argparse,gzip,json,math,time,multiprocessing,hashlib
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import torch
from experiments.watch_level2 import LiveRun
from experiments.run_lifecycle_long import cache_static_geometry
from experiments.lifecycle_rules import current_rules,RULES
from experiments.collect_lifecycle_experience import SPLITS
from experiments.lifecycle_experience import ExperienceRecorder
from spider_sim.ecology import ecology_round
from spider_sim.learned_lifecycle import LearnedLifecyclePolicy,load
from spider_sim.recovery_advisor import RecoveryAdvisor


def collect_one(output,pursuit,weights,seed,turns):
    torch.set_num_threads(1);output=Path(output);start=time.monotonic()
    run=LiveRun(output,seed,turns,spider_model=Path(pursuit),spider_facing='track',spider_local_motion=True,
        replenish_prey=True,feeding_cycle=True,random_prey=True,spider_pair=True,lifecycle=True)
    world=run.world;cache_static_geometry(world.geometry);model=load(weights)
    for actor in world.spiders:run.policies[actor]=LearnedLifecyclePolicy(run.policies[actor].hunter,model,world.spiders[actor])
    recorder=ExperienceRecorder(world,run.policies);wrapped=set();labels={};goals=Counter();corrections=Counter();statuses=Counter();decisions=0
    with gzip.open(output/'decisions.jsonl.gz','wt') as log:
        for turn in range(turns):
            current_rules(run,turn);labels.clear()
            for actor in world.spiders:
                policy=run.policies[actor]
                if not isinstance(policy,LearnedLifecyclePolicy):raise AssertionError('Offspring lost learned controller')
                policy.proprioception=lambda actor=actor:{'sex':world.spiders[actor],
                    'mate_cooldown_turns':math.ceil(max(0,world.mate_cooldown.get(actor,0)-world.elapsed_seconds)/6)}
                if actor in wrapped:continue
                advisor=RecoveryAdvisor();choose=policy.choose
                def capture(obs,actor=actor,policy=policy,advisor=advisor,choose=choose):
                    labels[actor]=advisor.recommend(obs,policy)
                    return choose(obs)  # Only this command reaches the engine.
                policy.choose=capture;wrapped.add(actor)
            recorder.prepare();rows=ecology_round(world,run.policies,turn)
            for row in recorder.decisions(seed,turn+1,rows):
                label,proposal=labels[row['actor']];actual=row['teacher_goal']
                row.update(split='train',executed_goal=actual,teacher_goal=label,advisor_command=proposal,
                    label_source='private-observation teacher correction; student command executed')
                decisions+=1;goals[label]+=1;statuses[row['result']['status']]+=1
                if label!=actual:corrections[actual+' -> '+label]+=1
                log.write(json.dumps(row,separators=(',',':'))+'\n')
            if (turn+1)%100==0:(output/'progress.json').write_text(json.dumps({'turn':turn+1,'decisions':decisions,'living':len(world.active&set(world.spiders))})+'\n')
    summary={'seed':seed,'turns':turns,'decisions':decisions,'living_spiders':len(world.active&set(world.spiders)),
        'teacher_goals':dict(goals),'corrections':dict(corrections),'statuses':dict(statuses),
        'all_spider_policies_learned':all(isinstance(run.policies[a],LearnedLifecyclePolicy) for a in world.spiders),'seconds':time.monotonic()-start}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--pursuit',required=True);p.add_argument('--model',required=True);p.add_argument('--turns',type=int,default=500);p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    manifest={'schema':'lifecycle-recovery-v2','splits':{'train':SPLITS['train']},'turns_per_seed':a.turns,'rules':RULES,
        'student_model':a.model,'student_sha256':hashlib.sha256(Path(a.model).read_bytes()).hexdigest(),
        'pursuit_sha256':hashlib.sha256(Path(a.pursuit).read_bytes()).hexdigest(),
        'execution':'Student argmax goal and command only; advisor never acts. Separate advisor per actor; fresh at birth.',
        'labels':'Teacher receives same private senses, actual prior goal and motor bookkeeping; no world queries, future outcomes or GRU states.'}
    if (a.output/'manifest.json').exists():raise ValueError('Use a new recovery corpus directory')
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');results=[]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        fs=[pool.submit(collect_one,str(a.output/f'seed-{s}'),a.pursuit,a.model,s,a.turns) for s in SPLITS['train']]
        for f in as_completed(fs):
            r=f.result();results.append(r);print(json.dumps(r),flush=True)
            (a.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')

if __name__=='__main__':main()
