"""Full population runs with opt-in food-supported healing and mating cutoff."""
import argparse,gzip,json,hashlib,time,multiprocessing
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import torch
from experiments.watch_level2 import LiveRun
from experiments.run_lifecycle_long import cache_static_geometry
from experiments.lifecycle_experience import ExperienceRecorder
from experiments.collect_lifecycle_experience import SPLITS
from spider_sim.ecology import ecology_round


def run_one(output,weights,seed,turns):
    torch.set_num_threads(1);output=Path(output);start=time.monotonic()
    run=LiveRun(output,seed,turns,spider_model=Path(weights),spider_facing='track',spider_local_motion=True,
        replenish_prey=True,feeding_cycle=True,random_prey=True,spider_pair=True,lifecycle=True)
    w=run.world;cache_static_geometry(w.geometry);recorder=ExperienceRecorder(w,run.policies)
    events=Counter();goals=Counter();healed=0;food=0.;peak=2;first=None
    with gzip.open(output/'decisions.jsonl.gz','wt') as log,(output/'population.jsonl').open('w') as pop:
        for t in range(turns):
            run.set_hunger_rate(t)
            w.healing_hp_per_round=1;w.healing_hunger_limit=.1;w.mate_hunger_limit=.3
            for policy in run.policies.values():
                policy.mate_hunger_threshold=.1;policy.resume_hunting=.3;policy.mate_hunger_limit=.3
            recorder.prepare();rows=ecology_round(w,run.policies,t)
            for r in recorder.decisions(seed,t+1,rows):
                log.write(json.dumps(r,separators=(',',':'))+'\n');goals[r['teacher_goal']]+=1;food+=r['result'].get('consumed',0)
                if r['result'].get('egg_sac') and first is None:first=t+1
            for e in w.lifecycle_events:
                events[e['event']]+=1;healed+=e.get('healed_hp',0)
            live=w.active&set(w.spiders);peak=max(peak,len(live))
            pop.write(json.dumps({'turn':t+1,'spiders':len(live),'living_hp':sum(w.config.characters[a].character.current_hp for a in live),
                'healed_hp':healed,'food_consumed':food,'egg_sacs':events['egg_sac_laid'],'starvation_deaths':events['starvation_death']})+'\n')
            if (t+1)%100==0:
                (output/'progress.json').write_text(json.dumps({'turn':t+1,'living':len(live),'healed_hp':healed,'seconds':time.monotonic()-start})+'\n')
    result={'seed':seed,'turns':turns,'first_egg_turn':first,'living_spiders':len(live),'peak_spiders':peak,
        'actors':len(w.spiders),'events':dict(events),'healed_hp':healed,'food_consumed':food,'goals':dict(goals),'seconds':time.monotonic()-start}
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--weights',required=True);p.add_argument('--turns',type=int,default=1000);p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);seeds=[s for ss in SPLITS.values() for s in ss]
    manifest={'seeds':seeds,'turns':a.turns,'checkpoint':'69ad4cd','teacher':'response teacher; fixed learned pursuit; no learned lifecycle selector',
        'weights_sha256':hashlib.sha256(Path(a.weights).read_bytes()).hexdigest(),
        'healing':'1 HP per round after feeding OR when post-round hunger <10%, capped, living spiders only',
        'mating':'initiate below10%, retain search and admit mating below30%; existing health/cooldown rules',
        'baseline':'lifecycle-response-v1','purpose':'full population outcomes on same diagnostic seeds, not independent confirmation'}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');results=[]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(run_one,str(a.output/f'seed-{s}'),a.weights,s,a.turns) for s in seeds]
        for f in as_completed(futures):
            r=f.result();results.append(r);print(json.dumps(r),flush=True)
            (a.output/'results.json').write_text(json.dumps(sorted(results,key=lambda x:x['seed']),indent=2)+'\n')

if __name__=='__main__':main()
