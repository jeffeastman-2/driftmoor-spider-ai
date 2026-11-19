"""Run the current live ecology without rendering or wall-clock turn delays."""
import argparse
import collections
import hashlib
import json
import time
from pathlib import Path

import torch

from experiments.watch_level2 import LiveRun
from spider_sim.ecology import ecology_round


def cache_static_geometry(geometry):
    """Exact-coordinate memoization, valid only while this geometry is unchanged."""
    from functools import lru_cache
    from models.space_time import Position
    # A point outside a polygon's exact bounding rectangle cannot be inside.
    # Keep the original ray test unchanged for all points within that rectangle.
    for polygon in geometry.zones.values():
        original_contains=polygon.contains
        xs=[v[0] for v in polygon.vertices];ys=[v[1] for v in polygon.vertices]
        if not xs:continue
        def bounded(method,x0,y0,x1,y1):
            return lambda point: x0<=point.x<=x1 and y0<=point.y<=y1 and method(point)
        polygon.contains=bounded(original_contains,min(xs),min(ys),max(xs),max(ys))
    for name in ('zone_at','floor_z_at'):
        original=getattr(geometry,name)
        def make_cached(method):
            @lru_cache(maxsize=65536)
            def cached(x,y,z):return method(Position(x,y,z))
            return lambda point:cached(point.x,point.y,point.z)
        setattr(geometry,name,make_cached(original))
    original_los=geometry.can_see
    @lru_cache(maxsize=131072)
    def cached_los(ax,ay,az,bx,by,bz):
        return original_los(Position(ax,ay,az),Position(bx,by,bz))
    geometry.can_see=lambda a,b:cached_los(a.x,a.y,a.z,b.x,b.y,b.z)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--spider-model',type=Path,required=True)
    parser.add_argument('--turns',type=int,default=10000)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--cache-geometry',action='store_true')
    args=parser.parse_args()
    if args.turns<1:parser.error('turns must be positive')
    # Tiny per-creature networks run without a large CPU thread pool.
    torch.set_num_threads(1)
    run=LiveRun(args.output,args.seed,args.turns,spider_model=args.spider_model,
        spider_facing='track',spider_local_motion=True,replenish_prey=True,
        feeding_cycle=True,random_prey=True,spider_pair=True,lifecycle=True)
    world=run.world
    if args.cache_geometry:cache_static_geometry(world.geometry)
    started=time.monotonic()
    event_counts=collections.Counter()
    metadata={'seed':args.seed,'turns_requested':args.turns,
        'model_sha256':hashlib.sha256(args.spider_model.read_bytes()).hexdigest(),
        'controller_signature':run.controller_signature,
        'hunger_schedule':run.hunger_schedule,'starvation_schedule':run.starvation_schedule,
        'behavior_schedule':run.behavior_schedule,'zone_web_schedule':run.zone_web_schedule,
        'initial_spiders':2,'learning':False,'torch_threads':1,'exact_geometry_cache':args.cache_geometry}
    (args.output/'parameters.json').write_text(json.dumps(metadata,indent=2)+'\n')
    def metrics(turn,records):
        living=[c for c in world.spiders if world.config.characters[c].character.current_hp>0]
        events=[e for row in records for e in row.get('lifecycle_events',[])]
        counts=collections.Counter(e['event'] for e in events)
        event_counts.update(counts)
        births=2*counts['egg_sac_hatched']
        alive=collections.Counter()
        for c in world.active:
            if c in world.spiders:continue
            key=('rats' if c.startswith('cave_rat_') else 'crickets' if c.startswith('cave_cricket_') else 'shadowbeasts' if c.startswith('freight_shadowbeast') else 'lizards' if c.startswith('pen_lizard_') else 'other')
            alive[key]+=1
        values=[world.hunger[c] for c in living]
        row={'turn':turn,'spiders':len(living),
            'males':sum(world.spiders[c]=='male' for c in living),
            'females':sum(world.spiders[c]=='female' for c in living),
            'births':births,'starvation_deaths':counts['starvation_death'],
            'food_consumed':sum(r['result'].get('consumed',0) for r in records),
            'food_remaining':sum(s['remaining'] for s in world.food.values()),
            'mean_hunger':sum(values)/len(values) if values else None,
            'max_hunger':max(values,default=None),
            'goals':dict(collections.Counter(r.get('goal') for r in records if r['actor'] in world.spiders)),
            'prey':dict(alive)}
        return row,events
    previous=2;peak=2;peak_turn=0
    with (args.output/'population.jsonl').open('w') as log,(args.output/'events.jsonl').open('w') as eventlog:
        row,_=metrics(0,[]);row['deaths']=0;log.write(json.dumps(row)+'\n')
        for turn in range(args.turns):
            run.set_hunger_rate(turn)
            records=ecology_round(world,run.policies,turn)
            row,events=metrics(turn+1,records)
            row['deaths']=previous+row['births']-row['spiders']
            assert row['deaths']>=0
            previous=row['spiders']
            if previous>peak:peak=previous;peak_turn=turn+1
            log.write(json.dumps(row)+'\n')
            for event in events:eventlog.write(json.dumps({'turn':turn+1,**event})+'\n')
            if (turn+1)%100==0 or turn+1==args.turns:
                log.flush();eventlog.flush()
                progress={'turn':turn+1,'spiders':previous,'peak':peak,'elapsed_seconds':round(time.monotonic()-started,1)}
                (args.output/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
                print(json.dumps(progress),flush=True)
    summary={**metadata,'turns_completed':args.turns,'final_spiders':previous,
        'peak_spiders':peak,'peak_turn':peak_turn,'events':dict(event_counts),
        'elapsed_seconds':time.monotonic()-started}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
