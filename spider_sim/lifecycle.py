"""Authored population and starvation mechanics for the optional live ecology.

Two-turn adult hatching is deliberate accelerated experiment pacing. The existing
Driftmoor spider swarm remains available for a future juvenile stage.
"""
import copy
import math
from models.body_geometry import Orientation
from models.space_time import Position
from models.thing import Thing
from engine.sensory import floor_relative_position
from engine.geometry import _check_impassable

HATCH_ROUNDS=2
STARVATION_INTERVAL=1
STARVATION_DAMAGE=1
BEAST='freight_shadowbeast'


def initialize(world):
    world.egg_sacs={};world.lifecycle_events=[];world.starvation_ticks={}
    world.replaced_beasts=set()
    world.exhausted_since={};world.retired_carcasses=set()
    world.life_templates={cid:copy.deepcopy(world.config.characters[cid]) for cid in (next(iter(world.spiders)),BEAST)}


def new_id(world,prefix):
    index=2
    while prefix+str(index) in world.positions:index+=1
    return prefix+str(index)


def nearby_position(world,origin,body,reserved=()):
    # Require a local body footprint on walkable floor, outside local obstacles.
    obstacles=[o for o in world.geometry.impassable if not all(o.contains(p.centroid()) for p in world.geometry.zones.values())]
    for radius in (4,6,8,12):
        angles=list(range(0,360,30));world.spawn_random.shuffle(angles)
        for angle in angles:
            x=origin[0]+radius*math.cos(math.radians(angle));y=origin[1]+radius*math.sin(math.radians(angle))
            points=[Position(x+dx,y+dy) for dx in (-body.width/2,0,body.width/2) for dy in (-body.depth/2,0,body.depth/2)]
            if not all(world.geometry.zone_at(p) and not any(o.contains(p) for o in obstacles) for p in points):continue
            clear,_,_=_check_impassable('hatchling',origin,(x,y,''),world.geometry.impassable,world.geometry.zones)
            if not clear:continue
            occupied=[p for cid,p in world.positions.items() if cid in world.active]+list(reserved)
            if any(math.dist((x,y),p[:2])<body.width for p in occupied):continue
            return (x,y,world.geometry.zone_at(Position(x,y)))
    # A crowded nursery defers hatching instead of teleporting offspring.
    return None


def clone_actor(world,source,cid,name,position):
    template=world.life_templates[source]
    session=copy.deepcopy(template);session.character.id=cid;session.character.name=name
    session.character.current_hp=world.max_hp[source]
    world.config.characters[cid]=session;world.positions[cid]=position
    world.heights[cid]=0.;world.bodies[cid]=world.bodies[source]
    world.orientations[cid]=Orientation();world.sockets[cid]=world.sockets[source]
    world.max_hp[cid]=world.max_hp[source];world.active.add(cid)
    world.contact_ids[cid]=f'contact_{len(world.contact_ids)+1}'
    world.tokens={actor:{other:token for other,token in world.contact_ids.items() if other!=actor} for actor in world.positions}


def lay_eggs(world,pair):
    female=next(cid for cid in pair if world.spiders[cid]=='female')
    key=f'spider_egg_sac_{len(world.egg_sacs)+1}'
    position=world.positions[female]
    sac=dict(id=key,parents=list(pair),position=list(position),laid_at=world.elapsed_seconds+6,
             hatch_at=world.elapsed_seconds+6+HATCH_ROUNDS*6,hatched=False)
    world.egg_sacs[key]=sac
    point=floor_relative_position(world.geometry,position,world.heights[female])
    world.config.items[key]=Thing(id=key,label=f'Spider Egg Sac {len(world.egg_sacs)}',description='Silken brood sac; hatches after two rounds.',
                                position=point,zone=position[2],size_ft=2,emoji='🥚',state={'egg_sac':'incubating'})
    world.lifecycle_events.append({'event':'egg_sac_laid','actor':female,'sac':key,'parents':list(pair)})
    return key


def start_round(world):
    world.lifecycle_events=[]


def finish_round(world,policies):
    from spider_sim.ecology import GoalBaseline
    from spider_sim.feeding import FeedingPolicy,INITIAL_HUNGER
    from spider_sim.pursuit import PursuitPolicy
    spider_source=next(iter(world.spiders))
    # Keep edible meals. Remove exhausted bodies from the observer map after
    # five rounds, retaining identity/death records for stable contact IDs.
    decay=getattr(world,'carcass_decay_rounds',5)
    if decay is not None:
        for cid in world.positions:
            if world.config.characters[cid].character.current_hp>0:continue
            if world.food.get(cid,{}).get('remaining',0)>0:continue
            since=world.exhausted_since.setdefault(cid,world.elapsed_seconds)
            if world.elapsed_seconds-since>=decay*6:world.retired_carcasses.add(cid)
    # Damage after an opportunity to feed avoids killing a spider just before a
    # meal would have reduced its hunger. A fed spider resets its starvation clock.
    for cid in list(world.spiders):
        if cid not in world.active:continue
        if world.hunger[cid]<1-1e-9:world.starvation_ticks[cid]=0;continue
        world.starvation_ticks[cid]=world.starvation_ticks.get(cid,0)+1
        if world.starvation_ticks[cid]%getattr(world,'starvation_interval',STARVATION_INTERVAL):continue
        character=world.config.characters[cid].character
        character.current_hp=max(0,character.current_hp-STARVATION_DAMAGE)
        died=character.current_hp==0
        if died:world.active.discard(cid)
        world.lifecycle_events.append({'event':'starvation_death' if died else 'starvation_damage','actor':cid,'damage_hp':STARVATION_DAMAGE})
    for parent in sorted(cid for cid in world.positions if cid.startswith(BEAST) and cid not in world.replaced_beasts and world.config.characters[cid].character.current_hp<=0):
        cid=new_id(world,BEAST+'_')
        position=world.random_floor_position(world.bodies[BEAST])
        clone_actor(world,BEAST,cid,'Shadowbeast '+cid.rsplit('_',1)[1],position)
        policies[cid]=GoalBaseline(seed=world.seed+len(world.contact_ids));world.replaced_beasts.add(parent)
        world.lifecycle_events.append({'event':'shadowbeast_replaced','actor':cid,'replaces':parent})
    for sac in world.egg_sacs.values():
        if sac['hatched'] or world.elapsed_seconds+6<sac['hatch_at']:continue
        first=nearby_position(world,sac['position'],world.bodies[spider_source])
        second=nearby_position(world,sac['position'],world.bodies[spider_source],reserved=[first]) if first else None
        if first is None or second is None:continue
        positions=[]
        for sex,pos in zip(('male','female'),(first,second)):
            cid=new_id(world,'freight_spider_'+sex+'_')
            clone_actor(world,spider_source,cid,('Mr' if sex=='male' else 'Ms')+' Spider '+cid.rsplit('_',1)[1],pos)
            world.spiders[cid]=sex;world.hunger[cid]=INITIAL_HUNGER
            parent_policy=policies[spider_source]
            if hasattr(parent_policy,'hunter'):
                template=parent_policy.hunter
                hunter=PursuitPolicy(template.model,approach=template.approach,facing=template.facing,
                               rest_without_goal=True,local_motion=True,seed=world.seed+len(world.contact_ids))
                policies[cid]=parent_policy.spawn(hunter,sex) if hasattr(parent_policy,'spawn') else FeedingPolicy(hunter)
            else:policies[cid]=GoalBaseline(seed=world.seed+len(world.contact_ids))
            positions.append(cid)
        sac['hatched']=True;sac['children']=positions
        world.config.items[sac['id']].state['egg_sac']='hatched'
        world.config.items[sac['id']].description='Empty, hatched silken egg sac.'
        world.lifecycle_events.append({'event':'egg_sac_hatched','actor':positions[0],'sac':sac['id'],'children':positions})
