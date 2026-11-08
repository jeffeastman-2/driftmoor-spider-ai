"""Focused engine experience mixed with ordinary private-sensory rollouts."""
import random
from experiments.combat_learning import collect_experience

CASES=("reachable","out_of_reach","sideways","ceiling","wounded",
       "critical","boundary","occluded")


def curriculum_world(role,seed):
    from spider_sim.dual_encounter import DualEncounter,ACTORS,SPIDER,PLAYER
    from models.body_geometry import Orientation
    from models.space_time import Position
    rng=random.Random(seed)
    case=CASES[(seed//8)%len(CASES)]
    world=DualEncounter(seed,outages=False,scenario="wounded_spider")
    x,y=70+rng.uniform(-1,1),195+rng.uniform(-1,1)
    distance=rng.uniform(3,6) if case not in ("out_of_reach","occluded") else rng.uniform(10,14)
    world.positions[SPIDER]=(x,y,"Secret Chamber")
    world.positions[PLAYER]=(x+distance,y,"Secret Chamber")
    world.heights[SPIDER]=10 if case=="ceiling" else 0
    world.orientations[SPIDER]=Orientation(roll=180 if case=="ceiling" else 0,
        heading=rng.uniform(-10,10)+(90 if case=="sideways" else 0))
    world.orientations[PLAYER]=Orientation(heading=180+rng.uniform(-10,10))
    world.concealed=case in ("ceiling","occluded")
    if case=="boundary":
        world.positions[SPIDER]=(61,195,"Secret Chamber")
        world.positions[PLAYER]=(67,195,"Secret Chamber")
    if case=="occluded":
        world.positions[PLAYER]=(90,195,"")
    for actor in ACTORS:
        px,py,_=world.positions[actor]
        world.positions[actor]=(px,py,world.geometry.zone_at(Position(px,py)) or "")
        hp_fraction=(.1 if case=="critical" else .35) if actor==role and case in ("critical","wounded") else 1.
        world.config.characters[actor].character.current_hp=max(1,round(world.max_hp[actor]*hp_fraction))
    world.curriculum_case=case
    return world


def collect_mixture(agent,role,start,ordinary=64,focused=128):
    normal=collect_experience(agent,role,range(start,start+ordinary),.4)
    focus=collect_experience(agent,role,range(start+256,start+256+focused),.4,rounds=4,
                             world_factory=curriculum_world,first_action=lambda seed:seed%8)
    for episode in normal:
        episode["source"]="ordinary"
    for episode in focus:
        episode["source"]="focused"
        episode["curriculum_case"]=CASES[(episode["seed"]//8)%len(CASES)]
    return normal+focus
