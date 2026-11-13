"""Separate, versioned experience corpus for the shorter advance action."""
from experiments.approach_action import ApproachEncounter,MOTOR_SCHEMA
from experiments.combat_learning import collect_experience
from experiments.combat_curriculum import curriculum_world,CASES


def normal_world(role,seed,scenario):
    world=ApproachEncounter(seed,scenario=scenario)
    world.approach_actor=role
    return world


def collect_approach(agent,role,start,ordinary=64,focused=128):
    from spider_sim.encounter_evaluation import SCENARIOS
    normal=collect_experience(agent,role,range(start,start+ordinary),.4,
        world_factory=lambda actor,seed:normal_world(actor,seed,SCENARIOS[(seed-start)%len(SCENARIOS)]))
    focus=collect_experience(agent,role,range(start+256,start+256+focused),.4,rounds=4,
        world_factory=lambda actor,seed:curriculum_world(actor,seed,ApproachEncounter),first_action=lambda seed:seed%8)
    for source,episodes in (("ordinary",normal),("focused",focus)):
        for episode in episodes:
            episode["motor_schema"]=MOTOR_SCHEMA
            episode["source"]=source
    for episode in focus:
        episode["curriculum_case"]=CASES[(episode["seed"]//8)%len(CASES)]
    return normal+focus
