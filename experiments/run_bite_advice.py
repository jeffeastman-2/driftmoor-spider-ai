"""Paired learned-controller evaluation with/without current private bite advice."""
import argparse
import json
from pathlib import Path
import torch
from experiments.game_runtime import game_root
from experiments.recurrent_agents import load
from experiments.bite_advice import load_advisor,AdviceEncounter,AdvisedPlanner
from experiments.approach_action import MOTOR_SCHEMA
from experiments.jepa_planner import Planner,CostWeights


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driftmoor-root",type=Path,required=True)
    parser.add_argument("--agent",type=Path,required=True)
    parser.add_argument("--advisor",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error("Use a new output directory")
    game_root(args.driftmoor_root)
    from spider_sim.dual_encounter import SPIDER,PLAYER
    from spider_sim.encounter_evaluation import SCENARIOS,TeacherAgent,encounter,summarize
    from experiments.audit_jepa_planning import fixture,CASES
    from experiments.engine_cost_control import response
    torch.set_num_threads(1);args.output.mkdir(parents=True)
    agent=load(args.agent,expected_schema=MOTOR_SCHEMA)
    validation=json.loads((args.advisor/"validation.json").read_text())
    advisor=load_advisor(args.advisor,validation)
    report={"method":"Only replace candidate attack admission estimate; all damage/injury/cost/motor/model weights fixed",
        "calibration":{"slope":advisor.slope,"bias":advisor.bias,"split":"accurate-bite-v2 validation only"},
        "groups":{},"controlled":{},"seeds":{"encounters":[950000,950015],"controlled":[960000,960003]}}
    settings={"neutral":CostWeights(),"hunter":CostWeights(healthy_injury_multiplier=.25,healthy_attack_multiplier=4)}
    for name,weights in settings.items():
        for advised in (False,True):
            controller=AdvisedPlanner(agent,advisor,weights) if advised else Planner(agent,weights)
            def factory(seed,scenario):
                world=AdviceEncounter(seed,scenario=scenario);world.approach_actor=SPIDER;return world
            episodes=[encounter(seed,scenario,{SPIDER:controller,PLAYER:TeacherAgent()},world_factory=factory)
                      for scenario in SCENARIOS for seed in range(950000,950016)]
            key=name+("_advice" if advised else "_baseline")
            report["groups"][key]=summarize(episodes)
            (args.output/(key+"-episodes.json")).write_text(json.dumps(episodes))
            controlled={}
            for case in CASES:
                totals={"attacks":0,"admitted":0,"rejected":0,"own_damage":0,"survived":0,"first_actions":[]}
                for seed in range(960000,960004):
                    world=fixture(SPIDER,case,seed,AdviceEncounter);session=controller.session();teacher=TeacherAgent().session()
                    initial=world.config.characters[SPIDER].character.current_hp
                    for turn in range(6):
                        if any(world.config.characters[c].character.current_hp<=0 for c in (SPIDER,PLAYER)):break
                        world.begin_turn(SPIDER);obs=world.observation_for(SPIDER)
                        action,belief=session.choose_observation(SPIDER,obs)
                        if turn==0:totals["first_actions"].append(action)
                        _,result=response(world,SPIDER,action,belief,teacher,"teacher" if case.endswith("teacher") else "idle")
                        totals["attacks"]+=int(action==7);totals["admitted"]+=int(result["admitted"])
                        totals["rejected"]+=int(action==7 and not result["admitted"])
                    hp=world.config.characters[SPIDER].character.current_hp
                    totals["own_damage"]+=initial-hp;totals["survived"]+=int(hp>0)
                controlled[case]=totals
            report["controlled"][key]=controlled
            print(key,json.dumps(report["groups"][key][SPIDER]),flush=True)
            (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    report["limitations"]=["One agent checkpoint; three classifier seeds ensembled, validation-calibrated scores",
        "Neutral primary comparison; existing hunter profile secondary, unchanged within each pair",
        "Classifier receives only current private geometry transformed to the candidate bite pose",
        "No simulation forks, future outcomes, authoritative opponent state or computed legality during inference",
        "Damage expectation unchanged; advice is a score component, not a hard gate",
        "New seeds in familiar cavern/fixtures, not new-map generalization",
        "No training, default promotion or game-engine change"]
    (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")

if __name__=="__main__":main()
