"""Healthy-spider aggression sweep; wounded costs preserved, model weights fixed."""
import argparse
from dataclasses import asdict
from collections import Counter
import json
from pathlib import Path
import torch
from experiments.game_runtime import game_root
from experiments.recurrent_agents import load
from experiments.jepa_planner import Planner,CostWeights
from experiments.engine_cost_control import run_episode
from experiments.approach_action import ApproachEncounter,MOTOR_SCHEMA
from experiments.approach_learning import normal_world

SETTINGS={"baseline":CostWeights(),
    "bold":CostWeights(healthy_injury_multiplier=.5,healthy_attack_multiplier=2),
    "hunter":CostWeights(healthy_injury_multiplier=.25,healthy_attack_multiplier=4)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driftmoor-root",type=Path,required=True)
    parser.add_argument("--checkpoints",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    game_root(args.driftmoor_root)
    from spider_sim.dual_encounter import SPIDER,PLAYER
    from experiments.audit_jepa_planning import CASES
    from spider_sim.encounter_evaluation import TeacherAgent,SCENARIOS,encounter,summarize
    torch.set_num_threads(1);args.output.mkdir(parents=True)
    agent=load(args.checkpoints/"gru-giant_spider.pt",expected_schema=MOTOR_SCHEMA)
    report={"method":"Three predeclared healthy-only cost settings; frozen approach-trained spider",
        "settings":{name:asdict(weights) for name,weights in SETTINGS.items()},
        "controlled":{},"fresh_encounters":{},"seeds":{"control":[810000,810002],"fresh":[820000,820007]}}
    for name,weights in SETTINGS.items():
        report["controlled"][name]={}
        for controller in ("engine_prior","learned"):
            episodes=[run_episode(agent,SPIDER,case,seed,controller,draws=8,world_class=ApproachEncounter,weights=weights)
                      for case in CASES for seed in range(810000,810003)]
            result={}
            for case in CASES:
                rows=[e for e in episodes if e["case"]==case]
                result[case]={key:sum(e["stats"][key] for e in rows)
                    for key in ("attacks","admitted","rejected_attacks","own_damage_received","opponent_hp_lost","survived")}
                result[case]["first_actions"]=dict(Counter(e["stats"]["first_action"] for e in rows))
            report["controlled"][name][controller]=result
            (args.output/(name+"-"+controller+"-episodes.json")).write_text(json.dumps(episodes))
            print(name,controller,json.dumps(result),flush=True)
        episodes=[encounter(seed,scenario,{SPIDER:Planner(agent,weights),PLAYER:TeacherAgent()},
                  world_factory=lambda n,s:normal_world(SPIDER,n,s))
                  for scenario in SCENARIOS for seed in range(820000,820008)]
        report["fresh_encounters"][name]={"total":summarize(episodes),
            "scenarios":{s:summarize([e for e in episodes if e["scenario"]==s]) for s in SCENARIOS}}
        (args.output/(name+"-fresh-episodes.json")).write_text(json.dumps(episodes))
        print(name,"fresh",json.dumps(report["fresh_encounters"][name]["total"][SPIDER]),flush=True)
        (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    report["limitations"]=["Development sweep, not untouched final benchmark or statistical proof",
        "Only healthy injury/attack weights change; same model, costs below .35 HP, and movements",
        "Simulator reference uses privileged sampled futures for evaluation only",
        "Different actions consume different combat rolls; more legal bites may mean greater injury",
        "No model retraining or automatic default promotion"]
    (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")

if __name__=="__main__":
    main()
