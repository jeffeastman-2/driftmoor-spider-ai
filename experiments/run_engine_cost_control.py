"""Run current costs with sampled true engine futures in controlled encounters."""
import argparse
from collections import Counter
import json
from pathlib import Path
import torch
from experiments.game_runtime import game_root
from experiments.recurrent_agents import load
from experiments.engine_cost_control import run_episode


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driftmoor-root",type=Path,required=True)
    parser.add_argument("--checkpoints",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--draws",type=int,default=8)
    args=parser.parse_args()
    if args.output.exists() or not 1<=args.draws<=64:
        parser.error("Use a new output directory and 1..64 draws")
    game_root(args.driftmoor_root)
    from experiments.audit_jepa_planning import CASES
    from spider_sim.dual_encounter import ACTORS
    torch.set_num_threads(1);args.output.mkdir(parents=True)
    report={"method":"Closed-loop controlled encounters; same costs, sampled true engine outcomes",
        "draws_per_action":args.draws,"rounds":6,"seeds":[610000,610001,610002],"groups":{}}
    for role in ACTORS:
        agent=load(args.checkpoints/("gru-"+role+".pt"))
        for controller in ("policy","learned","engine_prior","engine_cost_only"):
            episodes=[run_episode(agent,role,case,seed,controller,args.draws)
                      for case in CASES for seed in range(610000,610003)]
            result={}
            for case in CASES:
                rows=[e for e in episodes if e["case"]==case]
                result[case]={key:sum(e["stats"][key] for e in rows)
                    for key in ("attacks","admitted","rejected_attacks","own_damage_received",
                                "opponent_hp_lost","survived","opponent_defeated")}
                result[case]["first_actions"]=dict(Counter(e["stats"]["first_action"] for e in rows))
            key=role+"-"+controller;report["groups"][key]=result
            (args.output/(key+"-episodes.json")).write_text(json.dumps(episodes))
            print(key,json.dumps(result),flush=True)
            (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    report["limitations"]=["Privileged offline evaluator only; no deployed agent receives true futures",
        "Sampled costs, not exact expectations; independent branch dice do not peek at actual future RNG",
        "Same candidate action/aiming decoder and private neural belief across all controllers",
        "Preserves live opponent memory in candidate forks; fixed teacher or idle response as specified",
        "Six own turns, one-step horizon; a low cost need not mean successful hunting",
        "Three actual dice seeds share authored poses; not three distinct landscapes",
        "No cost tuning, network training, or controller promotion"]
    (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")

if __name__=="__main__":
    main()
