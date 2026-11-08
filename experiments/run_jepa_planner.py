"""Compare one-step cost planning with policy argmax on fixed teacher opponents."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import torch
from experiments.game_runtime import game_root
from experiments.recurrent_agents import load
from experiments.jepa_planner import Planner,CostWeights


def forecast_errors(episodes,role):
    """Audit selected forecasts against the next real private observation.

    Terminal turns have no subsequent decision observation and are omitted.
    This measures visited actions, not unexecuted counterfactuals.
    """
    errors=[]
    for episode in episodes:
        turns=[turn for turn in episode["turns"] if turn["actor"]==role]
        for turn,next_turn in zip(turns,turns[1:]):
            plan=turn.get("plan")
            if plan is None:
                continue
            pred=plan["predictions"][plan["selected"]]
            actual=next_turn["observation"]["values"]
            errors.append({"health_fraction_absolute_error":abs(pred[16]-actual[16]),
                "observation_mse":sum((a-b)**2 for a,b in zip(pred[:41],actual))/41,
                "admission_absolute_error":abs(pred[41]-float(turn["result"]["admitted"]))})
    return {"transitions":len(errors),**{key:sum(e[key] for e in errors)/len(errors)
        for key in errors[0]} } if errors else {"transitions":0}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driftmoor-root",type=Path,required=True)
    parser.add_argument("--checkpoints",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seeds",type=int,default=8)
    args=parser.parse_args()
    if not 1<=args.seeds<=1000:
        parser.error("Use 1..1000 seeds")
    if args.output.exists():
        parser.error("Output must not exist; preserve previous results")
    game_root(args.driftmoor_root)
    from spider_sim.dual_encounter import ACTORS
    from spider_sim.encounter_evaluation import TeacherAgent,SCENARIOS,encounter,summarize
    torch.set_num_threads(1)
    agents={cid:load(args.checkpoints/("gru-"+cid+".pt")) for cid in ACTORS}
    reports={}
    # Unused evaluation seeds; no cost tuning on these outcomes.
    for role in ACTORS:
        for mode in ("policy","planner"):
            pair={cid:TeacherAgent() for cid in ACTORS}
            pair[role]=Planner(agents[role]) if mode=="planner" else agents[role]
            episodes=[encounter(seed,scenario,pair) for scenario in SCENARIOS
                      for seed in range(135000,135000+args.seeds)]
            reports[mode+"-"+role]={"forecast_errors":forecast_errors(episodes,role),"total":summarize(episodes),
                "scenarios":{s:summarize([e for e in episodes if e["scenario"]==s]) for s in SCENARIOS}}
            print(mode,role,json.dumps(reports[mode+"-"+role]["total"][role]),flush=True)
    report={"method":"One-step exhaustive action ranking; same frozen GRU checkpoints",
            "weights":asdict(CostWeights()),
            "condition_goals":{"source":"private own deviations from authored baseline",
                "cortisol":"injury multiplier 1+.5*c; separation preference shift .5*c",
                "dopamine":"attack reward multiplier 1+.5*d",
                "other_dimensions":"available, inactive; absent dimensions ignored",
                "updates":"deterministic Driftmoor combat events; no LLM calls"},"seeds":list(range(135000,135000+args.seeds)),
            "scenarios":list(SCENARIOS),"encounters":reports,
            "limitations":["Hand-authored costs; uncalibrated learned forecasts",
                "No multi-step beam, opponent branching, reward training, or oracle lookahead",
                "Predictor learned teacher responses; counterfactual actions may be inaccurate",
                "Identical starts do not imply identical combat dice after action divergence"]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")

if __name__=="__main__":
    main()
