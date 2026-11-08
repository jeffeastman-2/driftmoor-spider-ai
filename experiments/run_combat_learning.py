"""Iterative simulation experience -> frozen-representation predictor updates."""
import argparse
import copy
import json
from pathlib import Path
import torch
from experiments.game_runtime import game_root
from experiments.recurrent_agents import load,save
from experiments.jepa_planner import Planner
from experiments.combat_learning import residual_predictor,collect_experience,replay_tensors,fit_predictor,forecast_metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driftmoor-root",type=Path,required=True)
    parser.add_argument("--checkpoints",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--rounds",type=int,default=2)
    parser.add_argument("--episodes",type=int,default=64)
    parser.add_argument("--epochs",type=int,default=80)
    args=parser.parse_args()
    if any(n<1 for n in (args.rounds,args.episodes,args.epochs)) or args.rounds>10 or args.episodes>1000:
        parser.error("Use positive counts, at most 10 rounds and 1000 episodes")
    if args.output.exists():
        parser.error("Use a new output directory")
    game_root(args.driftmoor_root)
    from spider_sim.dual_encounter import ACTORS
    from spider_sim.encounter_evaluation import TeacherAgent,SCENARIOS,encounter,summarize
    torch.set_num_threads(1)
    args.output.mkdir(parents=True)
    report={"method":"Frozen representation, policy, belief and costs; predictor-only experience replay",
        "exploration":.4,"rounds":args.rounds,"episodes_per_role_per_round":args.episodes,
        "epochs_per_round":args.epochs,"roles":{},"evaluation":{}}
    priors={cid:load(args.checkpoints/("gru-"+cid+".pt")) for cid in ACTORS}
    initial={cid:residual_predictor(priors[cid]) for cid in ACTORS}
    trained={}
    for role in ACTORS:
        agent=copy.deepcopy(initial[role]); replay=[]
        # Fixed, separately seeded experience sets never used as optimizer batches.
        validation=collect_experience(priors[role],role,range(310000,310024),.4)
        test=collect_experience(priors[role],role,range(320000,320032),.4)
        test_data=replay_tensors(agent,test)
        result={"prior_test":forecast_metrics(priors[role],test_data),
            "residual_initial_test":forecast_metrics(initial[role],test_data),"updates":[]}
        for iteration in range(args.rounds):
            start=300000+iteration*1000
            experience=collect_experience(agent,role,range(start,start+args.episodes),.4)
            replay.extend(experience)
            metrics=fit_predictor(agent,replay,validation,args.epochs,42+iteration)
            metrics["round"]=iteration+1
            metrics["test"]=forecast_metrics(agent,test_data)
            metrics["terminal_transitions"]=sum(row["terminal"] for e in replay for row in e["rows"])
            result["updates"].append(metrics)
            print(role,iteration+1,json.dumps(metrics),flush=True)
        # Verify the scope of training independently of optimizer configuration.
        for name,param in agent.state_dict().items():
            if not name.startswith("predictor."):
                assert torch.equal(param,priors[role].state_dict()[name]),name
        trained[role]=agent
        report["roles"][role]=result
        (args.output/(role+"-replay.json")).write_text(json.dumps(replay))
        (args.output/(role+"-validation.json")).write_text(json.dumps(validation))
        (args.output/(role+"-test.json")).write_text(json.dumps(test))
        save(agent,args.output/("gru-"+role+".pt"))
    # Fresh encounter seeds, distinct from replay, validation, test and earlier runs.
    for role in ACTORS:
        variants={"policy":priors[role],"prior_planner":Planner(priors[role]),
                  "residual_initial":Planner(initial[role]),"experience_planner":Planner(trained[role])}
        for name,controller in variants.items():
            pair={cid:TeacherAgent() for cid in ACTORS};pair[role]=controller
            episodes=[encounter(seed,scenario,pair) for scenario in SCENARIOS for seed in range(330000,330008)]
            key=role+"-"+name
            report["evaluation"][key]=summarize(episodes)
            print(key,json.dumps(report["evaluation"][key][role]),flush=True)
    report["seeds"]={"training_starts":[300000+i*1000 for i in range(args.rounds)],
        "validation":[310000,310023],"test":[320000,320031],"evaluation":[330000,330007]}
    report["limitations"]=["Single training initialization; development experiment",
        "Random exploration and fixed teacher opponent; not reward-based self-play",
        "Predictor HP output reparameterized to change; only that output row starts at zero",
        "Residual-initial control separates that structural change from experience learning",
        "Opponent response and combat dice remain uncertain; no guarantee of performance gain",
        "Terminal transitions retained; new episodes reset GRU state, not learned weights"]
    (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")

if __name__=="__main__":
    main()
