"""Iterative simulation experience -> frozen-representation predictor updates."""
import argparse
import copy
import json
from pathlib import Path
import torch
from experiments.game_runtime import game_root
from experiments.recurrent_agents import load,save
from experiments.jepa_planner import Planner
from experiments.combat_curriculum import collect_mixture
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
    if any(n<1 for n in (args.rounds,args.episodes,args.epochs)) or args.rounds>10 or args.episodes>128:
        parser.error("Use positive counts, at most 10 rounds and 128 ordinary episodes")
    if args.output.exists():
        parser.error("Use a new output directory")
    game_root(args.driftmoor_root)
    from spider_sim.dual_encounter import ACTORS
    from spider_sim.encounter_evaluation import TeacherAgent,SCENARIOS,encounter,summarize
    torch.set_num_threads(1)
    args.output.mkdir(parents=True)
    report={"method":"Focused engine curriculum mixed with ordinary replay; frozen representation, policy, belief and costs",
        "exploration":.4,"rounds":args.rounds,"episodes_per_role_per_round":args.episodes,
        "epochs_per_round":args.epochs,"roles":{},"evaluation":{}}
    priors={cid:load(args.checkpoints/("gru-"+cid+".pt")) for cid in ACTORS}
    initial={cid:residual_predictor(priors[cid]) for cid in ACTORS}
    trained={}
    for role in ACTORS:
        agent=copy.deepcopy(initial[role]); replay=[]
        # Fixed, separately seeded experience sets never used as optimizer batches.
        validation=collect_mixture(priors[role],role,410000,ordinary=16,focused=64)
        test=collect_mixture(priors[role],role,420000,ordinary=32,focused=128)
        test_data=replay_tensors(agent,test)
        result={"prior_test":forecast_metrics(priors[role],test_data),
            "residual_initial_test":forecast_metrics(initial[role],test_data),"updates":[]}
        for iteration in range(args.rounds):
            start=400000+iteration*1000
            experience=collect_mixture(agent,role,start,ordinary=args.episodes,focused=2*args.episodes)
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
        result["test_by_source"]={}
        for source in ("ordinary","focused"):
            subset=replay_tensors(agent,[episode for episode in test if episode["source"]==source])
            result["test_by_source"][source]={name:forecast_metrics(model,subset)
                for name,model in (("prior",priors[role]),("residual_initial",initial[role]),("learned",agent))}
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
            episodes=[encounter(seed,scenario,pair) for scenario in SCENARIOS for seed in range(430000,430008)]
            key=role+"-"+name
            report["evaluation"][key]=summarize(episodes)
            print(key,json.dumps(report["evaluation"][key][role]),flush=True)
            if role=="giant_spider" and name=="experience_planner":
                report["spider_attack_diagnostics"]=[
                    {"seed":episode["seed"],"scenario":episode["scenario"],
                     "status":turn["result"]["status"],"contact":turn["observation"]["contact"],
                     "predicted_admission":turn["plan"]["predictions"][7][41],
                     "predicted_damage":turn["plan"]["predictions"][7][42]*20}
                    for episode in episodes for turn in episode["turns"]
                    if turn["actor"]==role and turn["result"]["action"]=="attack"]
    report["seeds"]={"training_starts":[400000+i*1000 for i in range(args.rounds)],
        "validation":[410000,410015],"test":[420000,420031],"evaluation":[430000,430007]}
    from experiments.audit_jepa_planning import fixture,analyze,CASES
    report["ranking_audit"]={}
    for role in ACTORS:
        comparisons={}
        for name,model in (("prior",priors[role]),("residual_initial",initial[role]),("learned",trained[role])):
            rows=[]
            for index,case in enumerate(CASES):
                world=fixture(role,case,470000+index);world.audit_case=case
                rows.append(analyze(world,role,model,8,dice_start=480000))
            comparisons[name]={"mean_reference_regret":sum(row["forecast_ranking_regret"] for row in rows)/len(rows),
                "matching_choices":sum(row["learned_cost_action"]==row["reference_cost_action"] for row in rows),"cases":rows}
        report["ranking_audit"][role]=comparisons
    report["curriculum"]={"cases":["reachable","out_of_reach","sideways","ceiling","wounded","critical","boundary","occluded"],
        "focused_per_training_round":2*args.episodes,"focused_validation_seeds":[410256,410319],
        "focused_test_seeds":[420256,420383],"focused_training_offset":256,
        "first_action":"seed modulo 8; balanced within each case", "focused_turns":4,
        "ranking_fixture_seeds":[470000,470005],"ranking_dice_seeds":[480000,480007]}
    report["limitations"]=["Single training initialization; development experiment",
        "Mixed focused coverage and ordinary exploration with fixed teacher opponent; not self-play",
        "Predictor HP output reparameterized to change; only that output row starts at zero",
        "Residual-initial control separates that structural change from experience learning",
        "Opponent response and combat dice remain uncertain; no guarantee of performance gain",
        "Terminal transitions retained; new episodes reset GRU state, not learned weights"]
    (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")

if __name__=="__main__":
    main()
