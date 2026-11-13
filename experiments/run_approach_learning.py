"""Iterative simulation experience -> frozen-representation predictor updates."""
import argparse
import copy
import json
from pathlib import Path
import torch
from experiments.game_runtime import game_root
from experiments.recurrent_agents import load,save
from experiments.jepa_planner import Planner
from experiments.approach_learning import collect_approach as collect_mixture,normal_world
from experiments.approach_action import ApproachEncounter,MOTOR_SCHEMA
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
    report={"method":"Fine-tune transferred predictor using only versioned approach-action experience; frozen encoder, GRU, policy and costs",
        "motor_schema":MOTOR_SCHEMA,
        "exploration":.4,"rounds":args.rounds,"episodes_per_role_per_round":args.episodes,
        "epochs_per_round":args.epochs,"roles":{},"evaluation":{}}
    priors={cid:load(args.checkpoints/("gru-"+cid+".pt")) for cid in ACTORS}
    initial={cid:residual_predictor(priors[cid]) for cid in ACTORS}
    trained={}
    for role in ACTORS:
        agent=copy.deepcopy(initial[role]); replay=[]
        # Fixed, separately seeded experience sets never used as optimizer batches.
        validation=collect_mixture(priors[role],role,710000,ordinary=16,focused=64)
        test=collect_mixture(priors[role],role,720000,ordinary=32,focused=128)
        test_data=replay_tensors(agent,test)
        result={"prior_test":forecast_metrics(priors[role],test_data),
"updates":[]}
        for iteration in range(args.rounds):
            start=700000+iteration*1000
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
                for name,model in (("prior",priors[role]),("learned",agent))}
        report["roles"][role]=result
        (args.output/(role+"-replay.json")).write_text(json.dumps(replay))
        (args.output/(role+"-validation.json")).write_text(json.dumps(validation))
        (args.output/(role+"-test.json")).write_text(json.dumps(test))
        save(agent,args.output/("gru-"+role+".pt"),observation_schema=MOTOR_SCHEMA)
    # Fresh encounter seeds, distinct from replay, validation, test and earlier runs.
    for role in ACTORS:
        variants={"policy":priors[role],"prior_planner":Planner(priors[role]),
                  "experience_planner":Planner(trained[role])}
        for name,controller in variants.items():
            pair={cid:TeacherAgent() for cid in ACTORS};pair[role]=controller
            episodes=[encounter(seed,scenario,pair,world_factory=lambda n,s:normal_world(role,n,s)) for scenario in SCENARIOS for seed in range(730000,730008)]
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
    report["seeds"]={"training_starts":[700000+i*1000 for i in range(args.rounds)],
        "validation":[710000,710015],"test":[720000,720031],"evaluation":[730000,730007]}
    from experiments.audit_jepa_planning import fixture,analyze,CASES
    report["ranking_audit"]={}
    for role in ACTORS:
        comparisons={}
        for name,model in (("prior",priors[role]),("learned",trained[role])):
            rows=[]
            for index,case in enumerate(CASES):
                world=fixture(role,case,770000+index,ApproachEncounter);world.audit_case=case
                rows.append(analyze(world,role,model,8,dice_start=780000))
            comparisons[name]={"mean_reference_regret":sum(row["forecast_ranking_regret"] for row in rows)/len(rows),
                "matching_choices":sum(row["learned_cost_action"]==row["reference_cost_action"] for row in rows),"cases":rows}
        report["ranking_audit"][role]=comparisons
    report["curriculum"]={"cases":["reachable","out_of_reach","sideways","ceiling","wounded","critical","boundary","occluded"],
        "focused_per_training_round":2*args.episodes,"focused_validation_seeds":[710256,710319],
        "focused_test_seeds":[720256,720383],"focused_training_offset":256,
        "first_action":"seed modulo 8; balanced within each case", "focused_turns":4,
        "ranking_fixture_seeds":[770000,770005],"ranking_dice_seeds":[780000,780007]}
    from experiments.engine_cost_control import run_episode
    report["controlled_encounters"]={}
    for role in ACTORS:
        role_results={}
        for name,model in (("transferred",priors[role]),("learned",trained[role])):
            episodes=[run_episode(model,role,case,seed,"learned",rounds=6,world_class=ApproachEncounter)
                      for case in CASES for seed in range(740000,740003)]
            role_results[name]={case:{metric:sum(e["stats"][metric] for e in episodes if e["case"]==case)
                for metric in ("attacks","admitted","own_damage_received","survived") } for case in CASES}
        reference=[run_episode(trained[role],role,"healthy_far_idle",seed,"engine_prior",draws=8,world_class=ApproachEncounter)
                   for seed in range(740000,740003)]
        role_results["far_engine_reference"]={metric:sum(e["stats"][metric] for e in reference)
            for metric in ("attacks","admitted","own_damage_received","survived")}
        report["controlled_encounters"][role]=role_results
    report["limitations"]=["Single training initialization; development experiment",
        "Mixed focused coverage and ordinary exploration with fixed teacher opponent; not self-play",
        "HP-change predictor transferred from curriculum-learning-v1; no reinitialization",
        "Transferred forecasts are intentionally stale for shortened advance; no old-motor replay mixed in",
        "Opponent response and combat dice remain uncertain; no guarantee of performance gain",
        "Terminal transitions retained; new episodes reset GRU state, not learned weights"]
    (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")

if __name__=="__main__":
    main()
