"""Fixed-corpus encoder/GRU capacity comparison with two optimizer seeds."""
import argparse
import json
from pathlib import Path
import torch
from experiments.game_runtime import game_root
from experiments.recurrent_agents import load,save
from experiments.capacity_learning import VARIANTS,train_variant,evaluate_forecasts
from experiments.combat_curriculum import collect_mixture
from experiments.jepa_planner import Planner


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driftmoor-root",type=Path,required=True)
    parser.add_argument("--checkpoints",type=Path,required=True)
    parser.add_argument("--corpus",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--epochs",type=int,default=60)
    args=parser.parse_args()
    if args.output.exists() or not 1<=args.epochs<=500:
        parser.error("Use a new output directory and 1..500 epochs")
    game_root(args.driftmoor_root)
    from spider_sim.dual_encounter import ACTORS
    from spider_sim.encounter_evaluation import TeacherAgent,SCENARIOS,encounter,summarize
    from experiments.audit_jepa_planning import fixture,analyze,CASES
    torch.set_num_threads(1);args.output.mkdir(parents=True)
    report={"method":"Matched complete-sequence gradient training; fixed corpus, costs, policy/belief head weights",
        "epochs":args.epochs,"optimizer_seeds":[42,43],"roles":{},
        "seeds":{"forecast_test_ordinary":[510000,510031],"forecast_test_focused":[510256,510383],
                 "encounters":[520000,520003],"ranking_fixture":[530000,530005],"ranking_dice":[540000,540003]}}
    for role in ACTORS:
        prior=load(args.checkpoints/("gru-"+role+".pt"))
        training=json.loads((args.corpus/(role+"-replay.json")).read_text())
        validation=json.loads((args.corpus/(role+"-validation.json")).read_text())
        test=collect_mixture(prior,role,510000,ordinary=32,focused=128)
        (args.output/(role+"-test.json")).write_text(json.dumps(test))
        entries={}
        for seed in (42,43):
            for variant in VARIANTS:
                print("Training",role,seed,variant,flush=True)
                agent,metrics=train_variant(prior,variant,training,validation,args.epochs,seed)
                metrics["forecast"]=evaluate_forecasts(agent,prior,test)
                episodes=[encounter(n,scenario,{cid:(Planner(agent) if cid==role else TeacherAgent()) for cid in ACTORS})
                          for scenario in SCENARIOS for n in range(520000,520004)]
                metrics["encounters"]=summarize(episodes)
                metrics["attack_rejections"]=[{"status":turn["result"]["status"],"contact":turn["observation"]["contact"]}
                    for episode in episodes for turn in episode["turns"]
                    if turn["actor"]==role and turn["result"]["action"]=="attack" and not turn["result"]["admitted"]]
                audit=[]
                for index,case in enumerate(CASES):
                    world=fixture(role,case,530000+index);world.audit_case=case
                    audit.append(analyze(world,role,agent,4,dice_start=540000))
                metrics["ranking"]={"matching_choices":sum(r["learned_cost_action"]==r["reference_cost_action"] for r in audit),
                    "mean_reference_regret":sum(r["forecast_ranking_regret"] for r in audit)/len(audit),"cases":audit}
                key=variant+"-"+str(seed);entries[key]=metrics
                save(agent,args.output/(role+"-"+key+".pt"))
                print(role,key,json.dumps({"forecast":metrics["forecast"],"encounters":metrics["encounters"][role]}),flush=True)
                report["roles"][role]=entries
                (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    report["limitations"]=["Two optimizer seeds; selected cavern fixtures, not world generalization",
        "Fixed policy/belief heads can change outputs when their representation changes",
        "Forecast-only objective; no policy or belief preservation loss",
        "Only encoder depth differs between joint variants; predictor architecture fixed",
        "Four dice streams per ranking fixture: diagnostic only",
        "Training/validation corpus reused; fresh forecast and encounter evaluation seeds",
        "Predictor-only baseline uses same sequential loss/batch budget as joint variants, not the historical sampler"]
    (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")

if __name__=="__main__":
    main()
