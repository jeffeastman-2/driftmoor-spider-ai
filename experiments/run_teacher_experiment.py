"""Retrain unchanged networks on the improved teacher and compare encounter behavior."""
import argparse
import json
from pathlib import Path
import torch

from experiments.game_runtime import game_root
from experiments.run_recurrent_agents import batch,train
from experiments.recurrent_agents import save

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driftmoor-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--epochs",type=int,default=160)
    parser.add_argument("--train-episodes",type=int,default=160)
    parser.add_argument("--evaluation-seeds",type=int,default=8)
    args=parser.parse_args()
    if not 1<=args.train_episodes<=1000 or not 1<=args.evaluation_seeds<=1000 or args.epochs<1:
        parser.error("Use 1..1000 episodes/seeds and positive epochs")
    game_root(args.driftmoor_root)
    from spider_sim.dual_encounter import collect,ACTORS,ACTIONS
    from spider_sim.encounter_evaluation import evaluate,SCENARIOS
    output=args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Use an empty output directory to preserve earlier runs")
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    print("Collecting improved-teacher demonstrations...",flush=True)
    scenarios=("mixed","ground","ceiling","wounded_spider","wounded_gil","critical_spider","critical_gil")
    splits={"train":collect(args.train_episodes,110000,scenarios),
            "validation":collect(24,111000,scenarios),"test":collect(32,112000,scenarios)}
    for name,episodes in splits.items():
        (output/(name+".json")).write_text(json.dumps(episodes))
    report={"method":"distinct retreat/flee teachers; unchanged neural architectures",
            "teacher_version":"turn-and-move-v4","collection_scenarios":list(scenarios),
            "observation_schema":"body-center-motion-v4","epochs":args.epochs,
            "architecture":{"encoder":[41,32],"gru_hidden":48,"actions":8,"action_names":list(ACTIONS)},
            "splits":{key:[e["episode_id"] for e in episodes] for key,episodes in splits.items()},
            "agents":{}}
    learned={}
    for kind in ("mlp","gru"):
        learned[kind]={}
        report["agents"][kind]={}
        for cid in ACTORS:
            agent,metrics=train(kind,cid,*(batch(splits[s],cid) for s in ("train","validation","test")),args.epochs)
            learned[kind][cid]=agent
            report["agents"][kind][cid]=metrics
            save(agent,output/(kind+"-"+cid+".pt"),observation_schema="body-center-motion-v4")
            print(kind,cid,json.dumps(metrics),flush=True)
    print("Evaluating matched encounters and fixed teacher opponents...",flush=True)
    report["evaluation"]={"seeds":list(range(114000,114000+args.evaluation_seeds)),
                          "scenarios":list(SCENARIOS),"rounds":18}
    comparisons,traces=evaluate(learned,report["evaluation"]["seeds"])
    report["encounters"]=comparisons
    report["limitations"]=[
        "Synthetic body-center vision, coarse vibration, simplified 2.5D scene and movement",
        "Legacy and improved teachers use the same revised sensory observations",
        "Actor loadouts are fixed; Gil has a longsword and no spells",
        "Identical initial seeds, but different action choices consume different combat dice",
        "Survival includes actors starting at low health; no encounter completion reward",
        "Learned policies imitate rules; prediction heads are not used for planning",
        "Retreat and flee include orientation and translation in one movement slot",
        "Neural bearings come from current private measurements or the learned belief head",
        "Flee follows a remembered away heading; no route or obstacle planning",
        "Rear attacks use ordinary combat dice; no implemented backstab or opportunity-attack bonus",
        "Single training run and small held-out fixtures; no claim of statistical superiority",
    ]
    for name,episodes in traces.items():
        (output/(name+"-encounters.json")).write_text(json.dumps(episodes))
    (output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({name:value["total"] for name,value in comparisons.items()},indent=2))

if __name__=="__main__":
    main()
