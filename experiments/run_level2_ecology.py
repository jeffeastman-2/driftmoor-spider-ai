"""Run a deterministic multi-resident Level-2 baseline without an LLM."""
import argparse
import json
from pathlib import Path
from experiments.game_runtime import game_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driftmoor-root",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--rounds",type=int,default=30)
    args = parser.parse_args()
    if args.rounds<1 or args.rounds>1000:
        parser.error("rounds must be between 1 and 1000")
    if args.output.exists():
        parser.error("Use a new output path")
    game_root(args.driftmoor_root)
    from spider_sim.ecology import run_ecology,SPIDER
    report = run_ecology(args.seed,args.rounds)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    rows = [r for r in report["records"] if r["actor"]==SPIDER]
    print(json.dumps({"scene":report["scene"],"rounds":args.rounds,
          "spider_attacks":sum(r["command"]["action"]=="attack" for r in rows),
          "admitted":sum(r["result"]["admitted"] for r in rows),
          "damage":sum(r["result"]["damage"] for r in rows),"final_hp":report["final_hp"]},indent=2))
    print("Saved",args.output)

if __name__=="__main__":main()
