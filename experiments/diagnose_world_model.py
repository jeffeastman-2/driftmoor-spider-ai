"""Diagnose the frozen v1 policy using validation episodes only."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driftmoor-root",type=Path,required=True)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args = parser.parse_args()
    root = args.driftmoor_root.expanduser().resolve()
    run = args.run_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    code = (
        "import importlib.util,json; "
        f"s=importlib.util.spec_from_file_location('spider_predictor',{str(HERE/'world_model.py')!r}); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        f"p=m.load_predictor({str(run/'predictor.pt')!r}); "
        "from engine.spider_diagnostics import diagnose; "
        "print(json.dumps(diagnose(p)))"
    )
    print("Forking validation states to compare candidate actions...",flush=True)
    result = subprocess.run([sys.executable,"-c",code],cwd=root,text=True,capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr)
    report = json.loads(result.stdout)
    training = [json.loads(line) for line in (run/"train.jsonl").read_text().splitlines()]
    validation = [json.loads(line) for line in (run/"validation.jsonl").read_text().splitlines()]
    from collections import Counter
    report["summary"]["training_counts"] = {
        "transitions":len(training),
        "actions":dict(Counter(r["action"]["type"] for r in training)),
        "admitted_attacks":sum(r["target"][-1] for r in training),
    }
    import torch
    from world_model import load_predictor
    predictor = load_predictor(run/"predictor.pt")
    attack_rows = [r for r in validation if r["action"]["type"]=="attack"]
    prediction = predictor([r["input"] for r in attack_rows])
    report["summary"]["recorded_validation_attacks"] = {
        "count":len(attack_rows),
        "admitted":sum(r["target"][-1] for r in attack_rows),
        "brier":sum((p[-1]-r["target"][-1])**2 for p,r in zip(prediction,attack_rows))/max(1,len(attack_rows)),
    }
    # Compare nearest training input for the SAME action. This measures
    # coverage, not proof of a causal distribution-shift explanation.
    distances = {}
    for name,rows in (
        ("recorded_validation",validation),
        ("closed_loop_validation",[
            {"input":r["candidates"][r["selected"]]["input"],
             "action":r["candidates"][r["selected"]]["action"]}
            for r in report["traces"]
        ]),
    ):
        by_action = {}
        for kind in ("wait","orient","move","attack"):
            source = [r["input"] for r in training if r["action"]["type"]==kind]
            query = [r["input"] for r in rows if r["action"]["type"]==kind]
            if source and query:
                near = torch.cdist(torch.tensor(query),torch.tensor(source)).min(dim=1).values
                by_action[kind] = {"count":len(query),"mean_distance":float(near.mean()),
                                   "max_distance":float(near.max())}
        distances[name] = by_action
    report["summary"]["same_action_training_distances"] = distances
    import hashlib
    report["summary"]["frozen_checkpoint_sha256"] = hashlib.sha256(
        (run/"predictor.pt").read_bytes()
    ).hexdigest()
    output.mkdir(parents=True,exist_ok=True)
    (output/"diagnostics.json").write_text(json.dumps(report,indent=2)+"\n")
    (output/"summary.json").write_text(json.dumps(report["summary"],indent=2)+"\n")
    print(json.dumps(report["summary"],indent=2))


if __name__ == "__main__":
    main()
