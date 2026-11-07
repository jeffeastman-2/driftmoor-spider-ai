"""V2 fitting/validation and a separately invoked fresh test evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

if __package__:
    from .game_runtime import game_root, run_game
else:
    from game_runtime import game_root, run_game

import torch
from torch import nn

if __package__:
    from .world_model_v2 import Network, load_predictor
else:
    from world_model_v2 import Network, load_predictor

HERE = Path(__file__).resolve().parent


def engine(root,code):
    result = run_game(root, code)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout)


def imports(v1_path,v2_path=None):
    code = (
        "import importlib.util,json; "
        f"s=importlib.util.spec_from_file_location('v1',{str(HERE/'world_model.py')!r}); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        f"v1=m.load_predictor({str(v1_path)!r}); "
    )
    if v2_path:
        code += (
            f"s2=importlib.util.spec_from_file_location('v2',{str(HERE/'world_model_v2.py')!r}); "
            "m2=importlib.util.module_from_spec(s2); s2.loader.exec_module(m2); "
            f"v2=m2.load_predictor({str(v2_path)!r}); "
        )
    return code


def tensors(rows):
    return (
        torch.tensor([r["input"] for r in rows],dtype=torch.float32),
        torch.tensor([r["target"] for r in rows],dtype=torch.float32),
    )


def predict_metrics(predictions,targets):
    result = {"transitions":len(targets)}
    for index,name in ((17,"immediate"),(18,"followup")):
        p,y = predictions[:,index],targets[:,index]
        tp = ((p>=0.5)&(y>=0.5)).sum().item()
        result[name] = {
            "positives":int((y>=0.5).sum()),
            "brier":float(((p-y)**2).mean()),
            "precision":tp/max(1,int((p>=0.5).sum())),
            "recall":tp/max(1,int((y>=0.5).sum())),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("fit","test"),required=True)
    parser.add_argument("--driftmoor-root",type=Path,required=True)
    parser.add_argument("--v1-checkpoint",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--epochs",type=int,default=160)
    args = parser.parse_args()
    root = args.driftmoor_root.expanduser().resolve()
    original = args.v1_checkpoint.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not (root/"engine/md_loader.py").exists() or not original.is_file():
        parser.error("Experimental checkout and existing v1 checkpoint required")
    if args.epochs<1:
        parser.error("--epochs must be positive")
    output.mkdir(parents=True,exist_ok=True)
    checkpoint = output/"predictor-v2.pt"
    torch.set_num_threads(1)
    if args.stage=="fit":
        if checkpoint.exists():
            parser.error("Use a new output directory; preserve fitted checkpoints")
        print("Collecting covered training and validation actions...",flush=True)
        code = imports(original)+(
            "from spider_sim.spider_v2 import collect_covered,collect_pose_pairs; "
            "print(json.dumps({"
            "'train':collect_covered(160,1000,v1)+collect_pose_pairs(200,40000),"
            "'validation':collect_covered(30,11000,v1)+collect_pose_pairs(40,50000)"
            "}))"
        )
        data = engine(root,code)
        ids = [{row["episode_id"] for row in data[name]} for name in ("train","validation")]
        assert not ids[0]&ids[1]
        for name,rows in data.items():
            (output/f"{name}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
        x,y = tensors(data["train"])
        vx,vy = tensors(data["validation"])
        torch.manual_seed(17)
        model = Network()
        optimizer = torch.optim.Adam(model.parameters(),lr=0.002)
        best = float("inf")
        chosen = 0
        print(f"Training on {len(x)} transitions; {int(y[:,17].sum())} immediate positives...",flush=True)
        for epoch in range(args.epochs):
            model.train()
            for batch in torch.randperm(len(x)).split(256):
                value = model(x[batch])
                loss = ((value[:,:17]-y[batch,:17])**2).mean()
                loss += nn.functional.binary_cross_entropy_with_logits(value[:,17:],y[batch,17:])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                value = model(vx)
                validation = float(
                    ((value[:,:17]-vy[:,:17])**2).mean()
                    + nn.functional.binary_cross_entropy_with_logits(value[:,17:],vy[:,17:])
                )
            if validation<best:
                best,chosen = validation,epoch+1
                torch.save({"version":2,"state_dict":model.state_dict(),"seed":17},checkpoint)
        predictor = load_predictor(checkpoint)
        predictions = torch.tensor(predictor(vx.tolist()))
        print("Comparing controllers on validation only...",flush=True)
        comparison = engine(root,imports(original,checkpoint)+(
            "from spider_sim.spider_v2 import evaluate_v2; "
            "print(json.dumps(evaluate_v2(range(11000,11030),v1,v2)))"
        ))
        report = {
            "stage":"validation","selected_epoch":chosen,"validation_loss":best,
            "configuration":{
                "training_seed":17,"epochs":args.epochs,
                "episode_train_seeds":[1000,1159],"pose_train_seeds":[40000,40199],
                "episode_validation_seeds":[11000,11029],"pose_validation_seeds":[50000,50039],
                "discount":0.8,"second_action_options":["bite","wait"],
            },
            "training":{
                "rows":len(x),"immediate_positives":int(y[:,17].sum()),
                "followup_positives":int(y[:,18].sum()),
            },
            "prediction":predict_metrics(predictions,vy),"controllers":comparison,
        }
        filename = "validation-report.json"
    else:
        if not checkpoint.exists() or not (output/"validation-report.json").exists():
            parser.error("Run fit and inspect validation first")
        if (output/"test-report.json").exists():
            parser.error("Fresh-test report already exists; preserve it")
        print("Evaluating the frozen model on fresh seeds 30000-30039...",flush=True)
        data = engine(root,imports(original,checkpoint)+(
            "from spider_sim.spider_v2 import collect_covered,collect_pose_pairs,evaluate_v2; "
            "print(json.dumps({"
            "'rows':collect_covered(40,30000,v1)+collect_pose_pairs(40,60000),"
            "'controllers':evaluate_v2(range(30000,30040),v1,v2)}))"
        ))
        (output/"test.jsonl").write_text("".join(json.dumps(r)+"\n" for r in data["rows"]))
        tx,ty = tensors(data["rows"])
        predictions = torch.tensor(load_predictor(checkpoint)(tx.tolist()))
        report = {
            "stage":"fresh_test","episode_seeds":[30000,30039],"pose_seeds":[60000,60039],
            "prediction":predict_metrics(predictions,ty),"controllers":data["controllers"],
            "selection":"Configuration frozen after validation, before generating these episodes",
        }
        filename = "test-report.json"
    report["v1_checkpoint_sha256"] = hashlib.sha256(original.read_bytes()).hexdigest()
    report["v2_checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report["scope"] = (
        "Direct two-step action outcome prediction for sequences [candidate, bite-or-wait]; "
        "not autoregressive latent planning, JEPA, or an LLM"
    )
    (output/filename).write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({
        "stage":report["stage"],"prediction":report["prediction"],
        "controllers":{k:v["totals"] for k,v in report["controllers"].items()},
        "training":report.get("training"),"selected_epoch":report.get("selected_epoch"),
    },indent=2))
    print(f"Report: {output/filename}")


if __name__=="__main__":
    main()
