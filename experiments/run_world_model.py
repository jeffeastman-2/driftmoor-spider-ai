"""Collect, train and compare observation-driven spider controllers."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch
from torch import nn

if __package__:
    from .world_model import Network, load_predictor
else:
    from world_model import Network, load_predictor

HERE = Path(__file__).resolve().parent


def driftmoor(root, code):
    result = subprocess.run(
        [sys.executable,"-c",code],cwd=root,text=True,capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout)


def tensors(rows):
    return (
        torch.tensor([r["input"] for r in rows],dtype=torch.float32),
        torch.tensor([r["target"] for r in rows],dtype=torch.float32),
    )


def metrics(prediction, target):
    admitted = target[:,-1]
    probabilities = prediction[:,-1]
    positive = probabilities >= 0.5
    actual = admitted >= 0.5
    true_positive = int((positive & actual).sum())
    visible = target[:,0] >= 0.5
    return {
        "observation_mse": float(((prediction[:,:-1]-target[:,:-1])**2).mean()),
        "visible_range_mae_ft": float(
            ((prediction[visible,1]-target[visible,1]).abs()*60).mean()
        ) if visible.any() else None,
        "attack_brier": float(((probabilities-admitted)**2).mean()),
        "attack_precision": true_positive/max(1,int(positive.sum())),
        "attack_recall": true_positive/max(1,int(actual.sum())),
        "admitted_attack_labels": int(actual.sum()),
        "transitions": len(target),
    }


def persistence_predictions(inputs, train_targets):
    # Current observation is the second 17-vector; no future information.
    current = inputs[:,17:34].clone()
    prior = train_targets[:,-1].mean().expand(len(current),1)
    return torch.cat((current,prior),dim=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driftmoor-root",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--train-episodes",type=int,default=160)
    parser.add_argument("--validation-episodes",type=int,default=30)
    parser.add_argument("--test-episodes",type=int,default=40)
    parser.add_argument("--epochs",type=int,default=140)
    args = parser.parse_args()
    if any(v < 1 for v in (
        args.train_episodes,args.validation_episodes,args.test_episodes,args.epochs
    )):
        parser.error("Counts must be positive")
    if args.train_episodes >= 10000 or args.validation_episodes >= 10000:
        parser.error("Counts would overlap reserved seed splits")
    root = args.driftmoor_root.expanduser().resolve()
    if not (root/"engine/spider_learning.py").is_file():
        parser.error("Choose the experimental Driftmoor checkout")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(7)
    code = (
        "import json; from engine.spider_learning import collect; "
        f"print(json.dumps({{'train':collect({args.train_episodes},0),"
        f"'validation':collect({args.validation_episodes},10000),"
        f"'test':collect({args.test_episodes},20000)}}))"
    )
    print("Collecting real engine transitions...",flush=True)
    data = driftmoor(root,code)
    for name,rows in data.items():
        (output/f"{name}.jsonl").write_text(
            "".join(json.dumps(row)+"\n" for row in rows)
        )
    sets = {name:tensors(rows) for name,rows in data.items()}
    ids = {name:{r["episode_id"] for r in rows} for name,rows in data.items()}
    assert not (ids["train"] & ids["validation"] or ids["train"] & ids["test"]
                or ids["validation"] & ids["test"])
    x,y = sets["train"]
    model = Network(x.shape[1],y.shape[1]-1)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.002)
    best_loss = float("inf")
    selected_epoch = 0
    checkpoint_path = output/"predictor.pt"
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(x))
        for batch in order.split(128):
            result = model(x[batch])
            loss = ((result[:,:-1]-y[batch,:-1])**2).mean()
            loss += nn.functional.binary_cross_entropy_with_logits(result[:,-1],y[batch,-1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            vx,vy = sets["validation"]
            value = model(vx)
            validation_loss = float(
                ((value[:,:-1]-vy[:,:-1])**2).mean()
                + nn.functional.binary_cross_entropy_with_logits(value[:,-1],vy[:,-1])
            )
        if validation_loss < best_loss:
            best_loss = validation_loss
            selected_epoch = epoch+1
            torch.save({
                "state_dict":model.state_dict(),"inputs":x.shape[1],
                "observations":y.shape[1]-1,"training_seed":7,
            },checkpoint_path)
    predictor = load_predictor(checkpoint_path)
    tx,ty = sets["test"]
    predictions = torch.tensor(predictor(tx.tolist()))
    constant = y.mean(dim=0).expand(len(ty),-1)
    persistence = persistence_predictions(tx,y)
    print("Evaluating controllers on held-out episodes...",flush=True)
    module_path = str(HERE/"world_model.py")
    eval_code = (
        "import importlib.util,json; "
        f"s=importlib.util.spec_from_file_location('spider_predictor',{module_path!r}); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        f"predictor=m.load_predictor({str(checkpoint_path)!r}); "
        "from engine.spider_learning import evaluate; "
        f"print(json.dumps(evaluate(range(20000,{20000+args.test_episodes}),predictor)))"
    )
    comparison = driftmoor(root,eval_code)
    report = {
        "configuration":{
            "train_episodes":args.train_episodes,
            "validation_episodes":args.validation_episodes,
            "test_episodes":args.test_episodes,
            "epochs":args.epochs,"training_seed":7,"steps":14,
        },
        "scope":"Structured-observation one-step predictor; not JEPA/V-JEPA or a full autonomous agent",
        "split":"Whole episodes; train seeds 0+, validation 10000+, test 20000+",
        "selected_epoch":selected_epoch,"validation_loss":best_loss,
        "predictor":metrics(predictions,ty),"constant_baseline":metrics(constant,ty),
        "persistence_baseline":metrics(persistence,ty),
        "controllers":comparison,
        "limitations":[
            "Scripted non-attacking player; no self-preservation evaluation",
            "Single player and cavern; doors assumed open; movement and occlusion remain simplified",
            "One-step candidate scoring, not multi-step model-predictive control",
            "Success label is attack admission, not dice hit",
            "Coarse contacts can cause rejected attacks despite predicted reach",
            "Same behavior families, disjoint episode seeds; no claim of new-world generalization",
        ],
    }
    (output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({
        "predictor":report["predictor"],"constant_baseline":report["constant_baseline"],
        "persistence_baseline":report["persistence_baseline"],
        "controllers":{k:v["totals"] for k,v in comparison.items()},
        "selected_epoch":selected_epoch,
    },indent=2))
    print(f"Report: {output/'report.json'}")


if __name__ == "__main__":
    main()
