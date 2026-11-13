"""Learn engine bite admission from coarse versus accurate private measurements."""
import argparse
import copy
import json
import math
import random
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from experiments.game_runtime import game_root
from experiments.accurate_senses import AccurateEncounter,observation,SCHEMA,FIELDS


def collect(count,seed):
    from spider_sim.dual_encounter import ACTORS,SPIDER,PLAYER
    from models.body_geometry import Orientation,Body
    from models.space_time import Position
    rng=random.Random(seed)
    template=AccurateEncounter(seed,outages=False,scenario="wounded_spider")
    rows=[]
    while len(rows)<count:
        world=copy.deepcopy(template)
        distance=rng.uniform(6,18);bearing=rng.uniform(-180,180)
        own_heading=bearing+rng.uniform(-35,35) if rng.random()<.7 else rng.uniform(-180,180)
        own_pitch=rng.uniform(-60,60);own_roll=180 if rng.random()<.25 else 0
        world.orientations[SPIDER]=Orientation(own_roll,own_pitch,own_heading)
        world.orientations[PLAYER]=Orientation(rng.uniform(-30,30),rng.uniform(-45,45),rng.uniform(-180,180))
        world.heights[SPIDER]=rng.choice((0.,0.,0.,10.))
        world.heights[PLAYER]=rng.choice((0.,0.,0.,2.,5.))
        world.bodies[PLAYER]=Body(rng.uniform(1.5,4),rng.uniform(1.5,4),rng.uniform(3,7),"ideal visible box fixture")
        x,y=70+distance*math.cos(math.radians(bearing)),195+distance*math.sin(math.radians(bearing))
        zone=world.geometry.zone_at(Position(x,y))
        if not zone:
            continue
        world.positions[PLAYER]=(x,y,zone)
        for cid in ACTORS:
            world.config.characters[cid].character.current_hp=world.max_hp[cid]
        world.config.characters[SPIDER].character.current_hp=max(1,round(world.max_hp[SPIDER]*rng.uniform(.1,1)))
        world.concealed=False;world.begin_turn(SPIDER)
        world.random.seed(seed*10000+len(rows))  # Independent combat rolls; admission is not a hit label.
        obs=observation(world,SPIDER)
        result=world.act(SPIDER,7,orientation=world.orientations[SPIDER])
        rows.append({"accurate":obs["values"],"coarse":obs["legacy_values"],
            "admitted":int(result["admitted"]),"status":result["status"],"visible":obs["visible"]})
    return rows


def tensors(rows,key):
    return torch.tensor([r[key] for r in rows],dtype=torch.float32),torch.tensor([r["admitted"] for r in rows],dtype=torch.float32)


@torch.no_grad()
def metrics(model,rows,key):
    x,y=tensors(rows,key);p=model(x).squeeze(-1).sigmoid();choice=p>=.5
    positive=y==1;negative=~positive
    recall=float(choice[positive].float().mean()) if positive.any() else None
    specificity=float((~choice[negative]).float().mean()) if negative.any() else None
    visible=torch.tensor([r["visible"] for r in rows])
    result={"samples":len(rows),"legal":int(positive.sum()),"visible":int(visible.sum()),
        "accuracy":float((choice==positive).float().mean()),"legal_recall":recall,"illegal_recall":specificity,
        "balanced_accuracy":(recall+specificity)/2 if recall is not None and specificity is not None else None,
        "brier":float((p-y).square().mean())}
    result["by_status"]={status:{"count":sum(r["status"]==status for r in rows),
        "mean_predicted_admission":float(p[torch.tensor([r["status"]==status for r in rows])].mean())}
        for status in sorted({r["status"] for r in rows})}
    return result


def train(training,validation,key,seed,epochs=120):
    torch.manual_seed(seed)
    x,y=tensors(training,key);vx,vy=tensors(validation,key)
    model=nn.Sequential(nn.Linear(x.shape[1],64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    weight=(1-y).sum()/y.sum().clamp(min=1)
    def objective(logits,target):
        return F.binary_cross_entropy_with_logits(logits.squeeze(-1),target,pos_weight=weight)
    best=float("inf");state=None
    generator=torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order=torch.randperm(len(y),generator=generator)
        for start in range(0,len(y),256):
            indices=order[start:start+256]
            optimizer.zero_grad();loss=objective(model(x[indices]),y[indices]);loss.backward();optimizer.step()
        with torch.no_grad():
            score=float(objective(model(vx),vy))
        if score<best:
            best=score;state=copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    return model.eval()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driftmoor-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error("Use a new directory")
    game_root(args.driftmoor_root);torch.set_num_threads(1);args.output.mkdir(parents=True)
    splits={}
    for name,n,seed in (("train",4096,900000),("validation",1024,910000),("test",2048,920000)):
        print("Collecting",name,flush=True);splits[name]=collect(n,seed)
        (args.output/(name+".json")).write_text(json.dumps(splits[name]))
    report={"schema":SCHEMA,"fields":list(FIELDS),"input_sizes":{"coarse":41,"accurate":57},
        "task":"Bite at current explicitly supplied orientation; engine admission, not hit/damage",
        "optimizer_seeds":[42,43,44],"epochs":120,"models":{}}
    for key in ("coarse","accurate"):
        for seed in (42,43,44):
            model=train(splits["train"],splits["validation"],key,seed)
            result=metrics(model,splits["test"],key);report["models"][key+"-"+str(seed)]=result
            torch.save({"schema":SCHEMA if key=="accurate" else "withdrawal-cornered-v6",
                "input_size":57 if key=="accurate" else 41,"fields":list(FIELDS) if key=="accurate" else None,
                "state":model.state_dict(),"task":"current-pose bite admission"},args.output/(key+"-"+str(seed)+".pt"))
            print(key,seed,json.dumps(result),flush=True)
    report["limitations"]=["Ideal exact visible boxes, not camera reconstruction or measured real-world shapes",
        "Separate admission diagnostic, not replacement of full recurrent controller",
        "Direct MLPs trained from scratch for both input sets; no frozen encoder bottleneck",
        "Same fixture distribution with held-out random seeds, not new maps or species",
        "Own reach/capability supplied; no closest-point distance, legality, target HP or identity supplied",
        "Class-weighted objective: scores are not automatically calibrated probabilities",
        "No aggression/cost changes, autonomous behavior claim, or deployment"]
    (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")

if __name__=="__main__":main()
