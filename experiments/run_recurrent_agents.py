"""Train two independent imitation agents and evaluate held-out episodes."""
import argparse
import copy
import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
from experiments.recurrent_agents import Agent,save

def batch(episodes,role):
    rows=[e["actors"][role] for e in episodes]
    length=max(map(len,rows))
    def pad(key,size):
        return torch.tensor([[r[key] for r in seq]+[[0]*size]*(length-len(seq)) for seq in rows],dtype=torch.float32)
    obs=pad("observation",41)
    memory=pad("memory_target",5)
    future=pad("next_observation",41)
    feedback=pad("feedback",2)
    mask=torch.tensor([[True]*len(seq)+[False]*(length-len(seq)) for seq in rows])
    labels=torch.tensor([[r["teacher_action"] for r in seq]+[0]*(length-len(seq)) for seq in rows])
    actions=torch.tensor([[r["executed_action"] for r in seq]+[0]*(length-len(seq)) for seq in rows])
    gaps=torch.tensor([[r["contact_missing"] for r in seq]+[False]*(length-len(seq)) for seq in rows])
    return obs,memory,torch.cat((future,feedback),-1),mask,labels,actions,gaps

def objective(agent,data):
    obs,memory,target,mask,labels,actions,gaps=data
    policy,belief,prediction,_=agent(obs,actions)
    policy_loss=F.cross_entropy(policy[mask],labels[mask])
    valid=mask & (memory[...,0]>0)
    belief_loss=F.binary_cross_entropy(belief[...,0][mask],memory[...,0][mask])
    if valid.any():
        belief_loss=belief_loss+F.mse_loss(belief[...,1:][valid],memory[...,1:][valid])
    return policy_loss+0.5*belief_loss+0.5*F.mse_loss(prediction[mask],target[mask])

@torch.no_grad()
def metrics(agent,data):
    obs,memory,target,mask,labels,actions,gaps=data
    policy,belief,prediction,_=agent(obs,actions)
    recall=mask & gaps & (memory[...,0]>0)
    return {
        "policy_accuracy":float((policy.argmax(-1)[mask]==labels[mask]).float().mean()),
        "next_observation_mse":float(F.mse_loss(prediction[...,:41][mask],target[...,:41][mask])),
        "memory_validity_accuracy":float(((belief[...,0]>0.5)[mask]==(memory[...,0]>0)[mask]).float().mean()),
        "missing_contact_recall_frames":int(recall.sum()),
        "remembered_range_mae_ft":float((belief[...,1][recall]-memory[...,1][recall]).abs().mean()*60) if recall.any() else None,
    }

def train(kind,role,train_data,validation,test,epochs):
    torch.manual_seed(42)
    agent=Agent(kind)
    optimizer=torch.optim.Adam(agent.parameters(),lr=.003)
    best=float("inf")
    state=None
    for epoch in range(epochs):
        agent.train()
        optimizer.zero_grad()
        loss=objective(agent,train_data)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(),1)
        optimizer.step()
        agent.eval()
        with torch.no_grad():
            score=float(objective(agent,validation))
        if score<best:
            best=score
            state=copy.deepcopy(agent.state_dict())
    agent.load_state_dict(state)
    return agent.eval(),{"validation_loss":best,"test":metrics(agent,test)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driftmoor-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--epochs",type=int,default=160)
    parser.add_argument("--train-episodes",type=int,default=160)
    args=parser.parse_args()
    if not 1 <= args.train_episodes <= 1000 or args.epochs < 1:
        parser.error("Use 1..1000 training episodes and at least one epoch.")
    sys.path.insert(0,str(args.driftmoor_root.resolve()))
    from engine.dual_encounter import collect,ACTORS,neural_encounters
    torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=True)
    splits={"train":collect(args.train_episodes,70000),"validation":collect(24,71000),"test":collect(32,72000)}
    for name,episodes in splits.items():
        (args.output/(name+".json")).write_text(json.dumps(episodes))
    report={"method":"supervised imitation plus sensory-memory and action-conditioned next-observation prediction","epochs":args.epochs,"splits":{k:[e["episode_id"] for e in v] for k,v in splits.items()},"agents":{}}
    trained={}
    for kind in ("mlp","gru"):
        trained[kind]={}
        report["agents"][kind]={}
        for role in ACTORS:
            agent,result=train(kind,role,*(batch(splits[k],role) for k in ("train","validation","test")),args.epochs)
            trained[kind][role]=agent
            report["agents"][kind][role]=result
            save(agent,args.output/(kind+"-"+role+".pt"))
            print(kind,role,json.dumps(result),flush=True)
    report["rollouts"]={}
    for kind,agents in trained.items():
        outcome=neural_encounters(range(73000,73016),agents)
        report["rollouts"][kind]=outcome["totals"]
        (args.output/(kind+"-rollouts.json")).write_text(json.dumps(outcome["traces"]))
    (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report["rollouts"],indent=2))

if __name__=="__main__":
    main()
