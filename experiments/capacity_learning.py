"""Matched sequential training for frozen, unfrozen, and deeper encoders."""
import copy
import torch
import torch.nn.functional as F
from torch import nn
from experiments.combat_learning import residual_predictor,forecast_metrics,replay_tensors

VARIANTS=("predictor_only","unfrozen","deeper")


def initialize(prior,variant,seed):
    if variant not in VARIANTS:
        raise ValueError(variant)
    agent=residual_predictor(prior)
    torch.manual_seed(seed)
    if variant=="deeper":
        old=agent.encoder[0]
        encoder=nn.Sequential(nn.Linear(41,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU())
        with torch.no_grad():
            encoder[0].weight[:32].copy_(old.weight)
            encoder[0].bias[:32].copy_(old.bias)
            # Extra hidden units have nonzero features, initially zero output
            # weights: exact old function without permanently dead extra units.
            encoder[2].weight.zero_();encoder[2].bias.zero_()
            encoder[2].weight[:,:32].copy_(torch.eye(32))
        agent.encoder=encoder;agent.encoder_depth=2
    for name,param in agent.named_parameters():
        param.requires_grad_(name.startswith("predictor.") or
            (variant!="predictor_only" and name.startswith(("encoder.","memory."))))
    return agent


def sequence_batch(episodes):
    episodes=[episode for episode in episodes if episode["rows"]]
    length=max(len(episode["rows"]) for episode in episodes)
    obs=torch.zeros(len(episodes),length,41)
    targets=torch.zeros(len(episodes),length,43)
    actions=torch.zeros(len(episodes),length,dtype=torch.long)
    mask=torch.zeros(len(episodes),length,dtype=torch.bool)
    for i,episode in enumerate(episodes):
        for j,row in enumerate(episode["rows"]):
            obs[i,j]=torch.tensor(row["observation"])
            targets[i,j]=torch.tensor(row["next_observation"]+row["feedback"])
            actions[i,j]=row["action"];mask[i,j]=True
    return obs,actions,targets,mask


def loss(agent,data,action_weights=None):
    obs,actions,target,mask=data
    _,_,pred,_=agent(obs,actions)
    per_row=((pred[...,:41]-target[...,:41]).square().mean(-1)
        +10*(pred[...,16]-target[...,16]).square()
        +2*(pred[...,41]-target[...,41]).square()
        +4*(pred[...,42]-target[...,42]).square())
    weights=torch.ones_like(per_row) if action_weights is None else action_weights[actions]
    return (per_row[mask]*weights[mask]).sum()/weights[mask].sum()


def train_variant(prior,variant,training,validation,epochs=60,seed=42):
    agent=initialize(prior,variant,seed)
    optimizer=torch.optim.Adam([p for p in agent.parameters() if p.requires_grad],lr=.001)
    valid=sequence_batch(validation)
    counts=torch.bincount(torch.tensor([row["action"] for e in training for row in e["rows"]]),minlength=8).clamp(min=1)
    weights=1/counts.float()
    batches=[sequence_batch(training[i:i+32]) for i in range(0,len(training),32)]
    generator=torch.Generator().manual_seed(seed)
    with torch.no_grad():
        best=float(loss(agent,valid))
    state=copy.deepcopy(agent.state_dict());best_epoch=0
    for epoch in range(epochs):
        for index in torch.randperm(len(batches),generator=generator).tolist():
            optimizer.zero_grad();objective=loss(agent,batches[index],weights)
            objective.backward();torch.nn.utils.clip_grad_norm_(agent.parameters(),1);optimizer.step()
        with torch.no_grad():
            value=float(loss(agent,valid))
        if value<best:
            best=value;state=copy.deepcopy(agent.state_dict());best_epoch=epoch+1
    agent.load_state_dict(state)
    for name,param in agent.state_dict().items():
        if name.startswith(("policy.","belief.")) or (variant=="predictor_only" and name.startswith(("encoder.","memory."))):
            assert torch.equal(param,prior.state_dict()[name]),name
    return agent.eval(),{"validation_loss":best,"best_epoch":best_epoch,"epochs":epochs}


@torch.no_grad()
def evaluate_forecasts(agent,prior,episodes):
    result=forecast_metrics(agent,replay_tensors(agent,episodes))
    obs,actions,target,mask=sequence_batch(episodes)
    policy,belief,_,_=agent(obs)
    old_policy,old_belief,_,_=prior(obs)
    result["policy_agreement_with_prior"]=float((policy.argmax(-1)[mask]==old_policy.argmax(-1)[mask]).float().mean())
    result["belief_output_mae_from_prior"]=float((belief[mask]-old_belief[mask]).abs().mean())
    return result
