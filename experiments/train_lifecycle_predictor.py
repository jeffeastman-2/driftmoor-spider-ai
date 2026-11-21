"""Frozen-context forecasts on demonstrated actions; no counterfactual claims."""
import argparse,json,random
from pathlib import Path
import torch
from spider_sim.lifecycle_model import LatentPredictor,action_features
from spider_sim.learned_lifecycle import load as load_actor
from experiments.train_lifecycle_model import load_corpus,batches


@torch.no_grad()
def transitions(model,sequences):
    states=[];actions=[];targets=[];valid=[]
    for x,lengths,_,group in batches(sequences):
        z=model.represent(x,lengths)
        for i,s in enumerate(group):
            rows=s['rows'];n=len(rows);state=z[i,:n]
            delta=torch.zeros_like(state);delta[:-1]=state[1:]-state[:-1]
            body=torch.tensor([[r['after_round'][k]-r['observation']['self'][k] for k in ('hp_fraction','hunger')] for r in rows])
            states.append(state);actions.append(torch.tensor([action_features(r['observation'],r['command']) for r in rows]))
            targets.append(torch.cat((delta,body),-1));valid.extend([True]*(n-1)+[False])
    return torch.cat(states),torch.cat(actions),torch.cat(targets),torch.tensor(valid)


def loss(pred,target,valid,scale):
    errors=((pred-target)/scale)**2
    context=errors[:,:64][valid].mean() if valid.any() else errors[:,:64].sum()*0
    return context+errors[:,64:].mean()


@torch.no_grad()
def metrics(model,data,scale):
    state,action,target,valid=data
    pred=torch.zeros_like(target) if model is None else model(state,action)
    return {'normalized_loss':float(loss(pred,target,valid,scale)),
        'next_context_delta_rmse':float(((pred[:,:64]-target[:,:64])[valid]**2).mean().sqrt()),
        'hp_fraction_delta_mae':float((pred[:,64]-target[:,64]).abs().mean()),
        'hunger_delta_mae':float((pred[:,65]-target[:,65]).abs().mean()),
        'transitions':len(target),'valid_next_context_targets':int(valid.sum()),
        'no_next_observation':int((~valid).sum()),
        'outcome_subsets':{name:{'count':int(mask.sum()),
            'hp_fraction_delta_mae':float((pred[mask,64]-target[mask,64]).abs().mean()) if mask.any() else None,
            'hunger_delta_mae':float((pred[mask,65]-target[mask,65]).abs().mean()) if mask.any() else None}
            for name,mask in {'injured':target[:,64]<-1e-6,'uninjured':target[:,64]>=-1e-6,
                'fed':target[:,65]<-1e-6,'not_fed':target[:,65]>=-1e-6}.items()}}


def main():
    p=argparse.ArgumentParser();p.add_argument('corpus',type=Path);p.add_argument('actor',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=30);a=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(2026);random.seed(2026);a.output.mkdir(parents=True,exist_ok=True)
    actor=load_actor(a.actor)
    data={k:transitions(actor,v) for k,v in load_corpus(a.corpus,actor.feature_version).items()}
    train=data['train'];scale=train[2][train[3]].std(0).clamp_min(.02);scale[64:]=.1
    result={'schema':'lifecycle-frozen-predictor-v1','actor':str(a.actor),
        'target':'next causal GRU context delta where next observation exists; after-round own HP/hunger delta on every recorded decision',
        'terminal_handling':'No fabricated next context for death or end-of-run. Body targets remain observed. Death-before-next-decision also has no next context.',
        'limitations':'Teacher actions only; predictive association does not validate imagined alternative actions or improve behavior. Body targets end this round, not next decision: attacks before next observation may be omitted. Not a next-turn combat-damage forecast.',
        'normalization':scale.tolist(),'models':{'persistence':{k:metrics(None,v,scale) for k,v in data.items() if k!='train'}}}
    for conditioned in (False,True):
        torch.manual_seed(2026);model=LatentPredictor(action_conditioned=conditioned);opt=torch.optim.AdamW(model.parameters(),lr=.001)
        best=float('inf');history=[];state,action,target,valid=train
        for epoch in range(1,a.epochs+1):
            model.train()
            for ids in torch.randperm(len(state)).split(1024):
                opt.zero_grad();value=loss(model(state[ids],action[ids]),target[ids],valid[ids],scale);value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            model.eval();score=metrics(model,data['validation'],scale)['normalized_loss'];history.append(score)
            if score<best:best=score;best_epoch=epoch;checkpoint={k:v.clone() for k,v in model.state_dict().items()}
        model.load_state_dict(checkpoint);name='action_conditioned' if conditioned else 'state_only'
        torch.save({'state_dict':model.state_dict(),'action_conditioned':conditioned,'actor':str(a.actor)},a.output/f'{name}.pt')
        result['models'][name]={'best_epoch':best_epoch,'validation_history':history,**{k:metrics(model,v,scale) for k,v in data.items() if k!='train'}}
        print(json.dumps({'model':name,'test':result['models'][name]['test']}),flush=True)
    (a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
