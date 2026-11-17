"""Fit explicit outcome heads on a fixed pursuit encoder/GRU representation.

Not a JEPA latent-prediction objective, reward learning, or a live policy update.
"""
import argparse
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

VARIANTS=('frozen_memory','context_only','frozen_memory_context')


class OutcomeHead(nn.Module):
    def __init__(self,inputs):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(inputs,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,12))
    def forward(self,x):return self.layers(x).reshape(-1,3,4)


def probabilities(raw):
    return torch.stack((raw[:,:,0].sigmoid(),raw[:,:,1].tanh(),raw[:,:,2].sigmoid(),raw[:,:,3].sigmoid()),dim=-1)


def loss(raw,target):
    prediction=probabilities(raw)
    return F.mse_loss(prediction[:,:,:2],target[:,:,:2])+F.binary_cross_entropy_with_logits(raw[:,:,2:],target[:,:,2:])


def inputs(record,action,variant):
    onehot=[float(action==i) for i in range(3)]
    if variant=='frozen_memory':return record['latent']+onehot
    if variant=='context_only':return record['context']+onehot
    if variant=='frozen_memory_context':return record['latent']+record['context']+onehot
    raise ValueError(variant)


def load_records(path):
    with gzip.open(path/'forks.jsonl.gz','rt') as stream:return [json.loads(line) for line in stream]


def tensors(records,split,variant):
    x=[];y=[];groups=[]
    for i,record in enumerate(records):
        if record['split']!=split:continue
        for branch in record['branches']:
            x.append(inputs(record,branch['action'],variant));y.append(branch['targets'])
            groups.append((i,branch['action']))
    return torch.tensor(x,dtype=torch.float32),torch.tensor(y,dtype=torch.float32),groups


def metrics(predictions,targets):
    result={}
    for h,index in ((3,0),(6,1),(12,2)):
        p=predictions[:,index];t=targets[:,index]
        bins=[]
        for lo in np.linspace(0,.8,5):
            mask=(p[:,2]>=lo)&(p[:,2]<(lo+.2) if lo<.8 else p[:,2]<=1)
            bins.append({'lower':float(lo),'count':int(mask.sum()),'predicted':float(p[mask,2].mean()) if mask.any() else None,'observed':float(t[mask,2].mean()) if mask.any() else None})
        result[str(h)]={'damage_mae_hp':float((p[:,0]-t[:,0]).abs().mean()*26),
                        'hunger_reduction_mae':float((p[:,1]-t[:,1]).abs().mean()),
                        'attack_brier':float(((p[:,2]-t[:,2])**2).mean()),
                        'death_brier':float(((p[:,3]-t[:,3])**2).mean()),
                        'observed_attack_rate':float(t[:,2].mean()),'predicted_attack_rate':float(p[:,2].mean()),
                        'observed_death_rate':float(t[:,3].mean()),'attack_calibration_bins':bins}
    return result


def ranking(predictions,targets,groups):
    buckets=defaultdict(lambda:defaultdict(list))
    for index,(origin,action) in enumerate(groups):buckets[origin][action].append(index)
    injury=[];food=[]
    for choices in buckets.values():
        if len(choices)!=3:continue
        pred=torch.stack([predictions[choices[a],2].mean(0) for a in range(3)])
        actual=torch.stack([targets[choices[a],2].mean(0) for a in range(3)])
        injury.append(float((actual[int(pred[:,0].argmin()),0]-actual[:,0].min())*26))
        food.append(float(actual[:,1].max()-actual[int(pred[:,1].argmax()),1]))
    return dict(origins=len(injury),mean_damage_only_choice_regret_hp=float(np.mean(injury)),mean_food_only_choice_regret=float(np.mean(food)),
                note='Separate single-objective diagnostics, not a combined decision cost. Empirical best of four rollout draws is noisy.')


class OutcomeEnsemble:
    """Read-only inference; caller supplies the already updated private memory.

    This does not choose actions or update the pursuit network/GRU.
    """
    def __init__(self,directory,variant,expected_source_sha256):
        self.variant=variant;self.members=[]
        for seed in range(3):
            payload=torch.load(Path(directory)/f'{variant}-{seed}.pt',map_location='cpu',weights_only=True)
            if payload.get('schema')!='meal-outcome-head-v1' or payload.get('variant')!=variant or payload.get('source_model_sha256')!=expected_source_sha256:
                raise ValueError('Outcome predictor/source model contract mismatch')
            head=OutcomeHead(payload['inputs']);head.load_state_dict(payload['state']);head.eval().requires_grad_(False)
            self.members.append((head,payload['mean'],payload['std']))

    @torch.no_grad()
    def tensor(self,latent,context):
        if len(latent)!=48 or len(context)!=8:raise ValueError('Expected 48 memory and 8 context values')
        record={'latent':list(latent),'context':list(context)}
        x=torch.tensor([inputs(record,a,self.variant) for a in range(3)],dtype=torch.float32)
        if not torch.isfinite(x).all():raise ValueError('Nonfinite predictor input')
        return torch.stack([probabilities(head((x-mean)/std)) for head,mean,std in self.members]).mean(0)

    def predict(self,latent,context,max_hp=26):
        values=self.tensor(latent,context)
        return {mode:{str(horizon):dict(damage_hp=float(values[a,h,0])*max_hp,
                        hunger_reduction=float(values[a,h,1]),damaging_attack_probability=float(values[a,h,2]),
                        death_probability=float(values[a,h,3])) for h,horizon in enumerate((3,6,12))}
                for a,mode in enumerate(('pursue','guard','resume'))}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('data',type=Path);parser.add_argument('--epochs',type=int,default=160)
    args=parser.parse_args();torch.set_num_threads(1)
    records=load_records(args.data);metadata=json.loads((args.data/'metadata.json').read_text())
    split_sets={split:{r['seed'] for r in records if r['split']==split} for split in ('train','val','test')}
    assert all(not split_sets[a]&split_sets[b] for a,b in (('train','val'),('train','test'),('val','test')))
    result={'kind':'supervised explicit outcome predictor; encoder and GRU frozen','dataset':metadata,'variants':{},'optimizer_seeds':[0,1,2]}
    for variant in VARIANTS:
        train_x,train_y,_=tensors(records,'train',variant);val_x,val_y,_=tensors(records,'val',variant)
        test_x,test_y,groups=tensors(records,'test',variant)
        mean=train_x.mean(0);std=train_x.std(0).clamp_min(.1)
        train_x=(train_x-mean)/std;val_x=(val_x-mean)/std;test_x=(test_x-mean)/std
        predictions=[];runs=[]
        for seed in range(3):
            torch.manual_seed(seed);head=OutcomeHead(train_x.shape[1]);optimizer=torch.optim.Adam(head.parameters(),lr=.002)
            best=float('inf');best_state=None;best_epoch=0
            for epoch in range(args.epochs):
                head.train()
                for batch in torch.randperm(len(train_x)).split(256):
                    objective=loss(head(train_x[batch]),train_y[batch]);optimizer.zero_grad();objective.backward();optimizer.step()
                head.eval()
                with torch.no_grad():validation=float(loss(head(val_x),val_y))
                if validation<best:
                    best=validation;best_epoch=epoch+1;best_state={k:v.detach().clone() for k,v in head.state_dict().items()}
                if epoch+1-best_epoch>=25:break
            head.load_state_dict(best_state);head.eval()
            with torch.no_grad():prediction=probabilities(head(test_x))
            predictions.append(prediction);runs.append({'seed':seed,'best_epoch':best_epoch,'validation_loss':best,'test_metrics':metrics(prediction,test_y)})
            torch.save({'schema':'meal-outcome-head-v1','variant':variant,'inputs':train_x.shape[1],'state':head.state_dict(),
                        'mean':mean,'std':std,'source_model_sha256':metadata['frozen_model_sha256'],'horizons':metadata['horizons'],
                        'context_names':metadata['context_names'],'outcome_names':metadata['outcome_names']},args.data/f'{variant}-{seed}.pt')
        ensemble=torch.stack(predictions).mean(0)
        result['variants'][variant]={'ensemble_test':metrics(ensemble,test_y),'ranking':ranking(ensemble,test_y,groups),'runs':runs,
                                     'train_samples':len(train_x),'val_samples':len(val_x),'test_samples':len(test_x),
                                     'by_reason':{reason:metrics(ensemble[mask],test_y[mask]) for reason in ('first_injury','first_observed_retreat','renewed_injury') if (mask:=torch.tensor([records[i]['reason']==reason for i,_ in groups])).any()},
                                     'by_response':{mode:metrics(ensemble[mask],test_y[mask]) for action,mode in enumerate(metadata['modes']) if (mask:=torch.tensor([a==action for _,a in groups])).any()}}
        print(variant,json.dumps(result['variants'][variant]['ensemble_test']['12']),flush=True)
    # Proper scoring baseline: training outcomes averaged by continuation choice.
    train_x,train_y,_=tensors(records,'train','context_only');_,test_y,groups=tensors(records,'test','context_only')
    means=torch.stack([train_y[train_x[:,-3+i]==1].mean(0) for i in range(3)])
    baseline=torch.stack([means[action] for _,action in groups])
    result['action_mean_baseline']={'test_metrics':metrics(baseline,test_y),'ranking':ranking(baseline,test_y,groups),
        'by_reason':{reason:metrics(baseline[mask],test_y[mask]) for reason in ('first_injury','first_observed_retreat','renewed_injury') if (mask:=torch.tensor([records[i]['reason']==reason for i,_ in groups])).any()}}
    # Stronger simple baseline: measured incoming damage present/absent plus
    # response choice, with all means estimated on training data only.
    test_x,_,_=tensors(records,'test','context_only')
    cue_means={}
    for damaged in (False,True):
        for action in range(3):
            mask=(train_x[:,2]>0)==damaged
            mask=mask & (train_x[:,-3+action]==1)
            cue_means[damaged,action]=train_y[mask].mean(0) if mask.any() else means[action]
    cue=torch.stack([cue_means[bool(test_x[i,2]>0),action] for i,(_,action) in enumerate(groups)])
    result['damage_cue_baseline']={'test_metrics':metrics(cue,test_y),'ranking':ranking(cue,test_y,groups),
        'by_reason':{reason:metrics(cue[mask],test_y[mask]) for reason in ('first_injury','first_observed_retreat','renewed_injury') if (mask:=torch.tensor([records[i]['reason']==reason for i,_ in groups])).any()},
        'note':'Training mean grouped only by measured incoming damage present/absent and proposed response; no hidden behavior labels'}
    result['always_no_attack_brier']={str(h):float(test_y[:,i,2].mean()) for i,h in enumerate((3,6,12))}
    result['limits']=['Disjoint seeds and unseen return delay; same rooms and species',
                      'Predicted outcomes describe committing to scripted continuations, not a learned new actor',
                      'Outcome supervision is JEPA-inspired world-model work, not a joint-embedding JEPA objective',
                      'Fixed 36-input encoder does not receive hunger; context variants supply eight private measurements directly to predictor head',
                      'The live controller is unchanged; no utility weights or combined cost were fitted']
    (args.data/'training_report.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
