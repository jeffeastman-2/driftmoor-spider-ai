"""Offline lifecycle imitation. Whole-seed heldouts; never installs live weights."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import random
import time
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from spider_sim.meal_target import select_meal
from spider_sim.lifecycle_model import LifecycleModel,GOALS,FEATURE_NAMES,features,feature_names,encode_observation


def load_corpus(directory,feature_version='v1',only_splits=None):
    manifest=json.loads((Path(directory)/'manifest.json').read_text())
    splits={}
    for split,seeds in manifest['splits'].items():
        if only_splits is not None and split not in only_splits:continue
        sequences=[]
        for seed in seeds:
            actors=defaultdict(list)
            with gzip.open(Path(directory)/f'seed-{seed}'/'decisions.jsonl.gz','rt') as f:
                for line in f:
                    row=json.loads(line)
                    assert row['seed']==seed and row['split']==split
                    actors[row['actor']].append(row)
            for actor,rows in actors.items():
                assert all(b['turn']==a['turn']+1 for a,b in zip(rows,rows[1:]))
                previous='hunt';remembered_meal=None;encoded=[]
                for row in rows:
                    obs=dict(row['observation'],self={**row['observation']['self'],'last_goal':previous})
                    selected=None
                    if feature_version=='v3':
                        selected=select_meal(obs,remembered_meal)
                        recorded=obs.get('selected_meal',selected)
                        if (recorded or {}).get('track')!=(selected or {}).get('track'):
                            raise ValueError('Recorded and causal selected-meal context disagree')
                        obs['selected_meal']=selected
                        if row.get('command',{}).get('action') in ('wrap','feed'):
                            if selected is None or row['command'].get('target')!=selected['track']:
                                raise ValueError('Causal meal target disagrees with recorded motor target')
                    encoded.append(encode_observation(obs,feature_version))
                    actual=row.get('executed_goal',row['teacher_goal'])
                    if actual in ('approach_meal','wrap','feed') and selected:
                        remembered_meal=selected['track']
                    elif actual=='hunt' and 'executed_goal' not in row and row.get('controller','teacher')=='teacher':
                        remembered_meal=None
                    previous=actual
                sequences.append({'seed':seed,'actor':actor,'rows':rows,
                    'x':torch.tensor(encoded),
                    'y':torch.tensor([GOALS.index(r['teacher_goal']) for r in rows])})
        splits[split]=sequences
    return splits


def batches(sequences,batch_size=16,shuffle=False):
    ordered=sorted(sequences,key=lambda s:len(s['y']))
    groups=[ordered[i:i+batch_size] for i in range(0,len(ordered),batch_size)]
    if shuffle:random.shuffle(groups)
    for group in groups:
        yield (pad_sequence([s['x'] for s in group],batch_first=True),
            torch.tensor([len(s['y']) for s in group]),
            pad_sequence([s['y'] for s in group],batch_first=True,padding_value=-100),group)


def classification(confusion):
    c=confusion.double();support=c.sum(1);predicted=c.sum(0);tp=c.diag()
    recall=tp/support.clamp_min(1);precision=tp/predicted.clamp_min(1)
    f1=2*precision*recall/(precision+recall).clamp_min(1e-12)
    present=support>0
    return {'accuracy':float(tp.sum()/support.sum().clamp_min(1)),
        'macro_f1_present':float(f1[present].mean()) if present.any() else 0.,
        'balanced_accuracy_present':float(recall[present].mean()) if present.any() else 0.,
        'per_goal':{g:{'support':int(support[i]),'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i])} for i,g in enumerate(GOALS)},
        'confusion_true_rows_predicted_columns':confusion.tolist()}


@torch.no_grad()
def evaluate(model,sequences):
    model.eval();conf=torch.zeros(len(GOALS),len(GOALS),dtype=torch.long);per_seed={}
    for x,lengths,y,group in batches(sequences):
        predicted=model(x,lengths).argmax(-1)
        for i,s in enumerate(group):
            n=len(s['y']);c=torch.bincount(y[i,:n]*len(GOALS)+predicted[i,:n],minlength=len(GOALS)**2).reshape(len(GOALS),len(GOALS))
            conf+=c;per_seed.setdefault(s['seed'],torch.zeros_like(conf)).add_(c)
    return {**classification(conf),'per_seed':{str(k):classification(v) for k,v in per_seed.items()}}


def train(splits,output,recurrent,epochs,seed=2026,feature_version='v1'):
    torch.manual_seed(seed);random.seed(seed)
    model=LifecycleModel(recurrent,feature_version=feature_version);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    counts=torch.bincount(torch.cat([s['y'] for s in splits['train']]),minlength=len(GOALS)).float()
    weights=(counts.sum()/counts.clamp_min(1)).sqrt();weights/=weights.mean()
    loss_fn=nn.CrossEntropyLoss(weight=weights,ignore_index=-100)
    best=-1;history=[];checkpoint=None;start=time.monotonic()
    for epoch in range(1,epochs+1):
        model.train();total=0.;steps=0
        for x,lengths,y,_ in batches(splits['train'],shuffle=True):
            opt.zero_grad();logits=model(x,lengths);loss=loss_fn(logits.reshape(-1,len(GOALS)),y.reshape(-1))
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();total+=loss.item();steps+=1
        validation=evaluate(model,splits['validation']);score=validation['macro_f1_present']
        history.append({'epoch':epoch,'training_batch_mean_loss':total/max(1,steps),'validation_macro_f1_present':score,'validation_accuracy':validation['accuracy']})
        if score>best:
            best=score;checkpoint={k:v.detach().clone() for k,v in model.state_dict().items()};best_epoch=epoch
        if epoch%5==0:print(json.dumps({'recurrent':recurrent,**history[-1]}),flush=True)
    model.load_state_dict(checkpoint)
    name='gru' if recurrent else 'snapshot'
    torch.save({'state_dict':model.state_dict(),'recurrent':recurrent,'features':feature_names(feature_version),'feature_version':feature_version,'goals':GOALS,'seed':seed},output/f'{name}.pt')
    return {'recurrent':recurrent,'parameters':sum(p.numel() for p in model.parameters()),'best_epoch':best_epoch,
        'seconds':time.monotonic()-start,'history':history,
        'validation':evaluate(model,splits['validation']),'test':evaluate(model,splits['test']) if 'test' in splits else None}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('corpus',type=Path);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--feature-version',choices=('v1','v2','v3'),default='v1')
    parser.add_argument('--recovery-corpus',type=Path)
    parser.add_argument('--recovery-repeat',type=int,default=4)
    parser.add_argument('--gru-only',action='store_true')
    parser.add_argument('--defer-test',action='store_true',help='Fit and select using train/validation; assess frozen checkpoint separately')
    args=parser.parse_args()
    if args.epochs<1 or args.recovery_repeat<1:parser.error('Positive epochs and recovery repeat required')
    torch.set_num_threads(1);args.output.mkdir(parents=True,exist_ok=True)
    splits=load_corpus(args.corpus,args.feature_version,only_splits=('train','validation') if args.defer_test else None)
    extra_count=0
    if args.recovery_corpus:
        extra=load_corpus(args.recovery_corpus,args.feature_version)
        if set(extra)!={'train'}:raise ValueError('Recovery corpus must contain training seeds only')
        if not {s['seed'] for s in extra['train']}<={s['seed'] for s in splits['train']}:raise ValueError('Recovery seed leakage')
        extra_count=sum(len(s['y']) for s in extra['train'])
        splits['train']=splits['train']+extra['train']*args.recovery_repeat
    for sequences in splits.values():
        for sequence in sequences:sequence.pop('rows',None)  # Encoded actor training does not need raw object graphs.
    result={'schema':'lifecycle-imitation-v1','corpus':str(args.corpus),'feature_names':feature_names(args.feature_version),'feature_version':args.feature_version,
        'recovery_corpus':str(args.recovery_corpus) if args.recovery_corpus else None,'recovery_repeat':args.recovery_repeat if args.recovery_corpus else 0,'unique_recovery_decisions':extra_count,'goals':GOALS,
        'architecture':f'{len(feature_names(args.feature_version))} -> 128 ReLU -> 64 ReLU -> optional GRU(64) -> 64 ReLU -> 9 logits',
        'selection':f'validation macro F1 on goals present in validation; fixed {args.epochs} epoch budget; whole seed splits',
        'epochs':args.epochs,'class_weight':'inverse square root training frequency, normalized mean 1',
        'counts':{k:{'actors':len(v),'decisions':sum(len(s['y']) for s in v),'seeds':sorted({s['seed'] for s in v})} for k,v in splits.items()},'models':{}}
    for recurrent in ((True,) if args.gru_only else (False,True)):
        result['models']['gru' if recurrent else 'snapshot']=train(splits,args.output,recurrent,args.epochs,feature_version=args.feature_version)
        (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:{'best_epoch':v['best_epoch'],'test_accuracy':v['test']['accuracy'] if v['test'] else None,'test_macro_f1':v['test']['macro_f1_present'] if v['test'] else None} for k,v in result['models'].items()},indent=2))

if __name__=='__main__':main()
