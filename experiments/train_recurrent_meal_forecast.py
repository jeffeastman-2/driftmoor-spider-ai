"""Train copies of perception and GRU for private meal-defense outcome prediction.

44 inputs all pass through the encoder and GRU; none bypass memory into the
predictor. Original actor weights are never optimized or overwritten.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_sequence
from experiments.train_meal_forecast import OutcomeHead,loss,probabilities,metrics,ranking
from spider_sim.pursuit import load

VARIANTS={'retrained36':(36,False),'retrained44':(44,False),'retrained44_snapshot':(44,True)}


class RecurrentOutcome(nn.Module):
    def __init__(self,source,inputs=44,snapshot=False,head_seed=0):
        super().__init__();self.inputs=inputs;self.snapshot=snapshot
        self.encoder=nn.Sequential(nn.Linear(inputs,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU())
        self.memory=nn.GRU(32,48,batch_first=True)
        with torch.no_grad():
            self.encoder[0].weight.zero_()
            self.encoder[0].weight[:,:36].copy_(source.encoder[0].weight)
            self.encoder[0].bias.copy_(source.encoder[0].bias)
        self.encoder[2].load_state_dict(source.encoder[2].state_dict())
        self.memory.load_state_dict(source.memory.state_dict())
        with torch.random.fork_rng():
            torch.manual_seed(100000+head_seed);self.head=OutcomeHead(51)

    def represent(self,history,lengths):
        history=history[:,:,:self.inputs]
        if self.snapshot:
            history=history[torch.arange(len(history)),lengths-1].unsqueeze(1)
            lengths=torch.ones_like(lengths)
        embedded=self.encoder(history)
        packed=pack_padded_sequence(embedded,lengths.cpu(),batch_first=True,enforce_sorted=False)
        _,hidden=self.memory(packed)
        return hidden[-1]

    def forward(self,history,lengths):
        state=self.represent(history,lengths)
        choices=torch.eye(3,device=state.device).expand(len(state),-1,-1)
        combined=torch.cat((state[:,None,:].expand(-1,3,-1),choices),dim=-1)
        return self.head(combined.reshape(-1,51)).reshape(-1,3,3,4)


def read_records(directory):
    with gzip.open(directory/'histories.jsonl.gz','rt') as stream:return [json.loads(line) for line in stream]


def batch(records,indices):
    selected=[records[int(i)] for i in indices]
    lengths=torch.tensor([len(r['history']) for r in selected],dtype=torch.long)
    histories=pad_sequence([torch.tensor(r['history'],dtype=torch.float32) for r in selected],batch_first=True)
    targets=[]
    for record in selected:
        targets.append(torch.stack([torch.tensor([b['targets'] for b in record['branches'] if b['action']==a],dtype=torch.float32).mean(0) for a in range(3)]))
    return histories,lengths,torch.stack(targets)


@torch.no_grad()
def validation_loss(model,records,indices):
    model.eval();total=0.;count=0
    for chunk in torch.tensor(indices).split(64):
        history,lengths,target=batch(records,chunk)
        total+=float(loss(model(history,lengths).reshape(-1,3,4),target.reshape(-1,3,4)))*len(chunk)
        count+=len(chunk)
    return total/count


@torch.no_grad()
def evaluate(models,records,indices):
    predictions=[];targets=[];groups=[];reasons=[];seeds=[]
    for chunk in torch.tensor(indices).split(64):
        history,lengths,_=batch(records,chunk)
        averaged=torch.stack([probabilities(model(history,lengths).reshape(-1,3,4)).reshape(-1,3,3,4) for model in models]).mean(0)
        for local,index in enumerate(chunk.tolist()):
            record=records[index]
            for branch in record['branches']:
                predictions.append(averaged[local,branch['action']]);targets.append(branch['targets'])
                groups.append((index,branch['action']));reasons.append(record['reason']);seeds.append(record['seed'])
    p=torch.stack(predictions);y=torch.tensor(targets,dtype=torch.float32)
    return {'test_metrics':metrics(p,y),'ranking':ranking(p,y,groups),
            'by_reason':{reason:metrics(p[mask],y[mask]) for reason in sorted(set(reasons)) if (mask:=torch.tensor([r==reason for r in reasons])).any()},
            'by_test_seed':{str(seed):metrics(p[mask],y[mask]) for seed in sorted(set(seeds)) if (mask:=torch.tensor([s==seed for s in seeds])).any()},
            'cumulative_attack_monotonicity_violations':float(((p[:,0,2]>p[:,1,2])|(p[:,1,2]>p[:,2,2])).float().mean())}


def load_predictor(checkpoint,source_checkpoint):
    """Load a read-only predictor copy, refusing a different source contract."""
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    sha=hashlib.sha256(Path(source_checkpoint).read_bytes()).hexdigest()
    if payload.get('schema')!='recurrent-meal-forecast-v1' or payload.get('source_model_sha256')!=sha:
        raise ValueError('Recurrent predictor/source checkpoint mismatch')
    variant=payload.get('variant')
    if variant not in VARIANTS or VARIANTS[variant]!=(payload.get('inputs'),payload.get('snapshot')):
        raise ValueError('Recurrent predictor input contract mismatch')
    model=RecurrentOutcome(load(source_checkpoint),payload['inputs'],payload['snapshot'])
    model.load_state_dict(payload['state']);return model.eval().requires_grad_(False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('data',type=Path)
    parser.add_argument('--source-model',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=120);parser.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    source_sha=hashlib.sha256(args.source_model.read_bytes()).hexdigest()
    metadata=json.loads((args.data/'metadata.json').read_text())
    if source_sha!=metadata['frozen_model_sha256']:raise ValueError('Source checkpoint mismatch')
    source=load(args.source_model).requires_grad_(False);original={k:v.clone() for k,v in source.state_dict().items()}
    records=read_records(args.data)
    splits={split:[i for i,r in enumerate(records) if r['split']==split] for split in ('train','val','test')}
    seed_sets={split:{records[i]['seed'] for i in indices} for split,indices in splits.items()}
    assert all(not seed_sets[a]&seed_sets[b] for a,b in (('train','val'),('train','test'),('val','test')))
    report={'schema':'recurrent-meal-forecast-v1','metadata':metadata,'variants':{},'optimizer_seeds':[0,1,2],
            'task':'Supervised outcome prediction; original actor unchanged; no new live action selection',
            'loss_note':'Average rollout targets at each origin/choice; equivalent expected MSE/BCE gradients to repeated draws'}
    for variant in args.variants:
        inputs,snapshot=VARIANTS[variant];models=[];runs=[]
        for seed in range(3):
            torch.manual_seed(seed);model=RecurrentOutcome(source,inputs,snapshot,seed)
            optimizer=torch.optim.Adam([{'params':list(model.encoder.parameters())+list(model.memory.parameters()),'lr':.0003},
                                        {'params':model.head.parameters(),'lr':.002}])
            best=float('inf');best_epoch=0;best_state=None
            for epoch in range(args.epochs):
                model.train()
                order=torch.tensor(splits['train'])[torch.randperm(len(splits['train']))]
                for chunk in order.split(48):
                    history,lengths,target=batch(records,chunk)
                    raw=model(history,lengths).reshape(-1,3,4)
                    objective=loss(raw,target.reshape(-1,3,4));optimizer.zero_grad();objective.backward()
                    nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                value=validation_loss(model,records,splits['val'])
                if value<best:
                    best=value;best_epoch=epoch+1;best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
                if epoch+1-best_epoch>=20:break
            model.load_state_dict(best_state);model.eval();models.append(model)
            changes={'encoder':sum(float((v-model.encoder.state_dict()[k]).abs().sum()) for k,v in RecurrentOutcome(source,inputs,snapshot,seed).encoder.state_dict().items()),
                     'gru':sum(float((v-model.memory.state_dict()[k]).abs().sum()) for k,v in source.memory.state_dict().items())}
            torch.save({'schema':'recurrent-meal-forecast-v1','variant':variant,'inputs':inputs,'snapshot':snapshot,
                        'state':model.state_dict(),'source_model_sha256':source_sha,'context_names':metadata['context_names'],
                        'outcome_names':metadata['outcome_names'],'horizons':metadata['horizons']},args.output/f'{variant}-{seed}.pt')
            runs.append(dict(seed=seed,best_epoch=best_epoch,validation_loss=best,weight_changes_l1=changes))
            print(f'{variant} seed {seed}: best epoch {best_epoch}, validation {best:.4f}',flush=True)
        evaluation=evaluate(models,records,splits['test'])
        report['variants'][variant]={'inputs':inputs,'snapshot_only':snapshot,'bypass_context':False,'trainable_parameters':sum(p.numel() for p in models[0].parameters()),'runs':runs,**evaluation}
        print(variant,json.dumps(evaluation['test_metrics']['12']),flush=True)
    assert all(torch.equal(v,source.state_dict()[k]) for k,v in original.items())
    assert hashlib.sha256(args.source_model.read_bytes()).hexdigest()==source_sha
    report['original_actor_unchanged']=True
    report['limits']=['Same existing outcomes and split as the frozen experiment; labels were not recollected',
                      'Test set was already examined in the previous experiment, so this is comparative research, not a fresh final acceptance test',
                      'New encoder/GRU belong only to the predictor copy; the actor scorer was not retrained',
                      'Outcome heads supervise explicit consequences, not JEPA latent targets',
                      'Choices commit to existing authored continuations; forecasts alone do not establish improved engine behavior']
    (args.output/'training_report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
