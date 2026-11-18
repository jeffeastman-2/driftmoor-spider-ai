"""Reconstruct private 44-input histories for the exact existing fork dataset.

Teacher trajectories are replayed, not relabeled or resampled. The original
36-input representation is checked at each origin before histories are saved.
"""
import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import torch
from experiments.compare_meal_defense import setup
from experiments.meal_forecast_data import begin_round,private_context,SPIDER
from experiments.train_meal_forecast import load_records
from spider_sim.pursuit import load


def history_for_case(case,delay,model,origins):
    world,policies=setup(case,model,'guard');spider=policies[SPIDER]
    policies['freight_shadowbeast'].return_turns=delay
    measured=[];last_input=[]
    hook=model.encoder.register_forward_pre_hook(lambda module,args:last_input.append(args[0].detach().reshape(-1,36)[0].tolist()))
    result=[];by_turn={record['turn']:record for record in origins}
    try:
        for turn in range(max(by_turn)+1):
            obs=begin_round(world,policies)
            if obs is None:raise ValueError('Replay died before a saved decision')
            context=private_context(obs,spider.meal,spider.goal)
            command=spider.choose(obs)
            assert len(last_input)==1
            measured.append(last_input.pop()+context)
            if turn in by_turn:
                original=by_turn[turn]
                torch.testing.assert_close(spider.hunter.hidden[0,0],torch.tensor(original['latent']),atol=1e-6,rtol=1e-6)
                assert context==original['context']
                result.append(dict(original,history=[list(frame) for frame in measured]))
            world.act(SPIDER,command);world.elapsed_seconds+=6
    finally:hook.remove()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('data',type=Path)
    parser.add_argument('--model',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    metadata=json.loads((args.data/'metadata.json').read_text());sha=hashlib.sha256(args.model.read_bytes()).hexdigest()
    if sha!=metadata['frozen_model_sha256']:raise ValueError('Source pursuit checkpoint differs from fork dataset')
    records=load_records(args.data);groups=defaultdict(list)
    for record in records:groups[json.dumps(record['condition'],sort_keys=True)].append(record)
    model=load(args.model).requires_grad_(False);count=0;max_length=0
    with gzip.open(args.output/'histories.jsonl.gz','wt') as stream:
        for index,(key,origins) in enumerate(groups.items()):
            case=json.loads(key);delay=case.pop('return_delay')
            for record in history_for_case(case,delay,model,origins):
                stream.write(json.dumps(record)+'\n');count+=1;max_length=max(max_length,len(record['history']))
            if (index+1)%96==0:print(f'Replayed {index+1}/{len(groups)} conditions, {count} verified histories',flush=True)
    assert count==len(records)
    metadata.update(schema='meal-forecast-histories-v1',source_dataset=str(args.data),history_inputs=44,
                    observations_per_turn=44,original_inputs=36,context_inputs=8,
                    origins=count,max_history_length=max_length,replay_verified_all_origins=True)
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'{count} histories; longest {max_length} turns; all old latents and contexts match')

if __name__=='__main__':main()
