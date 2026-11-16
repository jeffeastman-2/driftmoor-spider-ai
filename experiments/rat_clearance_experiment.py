"""Paired rat clearance-input experiment; reuse the same collected corpus.

Only the rat architecture changes. Other species weights, optimizer seed,
training corpus and epoch budget are held fixed. Outputs stay outside git.
"""
import argparse
import json
import shutil
from pathlib import Path
import torch
from experiments.train_ecology_djepa import train,evaluate as evaluate_joint
from experiments.evaluate_ecology_isolated import evaluate
from spider_sim.djepa_ecology import CANDIDATE_SIZE


def run(source,output,epochs=300):
    output.mkdir(exist_ok=False)
    control=output/'control';control.mkdir()
    for species in ('rat','spider','shadowbeast'):
        shutil.copy2(source/(species+'.pt'),control/(species+'.pt'))
        if species!='rat':shutil.copy2(source/(species+'.pt'),output/(species+'.pt'))
    training=json.loads((source/'rat-training.json').read_text())
    validation=json.loads((source/'rat-validation.json').read_text())
    report={'source':str(source.resolve()),'change':'Rat action-relative clearance: 18 to 19 candidate values',
            'held_fixed':'Corpus, optimizer seed, epoch budget, other species weights',
            'epochs':epochs,'rat':train('rat',training,validation,output,epochs,CANDIDATE_SIZE+1)}
    report['control']=evaluate(control,range(360000,360016),species_list=('rat',))
    report['treatment']=evaluate(output,range(360000,360016),species_list=('rat',))
    report['joint_seeds']=list(range(370000,370008))
    report['encounters']=evaluate_joint(output,range(370000,370008))
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=300)
    args=parser.parse_args()
    if args.epochs<1:parser.error('Epoch budget must be positive')
    torch.set_num_threads(1);run(args.source,args.output,args.epochs)
