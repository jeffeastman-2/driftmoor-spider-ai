"""Continue pursuit training on student-visited states; select by validation catches."""
import argparse
import json
from pathlib import Path
import torch
from experiments.train_single_prey import collect,fit,evaluate
from spider_sim.pursuit import load,save,SCHEMA


def run(source,output,epochs=300,passes=2):
    output.mkdir(exist_ok=False);training=json.loads((source/'training.json').read_text())
    validation=json.loads((source/'validation.json').read_text());model=load(source/'spider.pt')
    best=None;best_key=None;history=[]
    for phase in range(passes+1):
        if phase:
            added=collect(range(550000+phase*1000,550024+phase*1000),student=model,candidate_size=model.action_size)
            training+=added;model,loss=fit(training,validation,model,epochs=epochs)
        metrics=evaluate(model,range(560000,560012))
        caught=sum(row['neural']['caught'] for row in metrics.values())
        rejected=sum(row['neural']['attacks']-row['neural']['admitted'] for row in metrics.values())
        loops=sum(row['neural']['return_loops'] for row in metrics.values())
        key=(caught,-rejected,-loops);history.append({'phase':phase,'metrics':metrics})
        save(model,output/(f'phase-{phase}.pt'))
        if best_key is None or key>best_key:
            best_key=key;best=phase;save(model,output/'spider.pt')
    report={'schema':SCHEMA,'source':str(source),'aggregation_bug_fixed':True,'selection':'Validation catches, then rejected attacks, then return loops',
            'validation_seeds':list(range(560000,560012)),'selected_phase':best,'history':history,
            'test_seeds':list(range(570000,570032)),'evaluation':evaluate(load(output/'spider.pt'),range(570000,570032)),
            'limitations':['One optimizer seed; familiar Level-2 rooms only','One-step privileged engine supervision is offline only; no future states at inference',
                           'Authored target commitment with bounded dead reckoning; motor choice is learned',
                           'Scalar goal-consequence model, not a full latent world model or joint-embedding JEPA objective',
                           'Baseline continuous headings differ from the neural discrete motor inventory','No live controller replacement unless benchmark warrants it']}
    (output/'training.json').write_text(json.dumps(training));(output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=300);parser.add_argument('--passes',type=int,default=2)
    args=parser.parse_args()
    if args.epochs<1 or args.passes<0:parser.error('Positive epochs and nonnegative passes required')
    torch.set_num_threads(1);run(args.source,args.output,args.epochs,args.passes)
