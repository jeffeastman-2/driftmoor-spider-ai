import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
parser=argparse.ArgumentParser();parser.add_argument('artifact_root',type=Path);parser.add_argument('--summary',type=Path,required=True);args=parser.parse_args()
base=args.artifact_root;corpus=base/'lifecycle-corpus-v1';out=base/'lifecycle-imitation-v1'
manifest=json.loads((corpus/'manifest.json').read_text());learned=json.loads((out/'results.json').read_text());forecast=json.loads((base/'lifecycle-predictor-v1/results.json').read_text());coverage=json.loads((corpus/'coverage-report.json').read_text());runs=json.loads((corpus/'collection-summary.json').read_text())
fig,axes=plt.subplots(3,1,figsize=(11,10),sharex=True,constrained_layout=True)
for ax,(split,seeds) in zip(axes,manifest['splits'].items()):
 for seed in seeds:
  rows=[json.loads(l) for l in (corpus/f'seed-{seed}'/'population.jsonl').read_text().splitlines()]
  ax.plot([r['turn'] for r in rows],[r['spiders'] for r in rows],label=f'Seed {seed}',linewidth=1.5)
 ax.set_ylabel(f'{split.title()}\nliving spiders');ax.set_ylim(bottom=0);ax.legend(ncol=4,fontsize=8);ax.grid(alpha=.2)
axes[-1].set_xlabel('Turn (6 seconds each)');fig.suptitle('Same controller and rules, twelve different seeds\n8 of 12 populations extinct by turn 1,000');fig.savefig(out/'population-seeds.png',dpi=160);plt.close(fig)
goals=learned['goals'];fig,ax=plt.subplots(figsize=(11,5),constrained_layout=True)
for offset,name,label in [(-.18,'snapshot','Current observation only'),(.18,'gru','Observation + GRU memory')]:
 ax.bar([i+offset for i in range(len(goals))],[learned['models'][name]['test']['per_goal'][g]['f1'] for g in goals],width=.36,label=label)
ax.set_xticks(range(len(goals)),goals,rotation=25,ha='right');ax.set_ylim(0,1);ax.set_ylabel('F1 score on held-out teacher decisions');ax.set_title('Memory helps lifecycle imitation; courtship remains weaker');ax.legend();ax.grid(axis='y',alpha=.2);fig.savefig(out/'goal-imitation.png',dpi=160);plt.close(fig)
compact={'schema':'lifecycle-learning-checkpoint-v1','runs':[{k:r[k] for k in ('seed','split','turns','decisions','actors','living_spiders','events')} for r in runs],
 'coverage':{k:coverage[k] for k in ('decisions','training_coverage_gaps','defense_sequences')},
 'imitation':{k:{n:v[n] for n in ('parameters','best_epoch','validation','test')} for k,v in learned['models'].items()},
 'forecasts':{k:{n:v[n] for n in ('validation','test')} for k,v in forecast['models'].items()}}
args.summary.write_text(json.dumps(compact,indent=2)+'\n')
