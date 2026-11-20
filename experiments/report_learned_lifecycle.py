"""Summarize offline imitation and closed-loop outcomes without conflating them."""
import argparse,json
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('artifact_root',type=Path);p.add_argument('--summary',type=Path,required=True);a=p.parse_args();b=a.artifact_root
    imitation=json.loads((b/'lifecycle-response-imitation-v1/results.json').read_text())
    runs=json.loads((b/'lifecycle-closed-loop-v1/results.json').read_text())
    diagnostic=json.loads((b/'lifecycle-closed-loop-diagnostic-v1/results.json').read_text())
    assert len(runs)==16 and len(diagnostic)==4
    totals={}
    for mode in ('teacher','student'):
        rows=[r for r in runs if r['mode']==mode];statuses=Counter()
        for r in rows:statuses.update(r['statuses'])
        totals[mode]={'surviving_populations':sum(r['living_spiders']>0 for r in rows),'egg_sacs':sum(r['eggs'] for r in rows),
            'food_consumed':sum(r['food_consumed'] for r in rows),'decisions':sum(r['decisions'] for r in rows),'statuses':dict(statuses)}
    result={'schema':'learned-lifecycle-closed-loop-v1','offline':{k:{'best_epoch':v['best_epoch'],'parameters':v['parameters'],'test':v['test'],'validation':v['validation']} for k,v in imitation['models'].items()},
        'prospective_runs':runs,'diagnostic_runs':diagnostic,'totals':totals,
        'interpretation':'Offline teacher agreement is not closed-loop competence. Student goals are unmasked argmax; no teacher goal fallback. Motor/target controllers and engine are authored. Teacher and student trajectories and random event consumption diverge after changed actions. Eight new seeds are descriptive, not a significance claim. Diagnostic seeds overlap imitation corpus.'}
    a.summary.write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(2,4,figsize=(13,6),sharex=True,sharey=True,constrained_layout=True)
    for ax,seed in zip(axes.flat,sorted({r['seed'] for r in runs})):
        for mode,color in [('teacher','#555555'),('student','#1769c1')]:
            rows=[json.loads(l) for l in (b/'lifecycle-closed-loop-v1'/f'{mode}-{seed}'/'population.jsonl').read_text().splitlines()]
            ax.plot([r['turn'] for r in rows],[r['spiders'] for r in rows],label=mode,color=color)
        ax.set_title(f'Seed {seed}');ax.grid(alpha=.2)
    axes[0,0].legend();fig.supxlabel('Turn');fig.supylabel('Living spiders');fig.suptitle('Learned lifecycle selector vs response teacher — eight new seeds')
    fig.savefig(b/'lifecycle-closed-loop-v1/population-comparison.png',dpi=150)
    print(json.dumps(totals,indent=2))

if __name__=='__main__':main()
