"""Compare complete diagnostic-seed population runs against checkpoint teacher."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def main():
    p=argparse.ArgumentParser();p.add_argument('artifact_root',type=Path);p.add_argument('--summary',type=Path,required=True);a=p.parse_args()
    b=a.artifact_root;root=b/'nutrition-recovery-10-30-v1'
    new=json.loads((root/'results.json').read_text());old=json.loads((b/'lifecycle-response-v1/collection-summary.json').read_text())
    assert json.loads((root/'manifest.json').read_text())['weights_sha256']==json.loads((b/'lifecycle-response-v1/manifest.json').read_text())['model_sha256']
    assert len(new)==12 and {r['seed'] for r in new}=={r['seed'] for r in old}
    indexed={r['seed']:r for r in old};comparisons=[]
    fig,axes=plt.subplots(3,4,figsize=(14,9),sharex=True,constrained_layout=True)
    for ax,r in zip(axes.flat,sorted(new,key=lambda r:r['seed'])):
        seed=r['seed'];prior=indexed[seed];traces={}
        for label,path,color in [('Checkpoint teacher',b/'lifecycle-response-v1'/f'seed-{seed}','#777777'),('Recovery + 30% cutoff',root/f'seed-{seed}','#15803d')]:
            rows=[json.loads(l) for l in (path/'population.jsonl').read_text().splitlines()];traces[label]=rows
            assert len(rows)==r['turns']==prior['turns']
            ax.plot([x['turn'] for x in rows],[x['spiders'] for x in rows],label=label,color=color,lw=1.5)
        ax.set_title(f"Seed {seed}: {prior['living_spiders']} → {r['living_spiders']} alive")
        ax.set_ylim(bottom=0);ax.grid(alpha=.2);ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        comparisons.append({'seed':seed,'before':{'living_spiders':prior['living_spiders'],'egg_sacs':prior['events'].get('egg_sac_laid',0),'starvation_deaths':prior['events'].get('starvation_death',0),
            'mean_population_last_200':sum(x['spiders'] for x in traces['Checkpoint teacher'][-200:])/200},
            'after':{**r,'mean_population_last_200':sum(x['spiders'] for x in traces['Recovery + 30% cutoff'][-200:])/200}})
    axes[0,0].legend(fontsize=8);fig.supxlabel('Turn (6 seconds each)');fig.supylabel('Living spiders; panel scales vary')
    fig.suptitle('Food-supported recovery below 10%, mating cutoff 30%\nSame twelve seeds; full 1,000-turn response-teacher episodes')
    fig.savefig(root/'population-comparison.png',dpi=150)
    totals={}
    for key in ('before','after'):
        rs=[c[key] for c in comparisons]
        totals[key]={'living_populations':sum(r['living_spiders']>0 for r in rs),
            'populations_above_pair_at_end':sum(r['living_spiders']>2 for r in rs),
            'lone_spider_runs':sum(r['living_spiders']==1 for r in rs),
            'reproducing_populations':sum((r.get('egg_sacs',r.get('events',{}).get('egg_sac_laid',0)))>0 for r in rs),
            'living_spiders':sum(r['living_spiders'] for r in rs),
            'egg_sacs':sum(r.get('egg_sacs',r.get('events',{}).get('egg_sac_laid',0)) for r in rs),
            'starvation_deaths':sum(r.get('starvation_deaths',r.get('events',{}).get('starvation_death',0)) for r in rs)}
    result={'schema':'nutrition-recovery-10-30-v1','manifest':json.loads((root/'manifest.json').read_text()),'totals':totals,'runs':comparisons,
        'limitations':'Same diagnostic seeds, authored response teacher with fixed learned pursuit. Combined recovery/cutoff intervention; no attribution of individual effects. Final population is not evidence of long-term equilibrium. Defaults and live viewer unchanged.'}
    a.summary.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(totals,indent=2))

if __name__=='__main__':main()
