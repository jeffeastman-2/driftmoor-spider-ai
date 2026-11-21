"""Selected-meal interface regression and frozen-model engine comparison."""
import argparse,gzip,hashlib,json
from collections import Counter
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from spider_sim.learned_lifecycle import load
from experiments.train_lifecycle_model import load_corpus,evaluate
from experiments.report_recovery_learning import refusal_sequences,REFUSALS


def snapshot_audit(path):
    checked=0;violations=0
    with gzip.open(path,'rt') as f:
        for line in f:
            row=json.loads(line)
            if row['command']['action'] not in ('wrap','feed'):continue
            selected=row['observation'].get('selected_meal')
            checked+=1
            if selected is None or selected['track']!=row['command'].get('target'):violations+=1
    return {'food_commands_checked':checked,'snapshot_motor_target_disagreements':violations}


def main():
    p=argparse.ArgumentParser();p.add_argument('artifact_root',type=Path);p.add_argument('--summary',type=Path,required=True);p.add_argument('--offline-only',action='store_true');a=p.parse_args();b=a.artifact_root
    torch.set_num_threads(1);weights=b/'target-bound-model-v3/gru.pt';digest=hashlib.sha256(weights.read_bytes()).hexdigest()
    cache=b/'target-binding-offline-v3.json';offline=json.loads(cache.read_text()) if cache.exists() else None
    if offline is None or offline['model_sha256']!=digest:
        model=load(weights);sequences=load_corpus(b/'recovery-teacher-corpus-v2','v3',only_splits=('test',))['test']
        checked=sum(r.get('command',{}).get('action') in ('wrap','feed') for s in sequences for r in s['rows'])
        for s in sequences:s.pop('rows',None)
        offline={'model_sha256':digest,'feature_count':152,'test':evaluate(model,sequences),'causally_checked_heldout_food_commands':checked}
        cache.write_text(json.dumps(offline,indent=2)+'\n')
    if a.offline_only:
        a.summary.write_text(json.dumps(offline,indent=2)+'\n');print(json.dumps({k:offline['test'][k] for k in ('accuracy','macro_f1_present')},indent=2));return
    root=b/'target-binding-closed-loop-v3';manifest=json.loads((root/'manifest.json').read_text());runs=json.loads((root/'results.json').read_text())
    assert {(r['seed'],r['mode']) for r in runs}=={(seed,mode) for seed in manifest['seeds'] for mode in ('teacher','previous','bound')}
    assert manifest['model_sha256']['bound']==offline['model_sha256']
    assert manifest['model_sha256']['previous']==hashlib.sha256((b/'recovery-corrected-v2/gru.pt').read_bytes()).hexdigest()
    assert all(r['all_spider_policies_learned'] for r in runs if r['mode']!='teacher')
    assert len(runs)==18 and all(r['turns']==1000 and r['current_profile'] for r in runs)
    totals={}
    for mode in ('teacher','previous','bound'):
        rs=[r for r in runs if r['mode']==mode];statuses=Counter();decisions=sum(r['decisions'] for r in rs)
        for r in rs:
            case=root/f"{mode}-{r['seed']}";statuses.update(r['statuses'])
            r['refusal_audit']=refusal_sequences(case/'decisions.jsonl.gz');r['binding_audit']=snapshot_audit(case/'decisions.jsonl.gz')
            trace=[json.loads(l) for l in (case/'population.jsonl').read_text().splitlines()]
            r['mean_last200']=sum(x['spiders'] for x in trace[-200:])/200
        totals[mode]={'runs':len(rs),'groups_above_pair':sum(r['living_spiders']>2 for r in rs),'any_survivors':sum(r['living_spiders']>0 for r in rs),
            'reproducing_runs':sum(r['eggs']>0 for r in rs),'living_spiders':sum(r['living_spiders'] for r in rs),'egg_sacs':sum(r['eggs'] for r in rs),
            'mean_last200_per_seed':sum(r['mean_last200'] for r in rs)/len(rs),'decisions':decisions,
            'food_refusals':statuses['food_not_reachable']+statuses['food_not_wrapped'],
            'lifecycle_refusals_per1000':1000*sum(statuses[s] for s in REFUSALS)/max(1,decisions),
            'food_refusals_per1000':1000*(statuses['food_not_reachable']+statuses['food_not_wrapped'])/max(1,decisions),
            'mean_seed_food_refusals_per1000':sum(1000*(r['statuses'].get('food_not_reachable',0)+r['statuses'].get('food_not_wrapped',0))/max(1,r['decisions']) for r in rs)/len(rs),
            'waits_per1000':1000*statuses['waited']/max(1,decisions),
            'unexecutable_goals':sum(r['unexecutable_goals'] for r in rs),
            'longest_wait_streak':max(r['refusal_audit']['longest_wait_streak'] for r in rs),
            'longest_hungry_food_wait_streak':max(r['refusal_audit']['longest_hungry_food_wait_streak'] for r in rs),
            'longest_same_target_refusal_streak':max(r['refusal_audit']['longest_same_target_refusal_streak'] for r in rs),
            'far_retained_with_nearer_food':sum(r['refusal_audit']['far_retained_meal_with_nearer_reachable_food'] for r in rs),
            'far_retained_outside_nearest_two':sum(r['refusal_audit']['far_retained_meal_not_in_nearest_two'] for r in rs),
            'snapshot_motor_target_disagreements':sum(r['binding_audit']['snapshot_motor_target_disagreements'] for r in rs),
            'food_commands_checked':sum(r['binding_audit']['food_commands_checked'] for r in rs),'statuses':dict(statuses)}
    result={'schema':'meal-target-binding-v3','plan':json.loads((b/'target-binding-plan-v3.json').read_text()),'offline':offline,
        'known_failure_probe':json.loads((b/'target-binding-diagnostic-v3.json').read_text()),'manifest':manifest,'totals':totals,'runs':runs,
        'limitations':['Six seeds and one training initialization; descriptive, not a statistical architecture ranking.',
            'Same training corpus/recipe/seed, but changed input shape changes parameter initialization stream.',
            'Snapshot consistency is an interface guarantee, not a feasibility mask or guarantee that goals are correct.',
            'Both learned modes log selected-meal snapshots; only the v3 actor encodes that snapshot.',
            'Meal preference remains authored and only currently perceived meals are selected; mate/attacker binding is outside this fix.',
            'No live viewer restart, online training, new planner or automatic deployment.']}
    a.summary.write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(2,3,figsize=(12,7),sharex=True,constrained_layout=True)
    colors={'teacher':'#777777','previous':'#2563eb','bound':'#15803d'};labels={'teacher':'Teacher','previous':'Previous learned','bound':'Bound meal target'}
    for ax,seed in zip(axes.flat,manifest['seeds']):
        for mode in colors:
            trace=[json.loads(l) for l in (root/f'{mode}-{seed}'/'population.jsonl').read_text().splitlines()]
            ax.plot([r['turn'] for r in trace],[r['spiders'] for r in trace],label=labels[mode],color=colors[mode],lw=1.5)
        ax.set_title(f'Seed {seed}');ax.set_ylim(bottom=0);ax.grid(alpha=.2);ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0,0].legend(fontsize=8);fig.supxlabel('Turn');fig.supylabel('Living spiders; panel scales vary');fig.suptitle('Selected-meal binding — six fresh seeds, unchanged engine rules')
    fig.savefig(root/'population-comparison.png',dpi=150);print(json.dumps(totals,indent=2))

if __name__=='__main__':main()
