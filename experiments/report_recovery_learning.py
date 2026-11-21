"""Frozen models: copying diagnostics and fresh closed-loop recovery comparison."""
import argparse,gzip,json
from collections import Counter
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from spider_sim.learned_lifecycle import load
from experiments.train_lifecycle_model import load_corpus,evaluate

REFUSALS={'food_not_reachable','mate_not_reachable','too_hungry_to_mate',
    'not_receptive','food_not_wrapped','incompatible_mate','not_touching_web'}


def refusal_sequences(path):
    previous={};segments=0;longest=0;waits={};food_waits={};longest_wait=0;longest_food_wait=0;food_wait_count=0;retained_conflicts=0;unencoded_retained=0;numeric_boundary=0
    with gzip.open(path,'rt') as f:
        for line in f:
            r=json.loads(line);a=r['actor'];t=r['turn']
            if r['result']['status']=='food_not_reachable':
                meals=sorted(r['observation'].get('food_in_view',[]),key=lambda m:m['range_ft'])
                selected=next(((i,m) for i,m in enumerate(meals) if m['track']==r['command'].get('target')),None)
                if selected:
                    i,target=selected
                    if 6<target['range_ft']<=6.000001:numeric_boundary+=1
                    if target['range_ft']>6 and meals and meals[0]['range_ft']<=6:retained_conflicts+=1
                    if target['range_ft']>6 and i>=2:unencoded_retained+=1
            waiting=r['result']['status']=='waited'
            food_waiting=waiting and r['observation']['self']['hunger']>=.3 and bool(r['observation'].get('food_in_view')) and not r['observation'].get('incoming_damage')
            for state,condition in ((waits,waiting),(food_waits,food_waiting)):
                old_wait=state.get(a);n_wait=old_wait[1]+1 if old_wait and old_wait[0]==t-1 else 1
                if condition:state[a]=(t,n_wait)
                else:state.pop(a,None)
            if waiting:longest_wait=max(longest_wait,waits[a][1])
            if food_waiting:
                food_wait_count+=1;longest_food_wait=max(longest_food_wait,food_waits[a][1])
            key=(r['command']['action'],r['command'].get('target'),r['result']['status'])
            if key[-1] not in REFUSALS:previous.pop(a,None);continue
            old=previous.get(a);n=old[2]+1 if old and old[0]==t-1 and old[1]==key else 1
            if n==3:segments+=1
            longest=max(longest,n);previous[a]=(t,key,n)
    return {'three_or_more_repeat_segments':segments,'longest_same_target_refusal_streak':longest,
        'longest_wait_streak':longest_wait,'waited_hungry_with_visible_food_no_new_damage':food_wait_count,
        'longest_hungry_food_wait_streak':longest_food_wait,
        'far_retained_meal_with_nearer_reachable_food':retained_conflicts,
        'far_retained_meal_not_in_nearest_two':unencoded_retained,'numerical_food_boundary_refusals':numeric_boundary}


def main():
    p=argparse.ArgumentParser();p.add_argument('artifact_root',type=Path);p.add_argument('--summary',type=Path,required=True);p.add_argument('--offline-only',action='store_true');a=p.parse_args();b=a.artifact_root;torch.set_num_threads(1)
    paths={'prior':b/'lifecycle-response-imitation-v1/gru.pt','demos':b/'recovery-encoder-demo-v2/gru.pt','recovery':b/'recovery-corrected-v2/gru.pt'}
    offline={};test_sets={}
    cache=b/'recovery-offline-v2.json'
    import hashlib
    hashes={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()}
    existing=json.loads(cache.read_text()) if cache.exists() else None
    if existing and existing.get('model_sha256')==hashes:offline=existing['metrics']
    for name,path in (paths.items() if not offline else []):
        model=load(path)
        if model.feature_version not in test_sets:
            sequences=load_corpus(b/'recovery-teacher-corpus-v2',model.feature_version,only_splits=('test',))['test']
            for sequence in sequences:sequence.pop('rows',None)
            test_sets[model.feature_version]=sequences
        offline[name]=evaluate(model,test_sets[model.feature_version])
        del model
    if not existing or existing.get('model_sha256')!=hashes:cache.write_text(json.dumps({'model_sha256':hashes,'metrics':offline},indent=2)+'\n')
    if a.offline_only:
        a.summary.write_text(json.dumps({'model_sha256':hashes,'metrics':offline},indent=2)+'\n')
        print(json.dumps({name:{k:v[k] for k in ('accuracy','macro_f1_present')} for name,v in offline.items()},indent=2));return
    corrections=json.loads((b/'student-recovery-corpus-v2/results.json').read_text());counts=Counter();statuses=Counter()
    for r in corrections:counts.update(r['corrections']);statuses.update(r['statuses'])
    root=b/'recovery-closed-loop-v2';runs=json.loads((root/'results.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
    assert len(runs)==len(manifest['seeds'])*4
    assert {(r['seed'],r['mode']) for r in runs}=={(seed,mode) for seed in manifest['seeds'] for mode in ('teacher','prior','demos','recovery')}
    assert all(r['current_profile'] and r['turns']==1000 for r in runs)
    totals={}
    for mode in ('teacher','prior','demos','recovery'):
        rs=[r for r in runs if r['mode']==mode];st=Counter();decisions=sum(r['decisions'] for r in rs)
        for r in rs:
            st.update(r['statuses']);r['repeat_refusals']=refusal_sequences(root/f"{mode}-{r['seed']}"/'decisions.jsonl.gz')
            trace=[json.loads(line) for line in (root/f"{mode}-{r['seed']}"/'population.jsonl').read_text().splitlines()]
            r['mean_population_last200']=sum(row['spiders'] for row in trace[-200:])/len(trace[-200:])
        totals[mode]={'runs':len(rs),'populations_above_pair':sum(r['living_spiders']>2 for r in rs),
            'any_survivors':sum(r['living_spiders']>0 for r in rs),'reproducing_runs':sum(r['eggs']>0 for r in rs),
            'mean_population_last200_across_runs':sum(r['mean_population_last200'] for r in rs)/len(rs),
            'living_spiders':sum(r['living_spiders'] for r in rs),'egg_sacs':sum(r['eggs'] for r in rs),'hatches':sum(r['hatches'] for r in rs),
            'decisions':decisions,'interaction_refusals':sum(st[s] for s in REFUSALS),
            'refusals_per_1000_decisions':1000*sum(st[s] for s in REFUSALS)/max(1,decisions),
            'mean_seed_refusals_per_1000':sum(1000*sum(r['statuses'].get(s,0) for s in REFUSALS)/max(1,r['decisions']) for r in rs)/len(rs),
            'longest_same_target_refusal_streak':max(r['repeat_refusals']['longest_same_target_refusal_streak'] for r in rs),
            'three_or_more_repeat_segments':sum(r['repeat_refusals']['three_or_more_repeat_segments'] for r in rs),
            'unexecutable_goals':sum(r['unexecutable_goals'] for r in rs),
            'waits_per_1000_decisions':1000*st['waited']/max(1,decisions),
            'longest_wait_streak':max(r['repeat_refusals']['longest_wait_streak'] for r in rs),
            'waited_hungry_with_visible_food_no_new_damage':sum(r['repeat_refusals']['waited_hungry_with_visible_food_no_new_damage'] for r in rs),
            'longest_hungry_food_wait_streak':max(r['repeat_refusals']['longest_hungry_food_wait_streak'] for r in rs),
            'far_retained_meal_with_nearer_reachable_food':sum(r['repeat_refusals']['far_retained_meal_with_nearer_reachable_food'] for r in rs),
            'far_retained_meal_not_in_nearest_two':sum(r['repeat_refusals']['far_retained_meal_not_in_nearest_two'] for r in rs),
            'numerical_food_boundary_refusals':sum(r['repeat_refusals']['numerical_food_boundary_refusals'] for r in rs),
            'statuses':dict(st),'all_spider_policies_learned':all(r['all_spider_policies_learned'] for r in rs) if mode!='teacher' else None}
    report={'schema':'recovery-learning-v2','plan':json.loads((b/'recovery-plan-v2.json').read_text()),
        'offline_current_teacher_test':offline,'corrections':{'unique_student_decisions':sum(r['decisions'] for r in corrections),
            'disagreements':sum(counts.values()),'most_common':counts.most_common(15),'executed_statuses':dict(statuses)},
        'fresh_manifest':manifest,'fresh_totals':totals,'fresh_runs':runs,
        'limitations':['One model initialization per condition; eight descriptive fresh seeds, no significance claim.',
            'Prior vs new models changes both encoder and teacher corpus. Demos vs recovery adds correction data with fourfold sampling and more optimizer steps per epoch; it is not a step-matched ablation.',
            'Corrective labels come from old99 student trajectories, not an iterative rollout of the newly fitted controller.',
            'Own previous intent is causal executed activity, never current teacher answer; all model actions are unmasked argmax.',
            'Shared authored target/motor controllers and engine remain; no trained world-model planner or learned cost in this test.',
            'Different decisions alter event order and random consumption. Aggregate event counts depend on population and lifespan.',
            'Retained-target mismatch counts are a post-hoc failure audit, not a prespecified primary outcome. No model was changed after inspecting fresh failures.',
            'Raw data and weights remain outside Git; no live deployment.']}
    a.summary.write_text(json.dumps(report,indent=2)+'\n')
    fig,axes=plt.subplots(2,4,figsize=(14,7),sharex=True,constrained_layout=True)
    colors={'teacher':'#777777','prior':'#d97706','demos':'#2563eb','recovery':'#15803d'}
    labels={'teacher':'Teacher','prior':'Old network','demos':'New inputs + demos','recovery':'With corrections'}
    for ax,seed in zip(axes.flat,manifest['seeds']):
        for mode in colors:
            rows=[json.loads(l) for l in (root/f'{mode}-{seed}'/'population.jsonl').read_text().splitlines()]
            ax.plot([r['turn'] for r in rows],[r['spiders'] for r in rows],label=labels[mode],color=colors[mode],lw=1.4)
        ax.set_title(f'Seed {seed}');ax.set_ylim(bottom=0);ax.grid(alpha=.2);ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0,0].legend(fontsize=8);fig.supxlabel('Turn');fig.supylabel('Living spiders; panel scales vary')
    fig.suptitle('Fresh-seed engine comparison — same healing, mating and defense rules')
    fig.savefig(root/'population-comparison.png',dpi=150)
    print(json.dumps(totals,indent=2))

if __name__=='__main__':main()
