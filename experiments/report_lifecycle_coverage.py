"""Audit corpus coverage by whole-seed split, with rare-case sampling flags."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
from experiments.lifecycle_experience import GOALS

REQUIRED=('feeding_under_attack','defense_without_new_damage','food_and_receptive_partner',
    'hungry_and_critical_hp','sated_with_receptive_partner','web_courtship_signal',
    'previous_contact_missing','contact_reappeared','previous_meal_missing',
    'near_hunger_10','near_hunger_20','near_hp_35','cooldown_near_expiry')


def aggregate(directory):
    manifest=json.loads((directory/'manifest.json').read_text());combined={};splitcounts={};defenses=[];summaries=[]
    for split,seeds in manifest['splits'].items():
        counts=defaultdict(lambda:{'turns':0,'actor_episodes':0,'actors':0,'seeds':set()})
        for seed in seeds:
            root=directory/f'seed-{seed}'
            if not (root/'summary.json').exists():continue
            summary=json.loads((root/'summary.json').read_text())
            if not summary['completed']:continue
            summaries.append(summary)
            data=json.loads((root/'coverage.json').read_text())
            defenses.extend({**event,'split':split} for event in data['defense_examples'])
            for flag,entry in data['counts'].items():
                target=counts[flag]
                for k in ('turns','actor_episodes','actors'):target[k]+=entry[k]
                target['seeds'].update(entry['seeds'])
        splitcounts[split]={flag:{**entry,'seeds':sorted(entry['seeds'])} for flag,entry in counts.items()}
        for flag,entry in counts.items():
            target=combined.setdefault(flag,{'turns':0,'actor_episodes':0,'actors':0,'seeds':set()})
            for k in ('turns','actor_episodes','actors'):target[k]+=entry[k]
            target['seeds'].update(entry['seeds'])
    combined={flag:{**entry,'seeds':sorted(entry['seeds'])} for flag,entry in combined.items()}
    gaps=[]
    for flag in ['goal:'+g for g in GOALS]+list(REQUIRED):
        entry=splitcounts.get('train',{}).get(flag,{'turns':0,'actor_episodes':0,'seeds':[]})
        if len(entry['seeds'])<3 or entry['actor_episodes']<20:
            gaps.append({'case':flag,**entry,'reason':'fewer than 3 training seeds or 20 contiguous actor episodes'})
    # These are screening targets, not a statistical guarantee of sufficiency.
    report={'completed_seeds':sorted(s['seed'] for s in summaries),
        'expected_seeds':sum(map(len,manifest['splits'].values())),
        'decisions':sum(s['decisions'] for s in summaries),
        'counts':combined,'by_split':splitcounts,'training_coverage_gaps':gaps,
        'defense_sequences':{'total':len(defenses),
            'outcomes':dict(Counter(e['outcome'] for e in defenses)),
            'before_turn_50':sum(e['started']<=50 for e in defenses),
            'after_turn_50':sum(e['started']>50 for e in defenses),
            'seeds':sorted({e['seed'] for e in defenses}),
            'sexes':dict(Counter(e['sex'] for e in defenses)),
            'hunger_range':[min((e['hunger'] for e in defenses),default=None),max((e['hunger'] for e in defenses),default=None)],
            'hp_fraction_range':[min((e['hp_fraction'] for e in defenses),default=None),max((e['hp_fraction'] for e in defenses),default=None)],
            'examples':defenses},
        'interpretation':'Contiguous actor episodes are correlated segments, not independent examples. Thresholds are predeclared coverage screens. Gaps use training seeds only.'}
    return report


def markdown(report):
    lines=['# Lifecycle experience coverage','',
        f"Collected {len(report['completed_seeds'])}/{report['expected_seeds']} seeds and {report['decisions']:,} spider decisions.",'',
        'Counts below refer to contiguous per-actor segments, not independent encounters.','',
        '| Case | Turns | Actor episodes | Seeds |','| --- | ---: | ---: | ---: |']
    for flag in ['goal:'+g for g in GOALS]+list(REQUIRED):
        e=report['counts'].get(flag,{'turns':0,'actor_episodes':0,'seeds':[]})
        lines.append(f"| {flag} | {e['turns']} | {e['actor_episodes']} | {len(e['seeds'])} |")
    lines += ['','## Training coverage screens','',
        'Flag cases with fewer than three training seeds or 20 contiguous actor episodes. These are sampling targets, not proof of adequate learning.','']
    lines += [f"- {e['case']}: {e['actor_episodes']} episodes across {len(e['seeds'])} training seeds." for e in report['training_coverage_gaps']]
    d=report['defense_sequences']
    lines += ['','## Interrupted meals','',f"{d['total']} recorded defense sequences; outcomes: {d['outcomes']}.",
        f"{d['before_turn_50']} began by turn 50; {d['after_turn_50']} began later.",
        f"Sex counts: {d['sexes']}; hunger range: {d['hunger_range']}; HP fraction range: {d['hp_fraction_range']}.",'',
        'Returning to food means any food-handling goal, not necessarily the same carcass. Open sequences at episode end are right-censored, not failures.','']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    args=parser.parse_args();report=aggregate(args.directory)
    (args.directory/'coverage-report.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.directory/'COVERAGE.md').write_text(markdown(report))
    print(json.dumps({k:report[k] for k in ('completed_seeds','decisions','training_coverage_gaps')},indent=2))


if __name__=='__main__':main()
