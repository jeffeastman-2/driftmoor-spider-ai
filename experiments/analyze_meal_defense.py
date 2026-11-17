"""Analyze paired outcomes; raw experiences remain outside the repository."""
import argparse,json,random,statistics
from collections import defaultdict
from pathlib import Path
parser=argparse.ArgumentParser(description='Paired meal-defense results with seed-level bootstrap intervals')
parser.add_argument('output',type=Path)
base=parser.parse_args().output
report=json.loads((base/'summary.json').read_text())
rows=[json.loads(line) for line in (base/'episodes.jsonl').read_text().splitlines()]
paired=defaultdict(dict)
for r in rows:paired[r['case_id']][r['mode']]=r
assert len(rows)==report['episodes'] and len(paired)==report['cases'] and all(set(p)=={'pursue','guard','resume'} for p in paired.values())

def comparison(mode,behavior=None):
 pairs=[p for p in paired.values() if behavior is None or p['pursue']['behavior']==behavior]
 bothfull=[p for p in pairs if p[mode]['became_full'] and p['pursue']['became_full']]
 byseed=defaultdict(list)
 for p in pairs:byseed[p['pursue']['seed']].append(int(p[mode]['became_full'])-int(p['pursue']['became_full']))
 means=[statistics.mean(v) for v in byseed.values()];rng=random.Random(8102)
 boot=sorted(statistics.mean(rng.choices(means,k=len(means))) for _ in range(2000))
 return {'cases':len(pairs),'candidate_full_only':sum(p[mode]['became_full'] and not p['pursue']['became_full'] for p in pairs),
         'pursuit_full_only':sum(p['pursue']['became_full'] and not p[mode]['became_full'] for p in pairs),
         'both_full_cases':len(bothfull),
         'mean_turn_difference_when_both_full':statistics.mean(p[mode]['turns_to_full']-p['pursue']['turns_to_full'] for p in bothfull) if bothfull else None,
         'fullness_rate_difference':statistics.mean(means),
         'fullness_difference_seed_bootstrap_95_interval':[boot[49],boot[1949]],
         'mean_damage_difference':statistics.mean(p[mode]['damage_taken']-p['pursue']['damage_taken'] for p in pairs)}
report['paired_against_pursue']={b or 'overall':{m:comparison(m,b) for m in ('guard','resume')} for b in (None,'current','first_hit','returner')}
report['by_initial_health']={str(h):{m:{'episodes':len(rr:=[r for r in rows if r['health']==h and r['mode']==m]),'survival_rate':statistics.mean(r['survived'] for r in rr),'fullness_rate':statistics.mean(r['became_full'] for r in rr)} for m in ('pursue','guard','resume')} for h in (.5,1.)}
report['uncertainty_note']=f'Intervals resample {len(report["seeds"])} seed-level paired mean differences; they do not measure generalization beyond these layouts and scripted opponents.'
(base/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'overall':report['overall'],'by_attacker':report['by_attacker'],'paired':report['paired_against_pursue']},indent=2))
