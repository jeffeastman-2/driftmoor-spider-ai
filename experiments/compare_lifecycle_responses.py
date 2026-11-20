"""Matched-seed descriptive comparison; no model tuning or significance claim."""
import argparse,gzip,json
from pathlib import Path


def first_egg(root):
    with gzip.open(root/'decisions.jsonl.gz','rt') as f:
        for line in f:
            row=json.loads(line)
            if row['result'].get('egg_sac'):return row['turn']
    return None


def compare(before,after):
    rows=[]
    for new in json.loads((after/'collection-summary.json').read_text()):
        seed=new['seed'];old=json.loads((before/f'seed-{seed}'/'summary.json').read_text())
        assert old['turns']==new['turns'] and old['model_sha256']==new['model_sha256']
        assert old['rules']==new['rules']
        rows.append({'seed':seed,'turns':new['turns'],
            'before':{'first_egg_turn':first_egg(before/f'seed-{seed}'),'living_spiders':old['living_spiders'],'egg_sacs':old['events'].get('egg_sac_laid',0)},
            'after':{'first_egg_turn':first_egg(after/f'seed-{seed}'),'living_spiders':new['living_spiders'],'egg_sacs':new['events'].get('egg_sac_laid',0)}})
    return {'schema':'lifecycle-courtship-response-v1','runs':rows,
        'before_extinctions':sum(r['before']['living_spiders']==0 for r in rows),
        'after_extinctions':sum(r['after']['living_spiders']==0 for r in rows),
        'limitations':'Same twelve diagnostic seeds used to identify failures. Descriptive regression comparison, not an independent test of generalization. Dynamics diverge after changed actions, including random event consumption. Only fixed pursuit weights are used; response rules are authored.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('before',type=Path);p.add_argument('after',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=compare(a.before,a.after);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
