"""Align private action records with the next decision, including enemy response.

Terminal/censored records use actual episode outcomes and no fabricated next
observation. Predictor training must distinguish terminal and horizon truncation.
"""
import argparse
import gzip
import json
from pathlib import Path


def align(directory):
    episodes={(r['case_id'],r['mode']):r for line in (directory/'episodes.jsonl').read_text().splitlines() if (r:=json.loads(line))}
    pending=None;count=0;ends=0
    def finish(row,following=None):
        nonlocal count,ends
        row.pop('timing',None)
        case=episodes[row['case'],row['mode']]
        row['terminal']=False;row['truncated']=False
        if following is not None:
            row['next_observation']=following['observation']
            row['next_observation_timing']='next spider decision, after opponent response'
        else:
            row['next_observation']=None
            row['terminal']=not case['survived'] or case['became_full']
            row['truncated']=not row['terminal']
            row['terminal_outcome']={'hp':case['final_hp'],'hunger':case['final_hunger'],'survived':case['survived'],'became_full':case['became_full']}
            row['next_observation_timing']='episode ended; no next decision observation'
            ends+=1
        destination.write(json.dumps(row)+'\n');count+=1
    with gzip.open(directory/'transitions.jsonl.gz','rt') as source,gzip.open(directory/'decision_transitions.jsonl.gz','wt') as destination:
        for line in source:
            row=json.loads(line)
            if pending is not None:
                same=(row['case'],row['mode'])==(pending['case'],pending['mode'])
                finish(pending,row if same else None)
            pending=row
        if pending is not None:finish(pending)
    assert ends==len(episodes),(ends,len(episodes))
    return {'transitions':count,'episodes':ends,'path':str(directory/'decision_transitions.jsonl.gz'),
            'note':'Use aligned next-decision records for learning; raw post-round records omit the next opponent response.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    print(json.dumps(align(parser.parse_args().output),indent=2))
