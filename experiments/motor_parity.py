"""Compare a frozen learner with continuous and motor-matched scripted control."""
import argparse
import json
import math
from pathlib import Path
import torch
from spider_sim.ecology import GoalBaseline
from spider_sim.djepa_ecology import candidates
from spider_sim.pursuit import load


class MotorMatchedBaseline:
    """Project the existing script onto the learner's exact motor inventory.

    No new planning, collision/reach checks or privileged observations.
    """
    def __init__(self,seed=0):self.script=GoalBaseline(seed=seed)
    def choose(self,obs):
        requested=self.script.choose(obs)
        if requested['action']!='move':return requested
        choices=[c for c in candidates(obs) if c['action']=='move']
        angle=math.radians(requested['heading_deg']);distance=requested['distance_ft']
        desired=(distance*math.cos(angle),distance*math.sin(angle))
        def error(command):
            angle=math.radians(command['heading_deg']);distance=command['distance_ft']
            return math.dist(desired,(distance*math.cos(angle),distance*math.sin(angle)))
        return min(choices,key=error)


def main():
    from experiments.train_single_prey import evaluate
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--approach',action='store_true',help='Compare the optional precise approach candidate with unchanged neural weights')
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a new report path')
    torch.set_num_threads(1)
    start=620000 if args.approach else 610000
    seeds=list(range(start,start+32))
    modes=('scripted','neural','neural_approach') if args.approach else ('scripted','motor_matched','neural')
    report={'experiment':'Frozen-model approach candidate' if args.approach else 'Frozen-model motor parity','seeds':seeds,'round_limit':48,'model':str(args.model.resolve()),
            'matched_motor_inventory':'Wait, 5/10 foot moves at eight body-relative headings, visible-target attack',
            'method':'Project existing scripted movement to nearest available endpoint; no reach or obstacle oracle',
            'approach_candidate':args.approach,'results':evaluate(load(args.model),seeds,modes=modes)}
    args.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
