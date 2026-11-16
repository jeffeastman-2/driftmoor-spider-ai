"""Single spider vs fixed scripted prey: offline engine values and held-out catches."""
import argparse
import copy
import json
import math
import random
from pathlib import Path
import torch
import torch.nn.functional as F
from spider_sim.ecology import EcologyEncounter,GoalBaseline,ForagerBaseline,SPIDER
from spider_sim.djepa_ecology import senses,CautiousTeacher
from spider_sim.pursuit import (PursuitGoal,PursuitModel,PursuitPolicy,features,actions,action_features,save,load,SCHEMA,ACTION_SIZE)
from models.space_time import Position
from models.body_geometry import Orientation


def fixture(seed,kind):
    w=EcologyEncounter(seed);rng=random.Random(seed)
    bounds=rng.choice(((49,78,86,150),(88,117,126,149),(12,40,86,109)))
    prey=rng.choice(('cave_rat_1','cave_cricket_1'));x0,x1,y0,y1=bounds
    for _ in range(1000):
        x=rng.uniform(x0+1,x1-1);y=rng.uniform(y0+1,y1-1)
        xx=rng.uniform(x0+1,x1-1);yy=rng.uniform(y0+1,y1-1)
        if 5<math.dist((x,y),(xx,yy))<28 and w.geometry.can_see(Position(x,y),Position(xx,yy)):break
    else:raise ValueError('No visible pursuit start')
    for actor,px,py in ((SPIDER,x,y),(prey,xx,yy)):
        w.positions[actor]=(px,py,w.geometry.zone_at(Position(px,py)))
        w.orientations[actor]=Orientation(heading=rng.uniform(0,360))
    # This benchmark has exactly two participants, including after movement.
    # Prevent dormant scene residents from activating in the stripped fixture.
    for cs in w.config.characters.values():cs.activation='always'
    w.active={SPIDER,prey}
    policy=CautiousTeacher('rat') if kind=='cautious' else ForagerBaseline(seed+700)
    return w,prey,policy


def prey_step(world,prey,policy,kind):
    if prey not in world.active:return
    if kind=='stationary':command={'action':'wait'}
    else:command=policy.choose(senses(world,prey) if kind=='cautious' else world.observation_for(prey))
    world.act(prey,command)


def advance(world,prey,policy,kind,command):
    result=world.act(SPIDER,command);prey_step(world,prey,policy,kind);world.elapsed_seconds+=6
    return result


def goal_cost(world,prey):
    hp=world.config.characters[prey].character.current_hp
    if hp<=0:return 0.
    # Task supervisor only. Hidden HP/coordinates never enter policy features.
    distance=math.dist(world.positions[SPIDER][:2],world.positions[prey][:2])
    return 1+min(distance,60)/10+hp/world.max_hp[prey]


def collect(seeds,student=None,rounds=24,candidate_size=ACTION_SIZE):
    episodes=[]
    for n,seed in enumerate(seeds):
        kind=('stationary','fleeing','cautious')[seed%3]
        world,prey,opponent=fixture(seed,kind);task=PursuitGoal();learner=PursuitPolicy(student) if student else None
        rng=random.Random(seed+400);frames=[]
        for turn in range(rounds):
            if prey not in world.active:break
            obs=senses(world,SPIDER);goal=task.observe(obs);choices=actions(obs,goal);outcomes=[]
            for choice in choices:
                branch=copy.deepcopy(world);other=copy.deepcopy(opponent)
                advance(branch,prey,other,kind,choice)
                outcomes.append(goal_cost(branch,prey))
            best=min(range(len(choices)),key=lambda i:outcomes[i])
            # Query the engine at student-visited states as well as oracle states.
            suggested=learner.choose(obs) if learner else None
            if learner and rng.random()<.8:command=suggested
            elif rng.random()<.15:command=rng.choice(choices)
            else:command=choices[best]
            frames.append({'own':features(obs,goal),'choices':[action_features(c,obs,goal,candidate_size) for c in choices],
                           'costs':outcomes,'best':best})
            task.previous=command
            if learner:learner.task.previous=command
            advance(world,prey,opponent,kind,command)
        episodes.append({'seed':seed,'kind':kind,'frames':frames})
        if (n+1)%12==0:print('collect',n+1,'episodes',flush=True)
    return episodes


def tensors(episodes):
    batch=len(episodes);steps=max(len(e['frames']) for e in episodes);count=max(len(f['choices']) for e in episodes for f in e['frames'])
    own=torch.zeros(batch,steps,36);choices=torch.zeros(batch,steps,count,len(episodes[0]['frames'][0]['choices'][0]))
    target=torch.zeros(batch,steps,count);mask=torch.zeros(batch,steps,count,dtype=torch.bool)
    for b,e in enumerate(episodes):
        for t,f in enumerate(e['frames']):
            own[b,t]=torch.tensor(f['own']);n=len(f['choices']);choices[b,t,:n]=torch.tensor(f['choices'])
            target[b,t,:n]=torch.tensor(f['costs']);mask[b,t,:n]=True
    return own,choices,target,mask


def objective(model,data):
    own,choices,target,mask=data;state,_=model.represent(own)
    values=model.values(state[:,:,None,:].expand(-1,-1,choices.shape[2],-1),choices)
    regression=F.smooth_l1_loss(values[mask],target[mask])
    # Soft outcome ranking: near-equivalent successful actions aren't arbitrary
    # contradictory one-hot targets. Temperature is fixed before held-out testing.
    valid=mask.any(-1);label=F.softmax((-target/.25).masked_fill(~mask,-1e9),dim=-1)
    log=F.log_softmax((-values/.25).masked_fill(~mask,-1e9),dim=-1)
    ranking=-(label*log).sum(-1)[valid].mean()
    return regression+ranking


def fit(training,validation,model=None,epochs=300):
    torch.manual_seed(42);model=model or PursuitModel(len(training[0]['frames'][0]['choices'][0]));optim=torch.optim.Adam(model.parameters(),lr=.002)
    train=tensors(training);valid=tensors(validation);best=float('inf');weights=None
    for epoch in range(epochs):
        model.train();optim.zero_grad();loss=objective(model,train);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optim.step()
        model.eval()
        with torch.no_grad():score=float(objective(model,valid))
        if score<best:best=score;weights=copy.deepcopy(model.state_dict())
        if (epoch+1)%100==0:print('epoch',epoch+1,'validation loss',round(score,4),flush=True)
    model.load_state_dict(weights);return model.eval(),best


def evaluate(model,seeds,rounds=48):
    groups={}
    for kind in ('stationary','fleeing','cautious'):
        groups[kind]={}
        for mode in ('scripted','neural'):
            total={'episodes':0,'caught':0,'turns_to_catch':[],'moves':0,'blocked':0,'return_loops':0,'longest_return_loop_streak':0,'reversals':0,'attacks':0,'admitted':0}
            for seed in seeds:
                world,prey,opponent=fixture(seed,kind);controller=PursuitPolicy(model) if mode=='neural' else GoalBaseline(seed=seed)
                positions=[world.positions[SPIDER][:2]];previous=None;loop_streak=0
                for turn in range(rounds):
                    obs=senses(world,SPIDER) if mode=='neural' else world.observation_for(SPIDER)
                    command=controller.choose(obs);result=advance(world,prey,opponent,kind,command)
                    position=world.positions[SPIDER][:2]
                    if command['action']=='move':
                        total['moves']+=1;total['blocked']+=int(result['status']=='blocked')
                        returned=len(positions)>1 and math.dist(position,positions[-2])<.25 and math.dist(position,positions[-1])>1
                        total['return_loops']+=int(returned)
                        if previous and previous['action']=='move' and abs((command['heading_deg']-previous['heading_deg'])%360-180)<1e-6:total['reversals']+=1
                    elif command['action']=='attack':total['attacks']+=1;total['admitted']+=int(result['admitted'])
                    returned=(command['action']=='move' and len(positions)>1 and math.dist(position,positions[-2])<.25 and math.dist(position,positions[-1])>1)
                    loop_streak=loop_streak+1 if returned else 0
                    total['longest_return_loop_streak']=max(total['longest_return_loop_streak'],loop_streak)
                    positions.append(position);previous=command
                    if world.config.characters[prey].character.current_hp<=0:
                        total['caught']+=1;total['turns_to_catch'].append(turn+1);break
                total['episodes']+=1
            groups[kind][mode]=total;print('evaluate',kind,mode,total,flush=True)
    return groups


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--action-features',type=int,choices=(21,23),default=23)
    parser.add_argument('--episodes',type=int,default=48);parser.add_argument('--epochs',type=int,default=300)
    args=parser.parse_args()
    if args.episodes<4 or args.epochs<1:parser.error('Use at least four episodes and positive epochs')
    torch.set_num_threads(1);args.output.mkdir(exist_ok=False)
    validation=collect(range(510000,510012),candidate_size=args.action_features);training=collect(range(500000,500000+args.episodes),candidate_size=args.action_features)
    model,loss=fit(training,validation,epochs=args.epochs)
    # One student-state aggregation pass; no final test encounter is used here.
    student=collect(range(520000,520024),student=model,candidate_size=model.action_size);training+=student
    model,loss=fit(training,validation,model,epochs=args.epochs)
    save(model,args.output/'spider.pt')
    (args.output/'training.json').write_text(json.dumps(training));(args.output/'validation.json').write_text(json.dumps(validation))
    report={'schema':SCHEMA,'action_features':args.action_features,'training_seeds':[500000,500000+args.episodes-1],'aggregation_seeds':[520000,520023],
            'validation_seeds':[510000,510011],'test_seeds':list(range(530000,530032)),
            'validation_loss':loss,'epochs_per_pass':args.epochs,'evaluation':evaluate(model,range(530000,530032)),
            'limitations':['One optimizer seed; same Level-2 room regions, not new landscapes','Authored persistent target task and motor inventory',
                           'Explicit one-step engine goal-cost supervision, not joint-embedding JEPA or reinforcement learning',
                           'Baseline has continuous headings while neural motor choices retain 45-degree increments','No live-controller replacement until behavior warrants promotion']}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
