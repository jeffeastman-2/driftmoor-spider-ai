"""Initialize the pursuit actor from private scripted decisions, without cost scoring."""
import argparse,copy,json,math,random
from pathlib import Path
import torch
import torch.nn.functional as F
from experiments.train_single_prey import fixture,advance,evaluate
from spider_sim.ecology import GoalBaseline,SPIDER
from spider_sim.djepa_ecology import senses
from spider_sim.pursuit import PursuitGoal,PursuitPolicy,PursuitModel,actions,features,action_features,save


def collect(seeds,student=None,rounds=48):
    episodes=[]
    for seed in seeds:
        kind=('stationary','fleeing','cautious')[seed%3];world,prey,opponent=fixture(seed,kind)
        expert=GoalBaseline(seed=seed);task=PursuitGoal();policy=PursuitPolicy(student,approach=True) if student else None
        rng=random.Random(seed+900);frames=[]
        for turn in range(rounds):
            if prey not in world.active:break
            obs=senses(world,SPIDER);goal=task.observe(obs);choices=actions(obs,goal,approach=True)
            requested=expert.choose(world.observation_for(SPIDER))
            def error(c):
                if c['action']!=requested['action']:return 1e6
                if c['action']=='attack':return 0 if c['target']==requested['target'] else 1e6
                if c['action']=='wait':return 0
                a=math.radians(c['heading_deg']);b=math.radians(requested['heading_deg'])
                return math.dist((c['distance_ft']*math.cos(a),c['distance_ft']*math.sin(a)),
                                 (requested['distance_ft']*math.cos(b),requested['distance_ft']*math.sin(b)))
            teacher=min(range(len(choices)),key=lambda i:error(choices[i]))
            frames.append({'own':features(obs,goal),'choices':[action_features(c,obs,goal,21) for c in choices],'teacher':teacher})
            suggested=policy.choose(obs) if policy else None
            command=suggested if policy and rng.random()<.8 else choices[teacher]
            task.previous=command
            if policy:policy.task.previous=command
            advance(world,prey,opponent,kind,command)
        episodes.append({'seed':seed,'kind':kind,'frames':frames})
    return episodes


def tensors(episodes):
    b=len(episodes);t=max(len(e['frames']) for e in episodes);n=max(len(f['choices']) for e in episodes for f in e['frames'])
    own=torch.zeros(b,t,36);choices=torch.zeros(b,t,n,21);mask=torch.zeros(b,t,n,dtype=torch.bool);labels=torch.zeros(b,t,dtype=torch.long)
    for i,e in enumerate(episodes):
        for j,f in enumerate(e['frames']):
            own[i,j]=torch.tensor(f['own']);k=len(f['choices']);choices[i,j,:k]=torch.tensor(f['choices']);mask[i,j,:k]=True;labels[i,j]=f['teacher']
    return own,choices,mask,labels


def loss(model,data):
    own,choices,mask,labels=data;state,_=model.represent(own)
    logits=-model.values(state[:,:,None].expand(-1,-1,choices.shape[2],-1),choices)
    logits=logits.masked_fill(~mask,-1e9);valid=mask.any(-1)
    return F.cross_entropy(logits[valid],labels[valid])


def fit(training,validation,model=None):
    torch.manual_seed(42);model=model or PursuitModel(21);model.value_kind='negative_action_logit'
    optimizer=torch.optim.Adam(model.parameters(),lr=.003);train=tensors(training);valid=tensors(validation);best=float('inf');weights=None
    for epoch in range(400):
        optimizer.zero_grad();objective=loss(model,train);objective.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        with torch.no_grad():score=float(loss(model,valid))
        if score<best:best=score;weights=copy.deepcopy(model.state_dict())
        if (epoch+1)%100==0:print('imitation epoch',epoch+1,'validation',round(score,4),flush=True)
    model.load_state_dict(weights);return model.eval(),best


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(1);args.output.mkdir(exist_ok=False)
    train=collect(range(650000,650064));valid=collect(range(651000,651012))
    print('teacher frames',sum(len(e['frames']) for e in train),flush=True)
    model,score=fit(train,valid)
    train+=collect(range(652000,652032),student=model)
    model,score=fit(train,valid,model);save(model,args.output/'spider.pt')
    (args.output/'training.json').write_text(json.dumps(train));(args.output/'validation.json').write_text(json.dumps(valid))
    report={'method':'Private-sensory behavior cloning + one student-state aggregation pass; no cost-based choice',
            'training_seeds':[650000,650063],'aggregation_seeds':[652000,652031],'validation_seeds':[651000,651011],
            'test_seeds':list(range(653000,653032)),'validation_ce':score,
            'results':evaluate(model,range(653000,653032),modes=('scripted','neural_approach')),
            'limitations':['One optimizer seed; familiar room regions only','Authored target commitment and approach motor; actor learned from script',
                           'Action scores are negative logits, not learned physical forecasts or JEPA costs','No live promotion']}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
