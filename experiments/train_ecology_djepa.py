"""Initialize species MLP/GRU controllers with engine outcomes and cautious proposals."""
import argparse
import copy
import json
import math
import random
from pathlib import Path
import torch
import torch.nn.functional as F
from experiments.game_runtime import game_root
from spider_sim.djepa_ecology import (SCHEMA,SPECIES,SELF_SIZE,CONTACT_SIZE,EcologyModel,
    NeuralPolicy,CautiousTeacher,senses,contacts,encode,candidates,candidate_values,species_for,save,load,CANDIDATE_SIZE)
from spider_sim.ecology import EcologyEncounter,baseline_policies,ecology_round,SPIDER
from models.space_time import Position
from models.body_geometry import Orientation

FOCUS={"rat":"cave_rat_1","spider":SPIDER,"shadowbeast":"freight_shadowbeast"}


def teachers(world):
    policies=baseline_policies(world,world.seed)
    for actor in policies:
        species=species_for(actor)
        if species:policies[actor]=CautiousTeacher(species)
    return policies


def teacher_observation(world,actor,policy):
    return senses(world,actor) if isinstance(policy,CautiousTeacher) else world.observation_for(actor)


def training_world(species,seed):
    """Stratified private encounters: near/far combat, danger and edge-biased movement.

    Sampling coordinates is an offline fixture operation, never an agent input.
    Keep opponents scripted and vary them independently of focal species.
    """
    world=EcologyEncounter(seed);rng=random.Random(seed);focus=FOCUS[species]
    room=rng.choice(((88,117,126,149),(49,78,86,150),(12,40,86,109)))
    x0,x1,y0,y1=room
    # Half the starts face a sampling-box edge. This biases proximity to
    # room edges but does not guarantee a wall: native geometry can be angled
    # or connect through that edge. Evaluation must measure actual clearance.
    wall=seed%2==0
    x=rng.uniform(x0+3,x1-3);y=rng.uniform(y0+3,y1-3)
    if wall:
        side=rng.randrange(4)
        x,y,heading=((x0+.2,y,180),(x1-.2,y,0),(x,y0+.2,270),(x,y1-.2,90))[side]
    else:heading=rng.uniform(0,360)
    world.positions[focus]=(x,y,world.geometry.zone_at(Position(x,y)))
    world.orientations[focus]=Orientation(heading=heading)
    if species=='rat':other=rng.choice((SPIDER,'freight_shadowbeast'))
    elif species=='spider':other=rng.choice(('cave_rat_1','cave_cricket_1','freight_shadowbeast'))
    else:other=rng.choice(('cave_rat_1','cave_cricket_1',SPIDER))
    # Equal episode strata cover close (often legal) and distant attack requests.
    near=(seed//2)%2==0
    for _ in range(100):
        angle=rng.uniform(-math.pi,math.pi);distance=rng.uniform(1.5,4.5) if near else rng.uniform(10,22)
        xx=x+distance*math.cos(angle);yy=y+distance*math.sin(angle)
        zone=world.geometry.zone_at(Position(xx,yy))
        if zone and x0<=xx<=x1 and y0<=yy<=y1:
            world.positions[other]=(xx,yy,zone);break
    else:raise ValueError('Could not place curriculum opponent')
    world.orientations[other]=Orientation(heading=rng.uniform(0,360))
    world.active={focus,other}
    character=world.config.characters[focus].character
    character.current_hp=max(1,round(world.max_hp[focus]*rng.choice((.25,.5,1.,1.))))
    return world


def transition(world,policies,actor,command):
    result=world.act(actor,command)
    for other in sorted(list(world.active)):
        if other==actor or other not in world.active:continue
        policy=policies[other]
        policy_obs=teacher_observation(world,other,policy)
        world.act(other,policy.choose(policy_obs))
    world.elapsed_seconds+=6
    return result


def collect(species,seeds,rounds=12):
    episodes=[];actor=FOCUS[species]
    for number,seed in enumerate(seeds):
        world=training_world(species,seed);policy=teachers(world);rng=random.Random(seed+50000);frames=[]
        for turn in range(rounds):
            if actor not in world.active:break
            obs=senses(world,actor);choices=candidates(obs)
            taught=policy[actor].choose(obs);teacher=choices.index(taught)
            actual=teacher if rng.random()>.35 else rng.randrange(len(choices))
            # Cover attack requests as well as successful movement; do not mask reach.
            attack_indices=[i for i,c in enumerate(choices) if c['action']=='attack']
            selected={teacher,actual,0}
            if attack_indices:selected.add(rng.choice(attack_indices))
            selected.update(rng.sample(range(1,17),2))
            targets=[];before=world.config.characters[actor].character.current_hp
            origin=world.positions[actor]
            nearest=min(contacts(obs).values(),key=lambda c:c['range_ft'],default=None)
            for index in sorted(selected):
                branch=copy.deepcopy(world);opponents=copy.deepcopy(policy)
                result=transition(branch,opponents,actor,choices[index])
                future=senses(branch,actor)
                after=branch.config.characters[actor].character.current_hp
                dx=branch.positions[actor][0]-origin[0];dy=branch.positions[actor][1]-origin[1]
                heading=math.radians(obs['self']['heading_deg'])
                target=choices[index].get('target') or (nearest['track'] if nearest else None)
                contact=contacts(future).get(target)
                label=[(before-after)/world.max_hp[actor],
                    (dx*math.cos(heading)+dy*math.sin(heading))/30,
                    (-dx*math.sin(heading)+dy*math.cos(heading))/30,
                    float(result['admitted']),result['damage']/20,float(contact is not None),
                    contact['range_ft']/30 if contact else 0.]
                targets.append({'index':index,'values':label})
            frames.append({'observation':obs,'choices':choices,'teacher':teacher,'targets':targets})
            transition(world,policy,actor,choices[actual])
        episodes.append({'seed':seed,'species':species,'frames':frames})
        if (number+1)%4==0:print('collect',species,number+1,'episodes',flush=True)
    return episodes


def tensors(episodes,species,candidate_size=CANDIDATE_SIZE):
    episodes=[e for e in episodes if e['frames']]
    batch=len(episodes);steps=max(len(e['frames']) for e in episodes)
    maximum=max(1,max(len(contacts(f['observation'])) for e in episodes for f in e['frames']))
    own=torch.zeros(batch,steps,SELF_SIZE);seen=torch.zeros(batch,steps,maximum,CONTACT_SIZE);mask=torch.zeros(batch,steps,maximum)
    frames=[];branches=[];targets=[];branch_frame=[];attack=[]
    for b,episode in enumerate(episodes):
        for t,frame in enumerate(episode['frames']):
            values,rows=encode(frame['observation'],species)
            own[b,t]=torch.tensor(values)
            if rows:seen[b,t,:len(rows)]=torch.tensor(rows);mask[b,t,:len(rows)]=1
            frames.append((b,t,frame))
            for target in frame['targets']:
                command=frame['choices'][target['index']]
                branches.append(candidate_values(command,frame['observation'],candidate_size))
                targets.append(target['values']);branch_frame.append(len(frames)-1)
                attack.append(command['action']=='attack')
    max_choices=max(len(frame['choices']) for _,_,frame in frames)
    proposal_choices=torch.zeros(len(frames),max_choices,len(branches[0]));proposal_mask=torch.zeros(len(frames),max_choices,dtype=torch.bool)
    proposal_labels=[]
    for n,(_,_,frame) in enumerate(frames):
        rows=[candidate_values(c,frame['observation'],candidate_size) for c in frame['choices']]
        proposal_choices[n,:len(rows)]=torch.tensor(rows);proposal_mask[n,:len(rows)]=True
        proposal_labels.append(frame['teacher'])
    return (own,seen,mask,frames,torch.tensor(branches),torch.tensor(targets),torch.tensor(branch_frame),torch.tensor(attack),
            proposal_choices,proposal_mask,torch.tensor(proposal_labels))


def objective(model,data):
    own,seen,mask,frames,choices,target,frame_ids,attack,proposal_choices,proposal_mask,proposal_labels=data
    states,_=model.represent(own,seen,mask)
    latent=torch.stack([states[b,t] for b,t,_ in frames])
    pred=model.outcomes(latent[frame_ids],choices)
    loss=10*F.mse_loss(pred[:,0],target[:,0])+F.mse_loss(pred[:,1:3],target[:,1:3])
    if attack.any():
        # Equal class influence; an all-reject classifier must not win because
        # distant requests outnumber admitted attacks in trajectories.
        terms=[]
        for accepted in (False,True):
            subset=attack & ((target[:,3]>.5)==accepted)
            if subset.any():terms.append(F.binary_cross_entropy_with_logits(pred[subset,3],target[subset,3]))
        loss=loss+torch.stack(terms).mean()
    admitted=attack & (target[:,3]>.5)
    if admitted.any():loss=loss+4*F.mse_loss(pred[admitted,4],target[admitted,4])
    loss=loss+.25*F.binary_cross_entropy_with_logits(pred[:,5],target[:,5])
    visible=target[:,5]>.5
    if visible.any():loss=loss+.2*F.mse_loss(pred[visible,6],target[visible,6])
    repeated=latent[:,None,:].expand(-1,proposal_choices.shape[1],-1)
    logits=model.proposal(torch.cat((repeated,proposal_choices),dim=-1)).squeeze(-1)
    logits=logits.masked_fill(~proposal_mask,-1e9)
    loss=loss+F.cross_entropy(logits,proposal_labels)
    return loss,pred


@torch.no_grad()
def metrics(model,data):
    loss,pred=objective(model,data);target=data[5];attack=data[7]
    actual=target[attack,3]>.5;chosen=pred[attack,3].sigmoid()>.5
    recalls=[float((chosen[actual]).float().mean()) if actual.any() else None,
             float((~chosen[~actual]).float().mean()) if (~actual).any() else None]
    return {'loss':float(loss),'branches':len(target),'attack_requests':int(attack.sum()),
            'admission_accuracy':float((chosen==actual).float().mean()) if attack.any() else None,
            'admitted_samples':int(actual.sum()),'rejected_samples':int((~actual).sum()),
            'admission_recall':recalls[0],'rejection_recall':recalls[1],
            'balanced_admission_accuracy':sum(recalls)/2 if all(r is not None for r in recalls) else None,
            'injury_fraction_mae':float((pred[:,0]-target[:,0]).abs().mean()),
            'attack_damage_hp_mae':float((pred[attack,3].sigmoid()*pred[attack,4]-target[attack,4]).abs().mean()*20) if attack.any() else None}


def train(species,training,validation,output,epochs,candidate_size=CANDIDATE_SIZE):
    torch.manual_seed(42);model=EcologyModel(candidate_size);optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    train_data=tensors(training,species,candidate_size);valid_data=tensors(validation,species,candidate_size)
    initial=metrics(model,valid_data);best=initial['loss'];state=copy.deepcopy(model.state_dict())
    for epoch in range(epochs):
        model.train();optimizer.zero_grad();loss,_=objective(model,train_data);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        model.eval();score=metrics(model,valid_data)['loss']
        if score<best:best=score;state=copy.deepcopy(model.state_dict())
        if (epoch+1)%30==0:print('train',species,epoch+1,'validation',round(score,4),flush=True)
    model.load_state_dict(state);model.eval();save(model,output/(species+'.pt'),species)
    return {'initial_validation':initial,'validation':metrics(model,valid_data),'training':metrics(model,train_data)}


def evaluate(directory,seeds,rounds=48):
    from spider_sim.djepa_ecology import policies as neural_policies
    groups={}
    for mode in ('scripted','cautious_teacher','neural','neural_proposal'):
        summary={s:{'actors':0,'survived':0,'hp_lost':0,'attacks':0,'admitted':0,'rejected':0,'damage':0,'moves':0,'blocked':0} for s in SPECIES}
        for seed in seeds:
            world=EcologyEncounter(seed)
            if mode.startswith('neural'):policy=neural_policies(world,directory,seed,selection='proposal' if mode=='neural_proposal' else 'predictive')
            elif mode=='cautious_teacher':
                policy=teachers(world)
                # The loop's sensor boundary recognizes this flag, not access inside choose.
                for p in policy.values():
                    if isinstance(p,CautiousTeacher):p.neural=True
            else:policy=baseline_policies(world,seed)
            records=[]
            for turn in range(rounds):records.extend(ecology_round(world,policy,turn))
            for cid in world.positions:
                species=species_for(cid)
                if not species:continue
                row=summary[species];hp=world.config.characters[cid].character.current_hp
                row['actors']+=1;row['survived']+=int(hp>0);row['hp_lost']+=world.max_hp[cid]-hp
            for record in records:
                species=species_for(record['actor'])
                if not species:continue
                row=summary[species];result=record['result'];action=record['command']['action']
                if action=='attack':
                    row['attacks']+=1;row['admitted']+=int(result['admitted']);row['rejected']+=int(not result['admitted']);row['damage']+=result['damage']
                if action=='move':row['moves']+=1;row['blocked']+=int(result['status']=='blocked')
            (directory/(mode+'-'+str(seed)+'.json')).write_text(json.dumps({'seed':seed,'mode':mode,'records':records}))
        groups[mode]=summary;print('evaluate',mode,summary,flush=True)
    return groups


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--candidate-clearance',action='store_true',help='Add requested-direction clearance to each candidate feature vector')
    parser.add_argument('--epochs',type=int,default=300)
    parser.add_argument('--episodes',type=int,default=64)
    args=parser.parse_args()
    if args.output.exists() or args.episodes<4 or args.epochs<1:parser.error('Use a new output directory, at least four episodes and positive epochs')
    torch.set_num_threads(1);args.output.mkdir(parents=True)
    report={'schema':SCHEMA,'method':'Species set encoder + individual GRU; engine outcome learning and cautious teacher proposals; one-step predictive cost selection',
            'epochs':args.epochs,'candidate_clearance':args.candidate_clearance,'models':{},'evaluation_seeds':list(range(340000,340008)),
            'limitations':['One optimizer seed per species; familiar Level-2 layout only','Authored goal configuration and costs, not a learned critic/configurator',
                           'Explicit numeric forecasts, not joint-embedding JEPA training','Six-second rounds; no feeding, breeding or time-skipping added',
                           'Local ideal point-clearance sensor, not full-body navigation or global map','Three requested species use neural weights; crickets/lizards retain baseline scripts']}
    for n,species in enumerate(SPECIES):
        start=100000+n*10000
        training=collect(species,range(start,start+args.episodes))
        validation=collect(species,range(start+1000,start+1004))
        (args.output/(species+'-training.json')).write_text(json.dumps(training))
        (args.output/(species+'-validation.json')).write_text(json.dumps(validation))
        report['models'][species]=train(species,training,validation,args.output,args.epochs,CANDIDATE_SIZE+int(args.candidate_clearance))
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    report['encounters']=evaluate(args.output,range(340000,340008))
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
