"""Experience replay for the predictor; all other network weights stay fixed."""
import copy
import random
import torch
import torch.nn.functional as F
from experiments.jepa_planner import Planner


def residual_predictor(prior):
    agent=copy.deepcopy(prior)
    if agent.prediction_mode=="absolute":
        # Preserve pretrained heads except the HP output, whose semantics change.
        with torch.no_grad():
            agent.predictor[-1].weight[16].zero_()
            agent.predictor[-1].bias[16].zero_()
        agent.prediction_mode="hp_delta"
    for name,param in agent.named_parameters():
        param.requires_grad_(name.startswith("predictor."))
    return agent.eval()


def collect_experience(agent,role,seeds,exploration=.3,rounds=18,world_factory=None,first_action=None):
    from spider_sim.dual_encounter import DualEncounter,ACTORS
    from spider_sim.encounter_evaluation import SCENARIOS,TeacherAgent
    from spider_sim.encounter_teacher import predicted_orientation
    episodes=[]
    for index,seed in enumerate(seeds):
        world=(world_factory(role,seed) if world_factory else
               DualEncounter(seed,scenario=SCENARIOS[index%len(SCENARIOS)]))
        session=Planner(agent).session()
        opponent=TeacherAgent().session()
        other=next(cid for cid in ACTORS if cid!=role)
        randomizer=random.Random(seed+900000)
        rows=[]
        for _ in range(rounds):
            if any(world.config.characters[c].character.current_hp<=0 for c in ACTORS):
                break
            world.begin_turn(role)
            obs=world.observation_for(role)
            proposed,belief=session.choose_observation(role,obs)
            explored=randomizer.random()<exploration
            action=randomizer.randrange(8) if explored else proposed
            if first_action is not None and not rows:
                action=first_action(seed)
                explored=True
            result=world.act(role,action,orientation=predicted_orientation(obs,action,belief))
            if all(world.config.characters[c].character.current_hp>0 for c in ACTORS):
                world.begin_turn(other)
                response,_=opponent.choose_observation(other,world.observation_for(other))
                world.act(other,response,orientation=opponent.last_orientation)
            world.begin_turn(role)
            future=world.observation_for(role)["values"]
            rows.append({"observation":obs["values"],"action":action,
                "next_observation":future,"feedback":[float(result["admitted"]),result["damage"]/20],
                "exploration":explored,"terminal":any(world.config.characters[c].character.current_hp<=0 for c in ACTORS)})
        episodes.append({"seed":seed,"role":role,"scenario":world.scenario,"rows":rows})
    return episodes


@torch.no_grad()
def replay_tensors(agent,episodes):
    states,observations,actions,targets=[],[],[],[]
    for episode in episodes:
        if not episode["rows"]:
            continue
        obs=torch.tensor([row["observation"] for row in episode["rows"]],dtype=torch.float32)
        state,_=agent.representation(obs[None])
        states.append(state[0]);observations.append(obs)
        actions.extend(row["action"] for row in episode["rows"])
        targets.extend(row["next_observation"]+row["feedback"] for row in episode["rows"])
    if not states:
        raise ValueError("Replay requires at least one transition")
    return torch.cat(states),torch.cat(observations),torch.tensor(actions),torch.tensor(targets,dtype=torch.float32)


def prediction_loss(agent,data):
    state,obs,actions,target=data
    pred=agent.predict_futures(state,actions,obs)
    # Separate physical feedback losses prevent rare events vanishing in 43-D MSE.
    return (F.mse_loss(pred[:,:41],target[:,:41])
        +10*F.mse_loss(pred[:,16]-obs[:,16],target[:,16]-obs[:,16])
        +2*F.mse_loss(pred[:,41],target[:,41])
        +4*F.mse_loss(pred[:,42],target[:,42]))


@torch.no_grad()
def forecast_metrics(agent,data):
    state,obs,actions,target=data
    pred=agent.predict_futures(state,actions,obs)
    attacks=actions==7
    injured=target[:,16]<obs[:,16]-1e-6
    return {"transitions":len(actions),"hp_fraction_mae":float((pred[:,16]-target[:,16]).abs().mean()),
        "injury_transitions":int(injured.sum()),
        "injury_hp_fraction_mae":float((pred[injured,16]-target[injured,16]).abs().mean()) if injured.any() else None,
        "no_injury_hp_fraction_mae":float((pred[~injured,16]-target[~injured,16]).abs().mean()) if (~injured).any() else None,
        "admission_mae":float((pred[:,41]-target[:,41]).abs().mean()),
        "attack_admission_mae":float((pred[attacks,41]-target[attacks,41]).abs().mean()) if attacks.any() else None,
        "attack_damage_mae":float((pred[attacks,42]-target[attacks,42]).abs().mean()*20) if attacks.any() else None,
        "actions":torch.bincount(actions,minlength=8).tolist()}


def fit_predictor(agent,training,validation,epochs=120,seed=1):
    torch.manual_seed(seed)
    train=replay_tensors(agent,training); valid=replay_tensors(agent,validation)
    optimizer=torch.optim.Adam(agent.predictor.parameters(),lr=.001)
    counts=torch.bincount(train[2],minlength=8).clamp(min=1)
    sampling=1/counts[train[2]].float()
    best=float(prediction_loss(agent,valid)); best_state=copy.deepcopy(agent.predictor.state_dict())
    for _ in range(epochs):
        indices=torch.multinomial(sampling,len(sampling),replacement=True)
        for start in range(0,len(indices),256):
            batch=tuple(t[indices[start:start+256]] for t in train)
            optimizer.zero_grad()
            loss=prediction_loss(agent,batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.predictor.parameters(),1)
            optimizer.step()
        with torch.no_grad():
            score=float(prediction_loss(agent,valid))
        if score<best:
            best=score;best_state=copy.deepcopy(agent.predictor.state_dict())
    agent.predictor.load_state_dict(best_state)
    return {"validation_loss":best,"training":forecast_metrics(agent,train),"validation":forecast_metrics(agent,valid)}
