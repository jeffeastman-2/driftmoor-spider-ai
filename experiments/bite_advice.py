"""Private-sensory bite advice; replaces only attack admission in the cost score."""
import copy
import math
import torch
from torch import nn
import torch.nn.functional as F
from experiments.accurate_senses import AccurateEncounter,observation,FIELDS,SCHEMA
from experiments.jepa_planner import Planner,CostWeights,predicted_cost
from spider_sim.encounter_teacher import predicted_orientation
from models.body_geometry import Orientation


class AdviceEncounter(AccurateEncounter):
    def observation_for(self,actor):
        legacy=super().observation_for(actor)
        legacy["body_senses"]=observation(self,actor,legacy=legacy)
        return legacy


def posed_values(senses,old_pose,new_pose):
    """Transform measured geometry to a candidate own pose, without world access."""
    values=list(senses["values"]);v=dict(zip(FIELDS,values))
    for name,angle in zip(("roll","pitch","heading"),(new_pose.roll,new_pose.pitch,new_pose.heading)):
        v[name+"_sin"]=math.sin(math.radians(angle));v[name+"_cos"]=math.cos(math.radians(angle))
    if senses["visible"]:
        keys=["target_center_"+axis for axis in ("forward","left","up")]
        transformed=new_pose.inverse(old_pose.rotate(tuple(v[k] for k in keys)))
        v.update(zip(keys,transformed))
        for axis in ("forward","left","up"):
            keys=["target_"+axis+"_"+coordinate for coordinate in ("x","y","z")]
            transformed=new_pose.inverse(old_pose.rotate(tuple(v[k] for k in keys)))
            v.update(zip(keys,transformed))
    return [v[key] for key in FIELDS]


class BiteAdvisor:
    def __init__(self,models,slope=1.,bias=0.):
        self.models=models;self.slope=slope;self.bias=bias

    @torch.no_grad()
    def logits(self,x):
        return torch.stack([m(x).squeeze(-1) for m in self.models]).mean(0)

    @torch.no_grad()
    def probability(self,values):
        x=torch.tensor(values,dtype=torch.float32)
        return float(torch.sigmoid(self.slope*self.logits(x)+self.bias))


def load_advisor(directory,validation):
    models=[]
    for seed in (42,43,44):
        payload=torch.load(directory/("accurate-"+str(seed)+".pt"),map_location="cpu",weights_only=True)
        if payload.get("schema")!=SCHEMA or payload.get("fields")!=list(FIELDS):
            raise ValueError("Bite advisor schema mismatch")
        model=nn.Sequential(nn.Linear(57,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
        model.load_state_dict(payload["state"]);model.eval();models.append(model)
    advisor=BiteAdvisor(models)
    x=torch.tensor([row["accurate"] for row in validation],dtype=torch.float32)
    y=torch.tensor([row["admitted"] for row in validation],dtype=torch.float32)
    logits=advisor.logits(x).detach()
    # Fit on the original validation split only, never encounter evaluation.
    log_slope=nn.Parameter(torch.tensor(0.));bias=nn.Parameter(torch.tensor(0.))
    optimizer=torch.optim.LBFGS([log_slope,bias],max_iter=60,line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad();loss=F.binary_cross_entropy_with_logits(log_slope.exp()*logits+bias,y)
        loss.backward();return loss
    optimizer.step(closure)
    advisor.slope=float(log_slope.exp().detach());advisor.bias=float(bias.detach())
    return advisor


class AdvisedPlanner:
    def __init__(self,agent,advisor,weights=CostWeights()):
        self.agent=agent;self.advisor=advisor;self.weights=weights
    def session(self):
        return AdvisedSession(self.agent,self.advisor,self.weights)


class AdvisedSession:
    def __init__(self,agent,advisor,weights):
        self.base=Planner(agent,weights).session();self.advisor=advisor;self.weights=weights
        self.last_plan=None;self.last_orientation=None

    def choose_observation(self,actor,obs):
        if "body_senses" not in obs:
            raise ValueError("Bite advice requires private accurate-body-bite-v1 observations")
        original,belief=self.base.choose_observation(actor,obs)
        plan=copy.deepcopy(self.base.last_plan)
        attack_pose=Orientation(*predicted_orientation(obs,7,belief))
        old_pose=Orientation(*obs["self"]["orientation_roll_pitch_heading_deg"])
        values=posed_values(obs["body_senses"],old_pose,attack_pose)
        admission=self.advisor.probability(values)
        futures=torch.tensor(plan["predictions"]);futures[7,41]=admission
        costs=predicted_cost(torch.tensor(obs["values"]),futures,self.weights,obs["self"].get("condition"))
        prior=torch.tensor(plan["scores"])-torch.tensor(plan["costs"])
        scores=costs+prior;action=int(scores.argmin())
        plan.update({"without_advice_action":original,"selected":action,"bite_advice":admission,
            "costs":costs.tolist(),"scores":scores.tolist(),"predictions":futures.tolist()})
        self.last_plan=plan;self.last_orientation=predicted_orientation(obs,action,belief)
        return action,belief
