"""One-step JEPA-like planning from private observations and learned predictions.

No environment object or authoritative geometry enters this module. The predictor
forecasts the next observation after an opponent response, not arbitrary latent
rollouts. Accordingly this planner deliberately has a one-step horizon.
"""
from dataclasses import dataclass
import torch
import torch.nn.functional as F

@dataclass(frozen=True)
class CostWeights:
    injury: float = 8.0
    attack: float = 1.0
    rejected_attack: float = 0.5
    progress: float = 1.0
    blocked_escape: float = 1.0
    policy_prior: float = 0.05


def goal_modulation(condition=None):
    """Bounded gameplay mappings from deviations from own authored baseline.

    Only dopamine and cortisol have defined cost roles in this experiment.
    Baseline-zero (absent) dimensions and missing context have no effect.
    """
    condition=condition or {}
    present=condition.get("present",())
    deltas=condition.get("deltas",{})
    def deviation(dim):
        return max(-1.,min(1.,float(deltas.get(dim,0)))) if dim in present else 0.
    stress=deviation("cortisol")
    motivation=deviation("dopamine")
    return {"injury_multiplier":1+0.5*stress,
            "attack_multiplier":1+0.5*motivation,
            "defensive_shift":0.5*stress}


def predicted_cost(current, futures, weights=CostWeights(), condition=None):
    """Return eight costs; normalized distances are in units of 60 feet.

    Costs use only own health, sensed range, blocked-flight feedback, and
    predicted action admission/damage. Clamps bound regression extrapolation;
    they are not calibrated probabilities or a guarantee against model errors.
    """
    own_hp = current[16].clamp(0,1)
    hp = futures[:,16].clamp(0,1)
    injury = (own_hp-hp).clamp(min=0)
    contact = current[5].clamp(0,1)*futures[:,5].clamp(0,1)
    distance_change = (futures[:,6].clamp(0,4)-current[6].clamp(0,4))*contact
    # Goal urgency is recomputed from current own state each decision. At half
    # health injury costs 50% more and approach gives way to holding distance;
    # below half, separation becomes valuable. No change to the frozen network.
    depletion = 1-own_hp
    goals=goal_modulation(condition)
    separation_preference=(2*depletion-1+goals["defensive_shift"]).clamp(-1,1)
    progress = separation_preference*distance_change
    costs = weights.injury*(1+depletion)*goals["injury_multiplier"]*injury-weights.progress*progress
    # Feedback refers to the candidate's executed action, not the opponent.
    admission = futures[:,41].clamp(0,1)
    damage = futures[:,42].clamp(0,2)
    costs[7] += weights.rejected_attack*(1-admission[7])-weights.attack*goals["attack_multiplier"]*damage[7]
    costs[4] += weights.blocked_escape*futures[4,29].clamp(0,1)
    return costs


class Planner:
    def __init__(self, agent, weights=CostWeights()):
        self.agent=agent
        self.weights=weights

    def session(self):
        return PlanningSession(self.agent,self.weights)


class PlanningSession:
    def __init__(self,agent,weights):
        self.agent=agent
        self.weights=weights
        self.hidden=None
        self.last_plan=None

    @torch.no_grad()
    def choose(self,values,condition=None):
        observation=torch.tensor(values,dtype=torch.float32)
        state,self.hidden=self.agent.representation(observation[None,None],self.hidden)
        latent=state[0,0]
        logits=self.agent.policy(latent)
        futures=self.agent.predict_futures(latent.expand(8,-1),torch.arange(8),observation.expand(8,-1))
        costs=predicted_cost(observation,futures,self.weights,condition)
        scores=costs-self.weights.policy_prior*F.log_softmax(logits,dim=-1)
        action=int(scores.argmin())
        belief=self.agent.belief(latent)
        belief=torch.cat((belief[:1].sigmoid(),belief[1:]))
        self.last_plan={"selected":action,"policy_action":int(logits.argmax()),
                        "goal_modulation":goal_modulation(condition),
                        "costs":costs.tolist(),"scores":scores.tolist(),
                        "predictions":futures.tolist()}
        return action,belief.tolist()

    def choose_observation(self,actor,observation):
        # Consume only the deciding actor's explicitly supplied private snapshot.
        action,belief=self.choose(observation["values"],observation["self"].get("condition"))
        from spider_sim.encounter_teacher import predicted_orientation
        self.last_orientation=predicted_orientation(observation,action,belief)
        return action,belief
