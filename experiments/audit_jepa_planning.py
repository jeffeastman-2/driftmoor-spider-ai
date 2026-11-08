"""Offline diagnostic: separate forecast ranking errors from cost preferences.

Authoritative simulation is used ONLY by this audit as a reference instrument.
It is never available to the deployed planner. Frozen networks, cold memory,
controlled starting poses, and repeated paired dice streams make this a local
counterfactual test, not an encounter-performance benchmark.
"""
import argparse
import copy
import json
import random
from pathlib import Path
import torch
import torch.nn.functional as F
from experiments.game_runtime import game_root
from experiments.recurrent_agents import load
from experiments.jepa_planner import Planner,CostWeights,predicted_cost

CASES=("healthy_close_idle","healthy_close_teacher","healthy_far_idle",
       "wounded_close_teacher","critical_blocked_teacher","no_contact_idle")


def fixture(role,case,seed):
    from spider_sim.dual_encounter import DualEncounter,ACTORS,PLAYER,SPIDER
    from models.body_geometry import Orientation
    world=DualEncounter(seed,outages=False,scenario="wounded_spider")
    for cid in ACTORS:
        world.config.characters[cid].character.current_hp=world.max_hp[cid]
    own_hp=.25 if case=="wounded_close_teacher" else .1 if case=="critical_blocked_teacher" else 1
    world.config.characters[role].character.current_hp=max(1,round(world.max_hp[role]*own_hp))
    world.escape_blocked[role]=case=="critical_blocked_teacher"
    world.positions[SPIDER]=(70,195,"Secret Chamber")
    world.positions[PLAYER]=(76,195,"Secret Chamber")
    if case=="healthy_far_idle":
        world.positions[PLAYER]=(80,195,"Secret Chamber")
    if case=="no_contact_idle":
        # Authored opaque web curtain plus concealment; no hidden target cue.
        world.outages=True
        world.tick=0
        world.concealed=True
        world.positions[PLAYER]=(110,195,world.positions[PLAYER][2])
    world.orientations[SPIDER]=Orientation(heading=0)
    world.orientations[PLAYER]=Orientation(heading=180)
    world.begin_turn(role)
    return world


def reference_future(world,role,action,belief,dice_seed,opponent):
    from spider_sim.dual_encounter import ACTORS
    from spider_sim.encounter_teacher import predicted_orientation
    from spider_sim.encounter_evaluation import TeacherAgent
    branch=copy.deepcopy(world)
    branch.random.setstate(random.Random(dice_seed).getstate())
    obs=branch.observation_for(role)
    result=branch.act(role,action,orientation=predicted_orientation(obs,action,belief))
    other=next(cid for cid in ACTORS if cid!=role)
    if all(branch.config.characters[cid].character.current_hp>0 for cid in ACTORS):
        branch.begin_turn(other)
        if opponent=="teacher":
            session=TeacherAgent().session()
            response,_=session.choose_observation(other,branch.observation_for(other))
            branch.act(other,response,orientation=session.last_orientation)
        else:
            branch.act(other,0)
    branch.begin_turn(role)
    future=branch.observation_for(role)["values"]
    return future+[float(result["admitted"]),result["damage"]/20],result


def analyze(world,role,agent,draws,dice_start=240000):
    from spider_sim.dual_encounter import ACTIONS
    observation=world.observation_for(role)
    values=torch.tensor(observation["values"],dtype=torch.float32)
    condition=observation["self"]["condition"]
    session=Planner(agent).session()
    selected,belief=session.choose(observation["values"],condition)
    learned=torch.tensor(session.last_plan["predictions"])
    opponent="teacher" if world.audit_case.endswith("teacher") else "idle"
    futures=[]
    for draw in range(draws):
        futures.append([reference_future(world,role,action,belief,dice_start+draw,opponent)[0]
                        for action in range(8)])
    actual=torch.tensor(futures,dtype=torch.float32)
    costs=torch.stack([predicted_cost(values,row,condition=condition) for row in actual]).mean(0)
    with torch.no_grad():
        state,_=agent.representation(values[None,None])
        prior=-CostWeights().policy_prior*F.log_softmax(agent.policy(state[0,0]),dim=-1)
    scores=costs+prior
    reference=int(scores.argmin())
    mean=actual.mean(0)
    ablations={}
    for label,indices in (("next_hp",[16]),("attack_feedback",[41,42]),
                          ("contact_range",[5,6]),("blocked_flight",[29])):
        corrected=learned[None].expand(len(actual),-1,-1).clone()
        corrected[:,:,indices]=actual[:,:,indices]
        corrected_cost=torch.stack([predicted_cost(values,row,condition=condition)
                                   for row in corrected]).mean(0)
        corrected_action=int((corrected_cost+prior).argmin())
        ablations[label]={"action":ACTIONS[corrected_action],
            "reference_regret":float(scores[corrected_action]-scores.min())}
    rows=[]
    for a,name in enumerate(ACTIONS):
        rows.append({"action":name,"predicted_cost":session.last_plan["costs"][a],
            "reference_expected_cost":float(costs[a]),"reference_score_with_prior":float(scores[a]),
            "predicted_next_hp":float(learned[a,16]),"reference_next_hp":float(mean[a,16]),
            "predicted_admission":float(learned[a,41]),"reference_admission":float(mean[a,41]),
            "predicted_damage":float(learned[a,42]*20),"reference_damage":float(mean[a,42]*20)})
    return {"role":role,"case":world.audit_case,"opponent":opponent,
        "contact":observation["contact"],"hp_fraction":float(values[16]),
        "policy_action":ACTIONS[session.last_plan["policy_action"]],
        "learned_cost_action":ACTIONS[selected],"reference_cost_action":ACTIONS[reference],
        "reference_cost_without_prior":ACTIONS[int(costs.argmin())],
        "forecast_ranking_regret":float(scores[selected]-scores.min()),
        "mean_next_hp_absolute_error":float((learned[:,16]-mean[:,16]).abs().mean()),
        "cost_prefers_attack_over_wait":bool(costs[7]<costs[0]),
        "cost_prefers_retreat_over_flee":bool(costs[2]<costs[4]),
        "offline_forecast_replacements":ablations,"actions":rows}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driftmoor-root",type=Path,required=True)
    parser.add_argument("--checkpoints",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--draws",type=int,default=32)
    args=parser.parse_args()
    if not 1<=args.draws<=256 or args.output.exists():
        parser.error("Use 1..256 dice draws and a new output file")
    game_root(args.driftmoor_root)
    from spider_sim.dual_encounter import ACTORS
    torch.set_num_threads(1)
    rows=[]
    for role in ACTORS:
        agent=load(args.checkpoints/("gru-"+role+".pt"))
        for i,case in enumerate(CASES):
            world=fixture(role,case,230000+i)
            world.audit_case=case
            result=analyze(world,role,agent,args.draws)
            rows.append(result)
            print(role,case,result["learned_cost_action"],result["reference_cost_action"],flush=True)
    report={"method":"Frozen one-step predictor versus repeated actual counterfactual outcomes",
        "draws_per_action":args.draws,"fixture_seeds":list(range(230000,230000+len(CASES))),
        "dice_seeds":list(range(240000,240000+args.draws)),"cases":rows,
        "limitations":["Offline privileged reference is diagnostic only, never actor input",
            "Cold memory and fresh teacher response; not arbitrary encountered histories",
            "Finite dice sample; different branches can consume different random draws",
            "Same cost and policy prior used for predicted and reference rankings",
            "No training, cost tuning, or deployment changes"]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")

if __name__=="__main__":
    main()
