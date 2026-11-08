"""Privileged offline cost diagnostic. Never a deployable creature controller."""
import copy
import random
import torch
from experiments.jepa_planner import predicted_cost,Planner


def response(world,role,action,belief,opponent,mode):
    from spider_sim.dual_encounter import ACTORS
    from spider_sim.encounter_teacher import predicted_orientation
    obs=world.observation_for(role)
    result=world.act(role,action,orientation=predicted_orientation(obs,action,belief))
    other=next(cid for cid in ACTORS if cid!=role)
    if all(world.config.characters[cid].character.current_hp>0 for cid in ACTORS):
        world.begin_turn(other)
        if mode=="teacher":
            choice,_=opponent.choose_observation(other,world.observation_for(other))
            world.act(other,choice,orientation=opponent.last_orientation)
        else:
            world.act(other,0)
    world.begin_turn(role)
    future=world.observation_for(role)["values"]+[float(result["admitted"]),result["damage"]/20]
    return future,result


def reference_scores(world,role,belief,opponent,mode,draws,dice_start):
    """Mean sample cost, not cost of mean prediction. No live RNG peeking."""
    current=world.observation_for(role)
    values=torch.tensor(current["values"])
    sampled=[]
    for draw in range(draws):
        futures=[]
        for action in range(8):
            branch=copy.deepcopy(world);teacher=copy.deepcopy(opponent)
            branch.random.setstate(random.Random(dice_start+draw).getstate())
            future,_=response(branch,role,action,belief,teacher,mode)
            futures.append(future)
        sampled.append(torch.tensor(futures))
    costs=torch.stack([predicted_cost(values,f,condition=current["self"]["condition"])
                       for f in sampled]).mean(0)
    return costs,torch.stack(sampled).mean(0)


def run_episode(agent,role,case,seed,controller,draws=8,rounds=6):
    from experiments.audit_jepa_planning import fixture
    from spider_sim.dual_encounter import ACTORS,ACTIONS
    from spider_sim.encounter_evaluation import TeacherAgent
    world=fixture(role,case,seed)
    session=Planner(agent).session();opponent=TeacherAgent().session()
    mode="teacher" if case.endswith("teacher") else "idle"
    other=next(cid for cid in ACTORS if cid!=role)
    initial={cid:world.config.characters[cid].character.current_hp for cid in ACTORS}
    trace=[]
    for turn in range(rounds):
        if any(world.config.characters[cid].character.current_hp<=0 for cid in ACTORS):
            break
        world.begin_turn(role);obs=world.observation_for(role)
        learned,belief=session.choose_observation(role,obs)
        plan=session.last_plan
        record={"observation":obs,"learned_cost_action":ACTIONS[learned]}
        if controller.startswith("engine"):
            costs,futures=reference_scores(world,role,belief,opponent,mode,draws,
                630000+(seed-610000)*1000+turn*draws)
            prior=torch.tensor(plan["scores"])-torch.tensor(plan["costs"])
            scores=costs+(prior if controller=="engine_prior" else 0)
            action=int(scores.argmin())
            record.update({"costs":costs.tolist(),"scores":scores.tolist(),
                "expected_admission":futures[:,41].tolist(),"expected_damage":(futures[:,42]*20).tolist()})
        else:
            action=plan["policy_action"] if controller=="policy" else learned
        future,result=response(world,role,action,belief,opponent,mode)
        record.update({"action":ACTIONS[action],"result":result,"next_hp_fraction":future[16]})
        trace.append(record)
    final={cid:world.config.characters[cid].character.current_hp for cid in ACTORS}
    return {"role":role,"case":case,"seed":seed,"controller":controller,"turns":trace,
        "stats":{"attacks":sum(t["action"]=="attack" for t in trace),
          "admitted":sum(t["result"]["admitted"] for t in trace),
          "rejected_attacks":sum(t["action"]=="attack" and not t["result"]["admitted"] for t in trace),
          "own_damage_received":initial[role]-final[role],"opponent_hp_lost":initial[other]-final[other],
          "survived":int(final[role]>0),"opponent_defeated":int(final[other]<=0),
          "first_action":trace[0]["action"] if trace else None,"final_hp":final[role]}}
