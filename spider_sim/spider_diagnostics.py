"""Privileged counterfactual diagnostics; never used as policy observations."""
from spider_sim.runtime import game_root
game_root()
import copy
from collections import Counter

from spider_sim.spider_controller import candidates, choose_rule, model_input, features
from spider_sim.spider_learning import setup, execute, player_command
from spider_sim.spider_experiment import SPIDER, PLAYER


def score(action, output):
    # Exact v1 objective, repeated here to keep the frozen policy unchanged.
    presence = max(0,min(1,output[0]))
    distance = max(0,min(1.5,output[1]))
    value = -0.25*distance-0.2*(1-presence)
    if action["type"]=="attack":
        admitted = max(0,min(1,output[-1]))
        value += 2*admitted-0.6*(1-admitted)
    if action["type"]=="move":
        value -= 0.06
    if action["type"]=="orient":
        value -= 0.025
    return value


def counterfactual(experiment, environment_rng, turn, family, action, *, probe_followup=False):
    """Resolve on a deep copy, including RNG, leaving the real episode alone."""
    clone = copy.deepcopy(experiment)
    local_rng = copy.deepcopy(environment_rng)
    result = execute(clone,action)
    combat = result["outcome"]["combat"] or {}
    if combat.get("attempted"):
        clone.active.add(SPIDER)
    command = player_command(clone,turn,family,local_rng)
    if clone.config.characters[PLAYER].character.current_hp > 0:
        clone.step(PLAYER,command["type"],command.get("destination"))
    next_features = features(clone.observe())
    next_features.append(float(combat.get("attempted",False)))
    followup = None
    if probe_followup:
        second = execute(clone,{"type":"attack","target_id":PLAYER})
        second_combat = second["outcome"]["combat"] or {}
        followup = {
            "admitted":bool(second_combat.get("attempted",False)),
            "status":second["outcome"]["status"],
        }
    return {
        "followup_attack":followup,
        "next":next_features,
        "actual_admitted":bool(combat.get("attempted",False)),
        "status":result["outcome"]["status"],
        "actual_score":score(action,next_features),
    }


def trace(seed, predictor, steps=14):
    experiment,rng = setup(seed)
    family = ("approach","retreat","pause","sidestep","feint")[seed%5]
    previous = experiment.observe()
    records = []
    for turn in range(steps):
        if experiment.config.characters[PLAYER].character.current_hp <= 0:
            break
        observation = experiment.observe()
        options = candidates(observation)
        inputs = [model_input(previous,observation,a) for a in options]
        predictions = predictor(inputs)
        evaluated = []
        for action,x,prediction in zip(options,inputs,predictions):
            evaluated.append({
                "action":action,"input":x,
                "predicted":prediction,"predicted_score":score(action,prediction),
                **counterfactual(experiment,rng,turn,family,action,probe_followup=True),
            })
        selected = max(range(len(options)),key=lambda i:evaluated[i]["predicted_score"])
        oracle = max(range(len(options)),key=lambda i:evaluated[i]["actual_score"])
        row = execute(experiment,options[selected])
        combat = row["outcome"]["combat"] or {}
        if combat.get("attempted"):
            experiment.active.add(SPIDER)
        command = player_command(experiment,turn,family,rng)
        if experiment.config.characters[PLAYER].character.current_hp > 0:
            experiment.step(PLAYER,command["type"],command.get("destination"))
        # Check that the diagnostic fork predicted the actual engine outcome.
        assert evaluated[selected]["status"] == row["outcome"]["status"]
        assert evaluated[selected]["actual_admitted"] == bool(combat.get("attempted",False))
        records.append({
            "seed":seed,"turn":turn,"family":family,
            "observation":observation,"candidates":evaluated,
            "selected":selected,"oracle_one_step":oracle,
            "rule_action":choose_rule(observation,previous),
        })
        previous = observation
    return records


def diagnose(predictor,start=10000,count=30):
    traces = [r for seed in range(start,start+count) for r in trace(seed,predictor)]
    chosen = Counter()
    oracle = Counter()
    rule = Counter()
    attacks = []
    all_candidate_attacks = []
    regret = []
    for row in traces:
        selected = row["candidates"][row["selected"]]
        best = row["candidates"][row["oracle_one_step"]]
        chosen[selected["action"]["type"]] += 1
        oracle[best["action"]["type"]] += 1
        rule[row["rule_action"]["type"]] += 1
        regret.append(best["actual_score"]-selected["actual_score"])
        for candidate in row["candidates"]:
            if candidate["action"]["type"]=="attack":
                all_candidate_attacks.append(candidate)
        if selected["action"]["type"]=="attack":
            attacks.append(selected)
    def calibration(rows):
        if not rows:
            return {}
        return {
            "count":len(rows),
            "admitted":sum(r["actual_admitted"] for r in rows),
            "mean_predicted_probability":sum(r["predicted"][-1] for r in rows)/len(rows),
            "brier":sum((r["predicted"][-1]-r["actual_admitted"])**2 for r in rows)/len(rows),
            "predicted_at_least_half":sum(r["predicted"][-1]>=0.5 for r in rows),
            "rejection_reasons":dict(Counter(r["status"] for r in rows if not r["actual_admitted"])),
        }
    preparation = {"orient_enables_bite_but_wait_does_not":0,
                   "move_enables_bite_but_wait_does_not":0,
                   "examples":[]}
    for row in traces:
        by_type = {c["action"]["type"]:c for c in row["candidates"]}
        wait = by_type["wait"]
        for kind in ("orient","move"):
            candidate = by_type.get(kind)
            if candidate and candidate["followup_attack"]["admitted"] and not wait["followup_attack"]["admitted"]:
                preparation[kind+"_enables_bite_but_wait_does_not"] += 1
                if len(preparation["examples"]) < 5:
                    preparation["examples"].append({
                        "seed":row["seed"],"turn":row["turn"],"preparation":kind,
                        "chosen":row["candidates"][row["selected"]]["action"]["type"],
                        "one_step_wait_score":wait["actual_score"],
                        "one_step_preparation_score":candidate["actual_score"],
                    })
    return {
        "summary":{
            "preparation_probe":preparation,
            "validation_seeds":[start,start+count-1],"steps":len(traces),
            "chosen_actions":dict(chosen),"oracle_one_step_actions":dict(oracle),
            "rule_actions_on_same_states":dict(rule),
            "chosen_attacks":calibration(attacks),
            "all_candidate_attacks":calibration(all_candidate_attacks),
            "mean_actual_score_regret":sum(regret)/max(1,len(regret)),
            "score_agreement":sum(r["selected"]==r["oracle_one_step"] for r in traces)/max(1,len(traces)),
            "oracle_scope":"Privileged exact one-step outcome under the same v1 objective; diagnostic only",
        },
        "traces":traces,
    }
