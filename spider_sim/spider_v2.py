"""V2: covered action data and direct two-step action-conditioned prediction."""
from spider_sim.runtime import game_root
game_root()
import random
import math

from spider_sim.spider_learning import setup, execute, player_command
from spider_sim.spider_controller import candidates, choose_rule, choose_learned, model_input
from spider_sim.spider_diagnostics import counterfactual, score
from spider_sim.spider_experiment import ResolvedCavernExperiment, SPIDER, PLAYER
from models.body_geometry import Orientation
from models.space_time import Position


def predict_candidates(observation, previous, predictor, horizon=2):
    options = candidates(observation)
    predictions = predictor([model_input(previous,observation,a) for a in options])
    values = []
    for action,prediction in zip(options,predictions):
        # Same v1 immediate reward; last two outputs are immediate and
        # follow-up bite admission probabilities. The second action is either
        # bite or wait, so its best expected attack reward is at least zero.
        immediate = score(action,prediction[:17]+[prediction[17]])
        future = max(0,2.6*max(0,min(1,prediction[18]))-0.6)
        values.append(immediate+(0.8*future if horizon==2 else 0))
    return options[max(range(len(options)),key=lambda i:values[i])]


def covered_rows(experiment,rng,turn,family,previous,episode_id):
    observation = experiment.observe()
    rows = []
    for action in candidates(observation):
        outcome = counterfactual(experiment,rng,turn,family,action,probe_followup=True)
        rows.append({
            "episode_id":episode_id,"turn":turn,
            "input":model_input(previous,observation,action),
            "target":outcome["next"]+[float(outcome["followup_attack"]["admitted"])],
            "action_type":action["type"],
        })
    return rows


def collect_covered(count,start_seed,v1_predictor):
    rows = []
    for seed in range(start_seed,start_seed+count):
        experiment,rng = setup(seed)
        behavior_rng = random.Random(seed+700000)
        family = ("approach","retreat","pause","sidestep","feint")[seed%5]
        previous = experiment.observe()
        for turn in range(14):
            if experiment.config.characters[PLAYER].character.current_hp <= 0:
                break
            observation = experiment.observe()
            rows.extend(covered_rows(experiment,rng,turn,family,previous,seed))
            if seed%3==0:
                action = choose_learned(observation,previous,v1_predictor)
            elif seed%3==1:
                action = choose_rule(observation,previous)
            else:
                action = behavior_rng.choice(candidates(observation))
            result = execute(experiment,action)
            if (result["outcome"]["combat"] or {}).get("attempted"):
                experiment.active.add(SPIDER)
            command = player_command(experiment,turn,family,rng)
            if experiment.config.characters[PLAYER].character.current_hp > 0:
                experiment.step(PLAYER,command["type"],command.get("destination"))
            previous = observation
    return rows


def collect_pose_pairs(count,start_seed):
    """Matched near-target body poses, grouped under one ID per initial scene.

    Explicit scene initializations vary target distance and pose; all labels
    come from real engine forks. No private geometry becomes an input.
    """
    rows = []
    for seed in range(start_seed,start_seed+count):
        randomizer = random.Random(seed)
        base = ResolvedCavernExperiment(seed=seed)
        x,y = randomizer.uniform(60,78),randomizer.uniform(185,205)
        base.positions[PLAYER] = (x,y,base.geometry.zone_at(Position(x,y)) or "")
        bearing = math.degrees(math.atan2(y-195,x-64))
        roll = randomizer.choice((160,180,200))
        # Matched forward/backward/head-down examples at identical positions.
        for variant,pose in enumerate((
            Orientation(roll=roll,pitch=randomizer.uniform(-30,30),heading=bearing),
            Orientation(roll=roll,pitch=randomizer.uniform(-30,30),heading=bearing+180),
            Orientation(roll=roll,pitch=randomizer.uniform(45,90),heading=bearing),
        )):
            base.orientations[SPIDER] = pose
            observation = base.observe()
            branch = covered_rows(base,random.Random(seed),4,"pause",observation,seed)
            for row in branch:
                row["pose_variant"] = variant
                row["source"] = "matched_pose"
            rows.extend(branch)
    return rows


def evaluate_v2(seeds,v1_predictor,v2_predictor):
    results = {}
    for policy in ("zone","rule","v1","v2_one_step","v2_two_step"):
        totals = {}
        details = []
        for seed in seeds:
            experiment,rng = setup(seed)
            family = ("approach","retreat","pause","sidestep","feint")[seed%5]
            previous = experiment.observe()
            entered = False
            stats = {
                "attacks":0,"admitted_attacks":0,"hits":0,"damage":0,
                "rejected_attacks":0,"orientations":0,"movement_ft":0.0,
            }
            for turn in range(14):
                if experiment.config.characters[PLAYER].character.current_hp<=0:
                    break
                observation = experiment.observe()
                if policy=="zone":
                    action = choose_rule(observation,previous) if entered else {"type":"wait"}
                elif policy=="rule":
                    action = choose_rule(observation,previous)
                elif policy=="v1":
                    action = choose_learned(observation,previous,v1_predictor)
                else:
                    action = predict_candidates(
                        observation,previous,v2_predictor,
                        horizon=1 if policy=="v2_one_step" else 2,
                    )
                result = execute(experiment,action)
                combat = result["outcome"]["combat"] or {}
                stats["attacks"] += int(action["type"]=="attack")
                stats["admitted_attacks"] += int(combat.get("attempted",False))
                stats["rejected_attacks"] += int(action["type"]=="attack" and not combat.get("attempted",False))
                stats["hits"] += int(combat.get("hit") is True)
                stats["damage"] += combat.get("damage",0)
                stats["orientations"] += int(action["type"]=="orient")
                stats["movement_ft"] += result["outcome"]["distance_moved_ft"]
                if combat.get("attempted"):
                    experiment.active.add(SPIDER)
                command = player_command(experiment,turn,family,rng)
                if experiment.config.characters[PLAYER].character.current_hp>0:
                    experiment.step(PLAYER,command["type"],command.get("destination"))
                entered |= experiment.positions[PLAYER][2].upper()=="SECRET CHAMBER"
                previous = observation
            stats["movement_ft"] = round(stats["movement_ft"],2)
            details.append({"seed":seed,**stats})
            for key,value in stats.items():
                totals[key] = totals.get(key,0)+value
        results[policy] = {"totals":totals,"episodes":details}
    return results
