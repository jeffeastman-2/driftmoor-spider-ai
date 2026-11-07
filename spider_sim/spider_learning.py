"""Seeded observation-driven episodes around the actual cavern mechanics."""
from spider_sim.runtime import game_root
game_root()
import random

from spider_sim.spider_experiment import ResolvedCavernExperiment, SPIDER, PLAYER
from spider_sim.spider_controller import (
    candidates, choose_rule, choose_learned, model_input, features,
)
from models.body_geometry import Orientation
from models.space_time import Position


def execute(experiment, action):
    return experiment.step(
        SPIDER, action["type"], action.get("destination"),
        target_id=action.get("target_id"),
        orientation=Orientation(*action["orientation"]) if "orientation" in action else None,
    )


def setup(seed):
    rng = random.Random(seed)
    experiment = ResolvedCavernExperiment(seed=seed)
    # Explicit initial conditions only; subsequent motion is always resolved.
    x,y = rng.uniform(130,150),rng.uniform(192,198)
    experiment.positions[PLAYER] = (x,y,experiment.geometry.zone_at(
        Position(x,y)) or "")
    experiment.orientations[SPIDER] = Orientation(
        roll=180,pitch=rng.uniform(-25,25),heading=rng.uniform(0,360),
    )
    return experiment, rng


def player_command(experiment, turn, family, rng):
    x,y,_ = experiment.positions[PLAYER]
    if family == "retreat" and turn >= 5:
        target = (min(150,x+12),195)
    elif family == "pause" and 3 <= turn <= 6:
        return {"type":"wait"}
    elif family == "sidestep" and turn >= 5:
        target = (max(63,x-7),185 if turn%2 else 205)
    elif family == "feint":
        direction = -1 if turn%6 < 3 else 1
        target = (max(63,min(145,x+direction*rng.uniform(5,12))),195)
    else:
        target = (max(63,x-rng.uniform(5,12)),195+rng.uniform(-3,3))
    return {"type":"move","destination":target}


def episode(seed, policy="rule", predictor=None, steps=14, family=None):
    experiment,rng = setup(seed)
    family = family or ("approach","retreat","pause","sidestep","feint")[seed%5]
    previous = experiment.observe()
    records = []
    stats = {
        "attacks":0,"admitted_attacks":0,"hits":0,"damage":0,"rejected_attacks":0,
        "pre_entry_orientations":0,"movement_ft":0.0,"steps":0,
    }
    entry_triggered = False
    for turn in range(steps):
        if experiment.config.characters[PLAYER].character.current_hp <= 0:
            break
        # Controller only receives current/previous filtered observations.
        observation = experiment.observe()
        if policy == "explore":
            possible = candidates(observation)
            action = rng.choice(possible) if rng.random()<0.7 else choose_rule(observation,previous)
        elif policy == "zone":
            action = choose_rule(observation,previous) if entry_triggered else {"type":"wait"}
        elif policy == "rule":
            action = choose_rule(observation,previous)
        elif policy == "learned":
            if predictor is None:
                raise ValueError("Learned policy needs a predictor")
            action = choose_learned(observation,previous,predictor)
        else:
            raise ValueError("Unknown policy")
        row = execute(experiment,action)
        combat = row["outcome"]["combat"] or {}
        if action["type"]=="attack":
            stats["attacks"] += 1
            stats["admitted_attacks"] += int(combat.get("attempted",False))
            stats["hits"] += int(combat.get("hit") is True)
            stats["damage"] += combat.get("damage",0)
            stats["rejected_attacks"] += int(not combat.get("attempted",False))
            # An executed attack reveals the experimental spider. The original
            # zone-trigger comparator is tracked independently below.
            if combat.get("attempted"):
                experiment.active.add(SPIDER)
        stats["pre_entry_orientations"] += int(action["type"]=="orient" and not entry_triggered)
        stats["movement_ft"] += row["outcome"]["distance_moved_ft"]
        # Opponent command is an exogenous intervention, never a model input.
        player_action = player_command(experiment,turn,family,rng)
        if experiment.config.characters[PLAYER].character.current_hp > 0:
            experiment.step(PLAYER,player_action["type"],player_action.get("destination"))
        entry_triggered |= experiment.positions[PLAYER][2].upper() == "SECRET CHAMBER"
        after = experiment.observe()
        target = features(after)+[float(combat.get("attempted",False))]
        records.append({
            "episode_id":seed,"family":family,"turn":turn,"policy":policy,
            "input":model_input(previous,observation,action),"target":target,
            "observer_before":observation,"action":action,"observer_after":after,
            "feedback":{k:combat.get(k) for k in ("attempted","hit","damage")},
            # No engine ground truth, player actions, or geometric debug fields.
        })
        previous = observation
        stats["steps"] += 1
    stats["movement_ft"] = round(stats["movement_ft"],2)
    return records,stats


def collect(count=160,start_seed=0):
    records = []
    for seed in range(start_seed,start_seed+count):
        rows,_ = episode(seed,policy="explore")
        records.extend(rows)
    return records


def evaluate(seeds, predictor=None):
    policies = ("zone","rule","learned") if predictor else ("zone","rule")
    result = {}
    for policy in policies:
        totals = {}
        episodes = []
        for seed in seeds:
            _,stats = episode(seed,policy=policy,predictor=predictor)
            episodes.append({"seed":seed,**stats})
            for key,value in stats.items():
                totals[key] = totals.get(key,0)+value
        result[policy] = {"totals":totals,"episodes":episodes}
    return result
