"""Observation-only spider controllers and numeric model interfaces."""
from spider_sim.runtime import game_root
game_root()
import math

from models.body_geometry import Body, Orientation, body_sockets

ACTIONS = ("wait", "orient", "move", "attack")
FEATURE_SIZE = 17
ACTION_SIZE = 10
INPUT_SIZE = FEATURE_SIZE * 2 + ACTION_SIZE
TARGET_SIZE = FEATURE_SIZE + 1  # next observation + admitted attack label


def contact(observation):
    visible = [c for c in observation["contacts"] if c["modality"] == "vision"]
    return min(visible, key=lambda c: (c["range_ft"], c["track_id"]), default=None)


def features(observation):
    c = contact(observation)
    own = observation["self"]
    roll,pitch,heading = own["orientation_roll_pitch_heading_deg"]
    angles = (math.radians(roll), math.radians(pitch), math.radians(heading))
    bearing = math.radians(c["bearing_deg"]) if c else 0
    elevation = math.radians(c["elevation_angle_deg"]) if c else 0
    x,y = own["anchor_xy"]
    return [
        float(c is not None), c["range_ft"]/60 if c else 0,
        math.cos(bearing) if c else 0, math.sin(bearing) if c else 0,
        math.sin(elevation) if c else 0, math.cos(elevation) if c else 0,
        *[value for angle in angles for value in (math.sin(angle),math.cos(angle))],
        own["height_above_floor_ft"]/10, own["hp"]/26,
        (x-64)/60, (y-195)/60, min(len(observation["contacts"]),4)/4,
    ]


def candidates(observation):
    """All geometric estimates use observations, own body, and a declared target proxy."""
    actions = [{"type": "wait"}]
    c = contact(observation)
    if c is None:
        return actions
    own = observation["self"]
    roll,pitch,heading = own["orientation_roll_pitch_heading_deg"]
    r = c["range_ft"]
    bearing = math.radians(c["bearing_deg"])
    elevation = math.radians(c["elevation_angle_deg"])
    horizontal = r*math.cos(elevation)
    dx,dy,dz = horizontal*math.cos(bearing),horizontal*math.sin(bearing),r*math.sin(elevation)
    # Aim toward a generic humanoid center, not the unseen target's real box.
    desired_pitch = -math.degrees(math.atan2(dz+2.75,max(horizontal,0.01)))
    actions.append({
        "type": "orient",
        "orientation": [roll, round(desired_pitch/5)*5, c["bearing_deg"]],
    })
    distance = min(8.0,max(0.0,horizontal-5))
    if distance > 0.5:
        x,y = own["anchor_xy"]
        actions.append({
            "type": "move",
            "destination": [x+distance*math.cos(bearing),y+distance*math.sin(bearing)],
        })
    actions.append({"type": "attack", "target_id": c["track_id"]})
    return actions


def action_features(observation, action):
    kind = action["type"]
    vector = [float(kind == name) for name in ACTIONS]
    own = observation["self"]
    if kind == "move":
        x,y = own["anchor_xy"]
        vector += [(action["destination"][0]-x)/10,(action["destination"][1]-y)/10]
    else:
        vector += [0,0]
    # Absolute intended orientation; normalized changes are sufficient for this model.
    pose = action.get("orientation",own["orientation_roll_pitch_heading_deg"])
    current = own["orientation_roll_pitch_heading_deg"]
    vector += [((a-b+180)%360-180)/180 for a,b in zip(pose,current)]
    vector += [float(kind=="attack")]
    return vector


def model_input(previous, observation, action):
    return features(previous) + features(observation) + action_features(observation, action)


def choose_rule(observation, previous=None):
    actions = candidates(observation)
    c = contact(observation)
    if c is None:
        return actions[0]
    own = observation["self"]
    angles = own["orientation_roll_pitch_heading_deg"]
    bearing = math.radians(c["bearing_deg"])
    elevation = math.radians(c["elevation_angle_deg"])
    r = c["range_ft"]
    target_anchor = (
        r*math.cos(elevation)*math.cos(bearing),
        r*math.cos(elevation)*math.sin(bearing),
        r*math.sin(elevation),
    )
    dimensions = own["body_dimensions_ft"]
    body = Body(*dimensions,"observed proprioception")
    estimated = body_sockets(body)["bite"].evaluate(
        (0,0,0), Orientation(*angles), Body(2,1,5.5,"declared target estimate"),
        target_anchor, Orientation(),
    )
    attack = next(a for a in actions if a["type"]=="attack")
    if estimated["facing"] and estimated["distance_ft"] <= own["bite_reach_ft"]:
        return attack
    orient = next(a for a in actions if a["type"]=="orient")
    error = max(abs((a-b+180)%360-180) for a,b in zip(angles,orient["orientation"]))
    if error > 10:
        return orient
    return next((a for a in actions if a["type"]=="move"), actions[0])


def choose_learned(observation, previous, predictor):
    actions = candidates(observation)
    outputs = predictor([model_input(previous,observation,a) for a in actions])
    scores = []
    for action, output in zip(actions, outputs):
        # The learned dynamics predict contact range; the head predicts whether
        # the engine will admit a bite, not the random dice result.
        presence = max(0,min(1,output[0]))
        predicted_range = max(0,min(1.5,output[1]))
        admitted = max(0,min(1,output[-1]))
        score = -0.25*predicted_range - 0.2*(1-presence)
        if action["type"] == "attack":
            score += 2*admitted - 0.6*(1-admitted)
        if action["type"] == "move":
            score -= 0.06
        if action["type"] == "orient":
            score -= 0.025
        scores.append(score)
    return actions[max(range(len(actions)),key=lambda i:scores[i])]
