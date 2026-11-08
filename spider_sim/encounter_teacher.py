"""Observation-only teachers; no environment or opponent-state access."""
import math
from spider_sim.runtime import game_root
game_root()
from models.body_geometry import Body, Orientation, ActionSocket

SPIDER = "giant_spider"
PLAYER = "gilthaness"
ACTIONS = ("wait","advance","retreat","face","flee","look_up","look_down","attack")

def angle_difference(a,b):
    return (a-b+180)%360-180

class LegacyTeacher:
    """Demonstration memory built only from this actor's observations."""
    def __init__(self):
        self.contact = None
        self.age = 100

    def update(self,observation):
        self.age += 1
        if observation["contact"]:
            self.contact = dict(observation["contact"])
            self.age = 0
        if self.age>6:
            self.contact = None
        if self.contact is None:
            return [0,0,0,0,0]
        c = self.contact
        bearing = math.radians(c["bearing_deg"])
        return [1,c["range_ft"]/60,math.cos(bearing),math.sin(bearing),
                (c.get("elevation_deg") or 0)/90]

    def choose(self,actor,observation):
        own = observation["self"]
        # Distinct authored survival thresholds; learned through imitation.
        threshold = 0.35 if actor==SPIDER else 0.45
        if own["hp_fraction"]<threshold:
            return ACTIONS.index("retreat")
        c = observation["contact"] or self.contact
        roll,pitch,heading = own["orientation_roll_pitch_heading_deg"]
        if c:
            delta = angle_difference(c["bearing_deg"],heading)
            if abs(delta)>20:
                return ACTIONS.index("face")
            desired = -(c.get("elevation_deg") or 0)
            if desired-pitch>20:
                return ACTIONS.index("look_down")
            if desired-pitch<-20:
                return ACTIONS.index("look_up")
            visible = observation["contact"] and observation["contact"]["source"]=="vision"
            if visible and c["range_ft"]<=15:
                return ACTIONS.index("attack")
            return ACTIONS.index("advance" if actor==PLAYER or c["range_ft"]>15 else "wait")
        if actor==PLAYER:
            if observation["web_evidence"] and pitch>-30:
                return ACTIONS.index("look_up")
            return ACTIONS.index("advance")
        return ACTIONS.index("wait")



class SensoryTeacher(LegacyTeacher):
    """Use an observed body proxy and own weapon profile, never true target pose."""

    def update(self, observation):
        # Last contact is stationary in the remembered frame, not assumed to
        # follow our movement. This compensates ego motion, not opponent motion.
        if observation["contact"] is None and self.contact and self.anchor:
            now = observation["self"]["anchor_xy"]
            dx, dy = self.anchor[0]-now[0], self.anchor[1]-now[1]
            bearing = math.radians(self.contact["bearing_deg"])
            elevation = math.radians(self.contact.get("elevation_deg") or 0)
            distance = self.contact["range_ft"]
            x = distance*math.cos(elevation)*math.cos(bearing)+dx
            y = distance*math.cos(elevation)*math.sin(bearing)+dy
            z = distance*math.sin(elevation)
            self.contact = dict(self.contact, range_ft=math.sqrt(x*x+y*y+z*z),
                                bearing_deg=math.degrees(math.atan2(y,x))%360,
                                elevation_deg=math.degrees(math.atan2(z,math.hypot(x,y))))
        self.anchor = list(observation["self"]["anchor_xy"])
        belief = super().update(observation)
        if self.contact:
            self.escape_heading = (self.contact["bearing_deg"]+180)%360
        return belief

    def __init__(self,fight_when_blocked=True):
        super().__init__()
        self.fight_when_blocked = fight_when_blocked
        self.anchor = None
        self.escape_heading = None

    @staticmethod
    def estimate(observation, contact, pose=None, shift=(0,0,0)):
        """Proxy body is constructed entirely from measured center and extents."""
        own = observation["self"]
        profile = own["attack_profile"]
        pose = pose or Orientation(*own["orientation_roll_pitch_heading_deg"])
        b = math.radians(contact["bearing_deg"])
        e = math.radians(contact.get("elevation_deg") or 0)
        r = contact["range_ft"]
        center = (r*math.cos(e)*math.cos(b), r*math.cos(e)*math.sin(b), r*math.sin(e))
        width, depth, height = contact["dimensions_ft"]
        body = Body(width,depth,height,"observed world-axis bounding size")
        anchor = (center[0],center[1],center[2]-height/2)
        socket = ActionSocket(tuple(profile["origin_local_ft"]),(1,0,0),45)
        shape = socket.evaluate(shift,pose,body,anchor,Orientation())
        delta = [a-b for a,b in zip(shape["aim_xyz"],shape["origin_xyz"])]
        shape["desired_pitch"] = -math.degrees(math.atan2(delta[2],math.hypot(*delta[:2])))
        return shape

    def intent(self, actor, observation):
        """Authored survival preferences, separate from the eight motor outputs."""
        hp = observation["self"]["hp_fraction"]
        if hp < (0.20 if actor==SPIDER else 0.25):
            if actor==SPIDER and self.fight_when_blocked and observation["self"].get("escape_blocked",False):
                return "cornered"
            return "flee"
        if hp < (0.35 if actor==SPIDER else 0.45):
            return "retreat"
        return "engage"

    def decision(self, actor, observation):
        intent = self.intent(actor,observation)
        action = self.choose(actor,observation)
        pose = action_orientation(observation,action,
                                  contact=observation["contact"] or self.contact,
                                  escape_heading=self.escape_heading)
        return {"intent":intent,"action":action,"orientation":pose}

    def choose(self, actor, observation):
        own = observation["self"]
        c = observation["contact"] or self.contact
        roll,pitch,heading = own["orientation_roll_pitch_heading_deg"]
        intent = self.intent(actor,observation)
        if intent=="flee":
            away = (c["bearing_deg"]+180)%360 if c else self.escape_heading
            return ACTIONS.index("flee" if away is not None else "wait")
        if intent=="retreat":
            return ACTIONS.index("retreat" if c else "wait")
        if intent=="cornered" and c and c.get("dimensions_ft"):
            visible = observation["contact"] and observation["contact"]["source"]=="vision"
            aimed = action_orientation(observation,ACTIONS.index("attack"),contact=c)
            shape = self.estimate(observation,c,Orientation(*aimed))
            if (visible and own["attack_profile"]["reach_ft"]>0 and shape["facing"]
                and shape["distance_ft"]<=own["attack_profile"]["reach_ft"]+0.15):
                return ACTIONS.index("attack")
        if not c:
            if actor==PLAYER:
                if observation["web_evidence"] and pitch>-30:
                    return ACTIONS.index("look_up")
                return ACTIONS.index("advance")
            return ACTIONS.index("wait")
        delta = angle_difference(c["bearing_deg"],heading)
        if abs(delta)>20:
            return ACTIONS.index("face")
        # Vibration gives neither body shape nor elevation; it cannot authorize
        # a bite or a pitch correction. Search toward the last measured bearing.
        if not c.get("dimensions_ft"):
            return ACTIONS.index("advance" if c["range_ft"]>8 else "wait")
        shape = self.estimate(observation,c)
        visible = observation["contact"] and observation["contact"]["source"]=="vision"
        reach = own["attack_profile"]["reach_ft"]
        # Reserve a small margin inside the engine's reach+0.5 tolerance.
        if visible and reach>0 and shape["facing"] and shape["distance_ft"]<=reach+0.15:
            return ACTIONS.index("attack")
        error = shape["desired_pitch"]-pitch
        if abs(error)>20:
            desired = max(-90,min(90,pitch+(30 if error>0 else -30)))
            candidate = Orientation(roll,desired,heading)
            if self.estimate(observation,c,candidate)["distance_ft"] <= shape["distance_ft"]+1:
                return ACTIONS.index("look_down" if error>0 else "look_up")
        a = math.radians(heading)
        shifted = self.estimate(observation,c,shift=(6*math.cos(a),6*math.sin(a),0))
        if shifted["distance_ft"]+0.1 < shape["distance_ft"]:
            return ACTIONS.index("advance")
        return ACTIONS.index("wait")


# Existing callers use ContactMemory; retain the original teacher explicitly.
ContactMemory = SensoryTeacher


def action_orientation(observation, action, *, contact=None, escape_heading=None):
    """Decode a whole motion command using only private evidence.

    Facing changes are part of the move, not extra turn slots. No target
    position is consulted. Without evidence, preserve own current orientation.
    """
    name = ACTIONS[action]
    if name not in ("advance","retreat","face","flee","attack"):
        return None
    roll,pitch,heading = observation["self"]["orientation_roll_pitch_heading_deg"]
    c = contact if contact is not None else observation["contact"]
    if name=="flee":
        heading = (c["bearing_deg"]+180)%360 if c else (
            escape_heading if escape_heading is not None else heading)
        pitch = 0
    elif c:
        heading = c["bearing_deg"]
        if name in ("retreat","face","attack"):
            if c.get("dimensions_ft"):
                pose = Orientation(roll,pitch,heading)
                pitch = SensoryTeacher.estimate(observation,c,pose)["desired_pitch"]
            elif c.get("elevation_deg") is not None:
                pitch = -c["elevation_deg"]
    return [roll,max(-90,min(90,pitch)),heading%360]


def predicted_orientation(observation, action, belief):
    """Use current measurements or the network's own predicted contact memory."""
    c = observation["contact"]
    if c is None and belief[0]>0.5:
        c = {"range_ft":max(0,belief[1]*60),
             "bearing_deg":math.degrees(math.atan2(belief[3],belief[2]))%360,
             "elevation_deg":max(-90,min(90,belief[4]*90)),
             "dimensions_ft":None}
    return action_orientation(observation,action,contact=c)
