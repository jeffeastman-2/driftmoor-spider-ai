"""Optional private-sensory approach motor primitive; no authoritative target query."""
MOTOR_SCHEMA="approach-advance-v1"

import math
from spider_sim.dual_encounter import DualEncounter
from spider_sim.encounter_teacher import SensoryTeacher


def approach_distance(observation,pose,speed):
    contact=observation["contact"]
    if not contact or not contact.get("dimensions_ft"):
        return speed  # Preserve exploratory motion when body geometry is unknown.
    reach=observation["self"]["attack_profile"]["reach_ft"]
    if reach<=0 or speed<=0:
        return 0.
    heading=math.radians(pose.heading)
    def distance(step):
        shape=SensoryTeacher.estimate(observation,contact,pose,
             shift=(step*math.cos(heading),step*math.sin(heading),0))
        return shape["distance_ft"]
    initial=distance(0)
    target=max(0.,reach-.25)
    if initial<=target:
        return 0.
    # Bounded search along the declared heading using the measured body proxy.
    # First entry into reach prevents overshooting through to the far side.
    best_step,best_distance=0.,initial
    previous=0.
    for i in range(1,math.ceil(speed/.25)+1):
        step=min(speed,i*.25);gap=distance(step)
        if gap<=target:
            low,high=previous,step
            for _ in range(16):
                middle=(low+high)/2
                if distance(middle)<=target: high=middle
                else: low=middle
            return high
        if gap<best_distance:
            best_step,best_distance=step,gap
        previous=step
    return best_step if best_distance<initial-.1 else 0.


class ApproachEncounter(DualEncounter):
    """Only advance distance changes; costs, time, legality and reactions do not."""
    approach_actor=None

    def movement_distance_ft(self,actor,action,orientation):
        if action=="advance" and (self.approach_actor is None or actor==self.approach_actor):
            return approach_distance(self.observation_for(actor),orientation,self.speed_ft(actor))
        return super().movement_distance_ft(actor,action,orientation)
