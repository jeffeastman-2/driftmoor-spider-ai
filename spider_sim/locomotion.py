"""Authored local motor behavior: gentle roaming and feedback-driven detours.

Only private clearance rays and actual own movement feedback are used. This
is not a learned planner and has no global route or hidden target access.
"""
import math
import random


def difference(a,b):return abs((a-b+180)%360-180)


class LocalLocomotion:
    def __init__(self,seed=0):
        self.random=random.Random(seed);self.roam_heading=None;self.last_move=None
        self.detour_heading=None;self.detour_steps=0

    def clearance(self,obs,heading):
        offset=round((heading-obs['self']['heading_deg'])/45)%8
        return obs['self']['local_clearance_ft'][offset]

    def blocked(self,obs):
        result=obs.get('last_result') or {}
        return self.last_move is not None and result.get('action')=='move' and result.get('distance_ft',0)<.2

    def open_direction(self,obs,desired,failed=None):
        options=[((obs['self']['heading_deg']+45*i)%360,d) for i,d in enumerate(obs['self']['local_clearance_ft']) if d>=1]
        if failed is not None:options=[pair for pair in options if difference(pair[0],failed)>=30]
        if not options:return None
        return max(options,key=lambda pair:(pair[1]>=5,-difference(pair[0],desired),pair[1]))

    def roam(self,obs):
        if self.roam_heading is None:self.roam_heading=obs['self']['heading_deg']
        heading=(self.roam_heading+self.random.uniform(-4,4))%360
        blocked=self.blocked(obs)
        reason='blocked_roam_turn' if blocked else 'roam_drift'
        clearance=self.clearance(obs,heading)
        if blocked or clearance<5:
            found=self.open_direction(obs,heading,self.last_move['heading_deg'] if blocked else heading)
            if not found:return {'action':'wait'},'no_local_exit'
            heading,clearance=found
            if not blocked:reason='roam_obstacle_turn'
        self.roam_heading=heading
        command={'action':'move','heading_deg':heading,'distance_ft':min(5.,max(.5,clearance-.2)),
                 'keep_facing':True,'face_travel':True,'turn_reason':reason}
        self.last_move=command
        return command,reason

    def pursue(self,obs,command):
        if command['action']!='move':
            self.last_move=None
            return command,None
        desired=command['heading_deg'];blocked=self.blocked(obs)
        near_wall=self.clearance(obs,desired)<.5
        if blocked or near_wall:
            failed=self.last_move['heading_deg'] if blocked else desired
            found=self.open_direction(obs,desired,failed)
            if not found:return {'action':'wait'},'no_local_exit'
            self.detour_heading=found[0];self.detour_steps=3
        if self.detour_steps:
            found=self.open_direction(obs,self.detour_heading)
            if not found:return {'action':'wait'},'no_local_exit'
            heading,clearance=found;self.detour_heading=heading;self.detour_steps-=1
            chosen={'action':'move','heading_deg':heading,'distance_ft':min(5.,max(.5,clearance-.2))}
            self.last_move=chosen
            return chosen,'blocked_recovery'
        self.last_move=command
        return command,None
