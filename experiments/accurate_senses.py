"""Ideal clear-view geometry contract, separate from legacy 41-number inputs.

Visible oriented boxes are exact idealized measurements, not rendered vision.
No target identity, HP, defenses, or computed attack legality enters the inputs.
"""
import math
from experiments.approach_action import ApproachEncounter
from engine.sensory import floor_relative_position,observe_contacts

SCHEMA="accurate-body-bite-v1"
FIELDS=(
    "own_width","own_depth","own_height","own_floor_height","own_hp_fraction",
    "roll_sin","roll_cos","pitch_sin","pitch_cos","heading_sin","heading_cos",
    "mouth_x","mouth_y","mouth_z","bite_direction_x","bite_direction_y","bite_direction_z",
    "bite_reach","bite_half_angle_cos","speed","reaction_available","movement_blocked",
    "vision_available","visible","web_available","vibration","web_evidence",
    "target_center_forward","target_center_left","target_center_up",
    "target_width","target_depth","target_height",
    *tuple("target_"+axis+"_"+coordinate for axis in ("forward","left","up") for coordinate in ("x","y","z")),
    "vibration_range","vibration_bearing_cos","vibration_bearing_sin",
    *tuple("last_action_"+name for name in ("wait","advance","retreat","face","flee","look_up","look_down","attack")),
    "last_attack_admitted","last_hit","last_damage","recent_injury")


class AccurateEncounter(ApproachEncounter):
    def act(self,actor,action_index,**kwargs):
        result=super().act(actor,action_index,**kwargs)
        if not hasattr(self,"accurate_feedback"):
            self.accurate_feedback={}
        self.accurate_feedback[actor]={"admitted":float(result["admitted"]),
            "blocked":float(result["status"]=="blocked")}
        return result


def observation(world,actor,legacy=None):
    from spider_sim.dual_encounter import ACTORS,ACTIONS
    other=next(cid for cid in ACTORS if cid!=actor)
    legacy=world.observation_for(actor) if legacy is None else legacy
    # Match both fixture visibility and the native engine's visual-contact gate.
    native=observe_contacts(actor,world.geometry,world.positions,world.active,world.config.characters,heights=world.heights)
    visible=bool(legacy["values"][1]) and any(
        c["track_id"]==other and c["modality"]=="vision" for c in native["contacts"])
    pose=world.orientations[actor];body=world.bodies[actor];socket=world.sockets[actor]["bite"]
    profile=world.attack_profile(actor)
    own=floor_relative_position(world.geometry,world.positions[actor],world.heights[actor])
    target=[0.]*15
    if visible:
        other_body=world.bodies[other];other_pose=world.orientations[other]
        anchor=floor_relative_position(world.geometry,world.positions[other],world.heights[other])
        offset=other_pose.rotate((0,0,other_body.height/2))
        relative=pose.inverse((anchor.x+offset[0]-own.x,anchor.y+offset[1]-own.y,anchor.z+offset[2]-own.z))
        axes=[pose.inverse(other_pose.rotate(axis)) for axis in ((1,0,0),(0,1,0),(0,0,1))]
        target=[v/10 for v in relative]+[other_body.width/10,other_body.depth/10,other_body.height/10]+[v for axis in axes for v in axis]
    feedback=getattr(world,"accurate_feedback",{}).get(actor,{})
    angles=[v for a in (pose.roll,pose.pitch,pose.heading) for v in (math.sin(math.radians(a)),math.cos(math.radians(a)))]
    direction_norm=math.sqrt(sum(v*v for v in socket.direction))
    vibration=bool(legacy["values"][3]);contact=legacy["contact"]
    if vibration and (not contact or contact["source"]!="vibration"):
        # Recover only the existing coarse vibration measurement, not visual precision.
        op=floor_relative_position(world.geometry,world.positions[other],world.heights[other])
        dx,dy,dz=op.x-own.x,op.y-own.y,op.z-own.z
        contact={"range_ft":round(math.sqrt(dx*dx+dy*dy+dz*dz)/5)*5,
                 "bearing_deg":round(math.degrees(math.atan2(dy,dx))/45)*45}
    b=math.radians(contact["bearing_deg"]) if vibration else 0
    values=[body.width/10,body.depth/10,body.height/10,world.heights[actor]/10,legacy["self"]["hp_fraction"],
        *angles,*[v/10 for v in socket.origin],*[v/direction_norm for v in socket.direction],
        profile["reach_ft"]/10,math.cos(math.radians(socket.half_angle_deg)),world.speed_ft(actor)/60,
        float(world.reaction_available[actor]),feedback.get("blocked",float(world.escape_blocked[actor])),
        legacy["values"][0],float(visible),legacy["values"][2],float(vibration),float(legacy["web_evidence"]),
        *target,contact["range_ft"]/60 if vibration else 0,math.cos(b) if vibration else 0,math.sin(b) if vibration else 0,
        *[float(world.last_action[actor]==a) for a in ACTIONS],feedback.get("admitted",0),*world.feedback[actor]]
    assert len(values)==len(FIELDS)==57
    return {"schema":SCHEMA,"values":values,"visible":visible,"legacy_values":list(legacy["values"])}
