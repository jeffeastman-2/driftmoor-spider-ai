from experiments.accurate_senses import AccurateEncounter,observation,FIELDS
from experiments.audit_jepa_planning import fixture
from spider_sim.dual_encounter import SPIDER,PLAYER


def test_exact_visible_geometry_and_own_bite_are_present_without_hidden_stats():
    world=fixture(SPIDER,"healthy_close_idle",930000,AccurateEncounter)
    obs=observation(world,SPIDER);values=dict(zip(FIELDS,obs["values"]))
    assert obs["visible"] and len(values)==57
    assert values["bite_reach"]==world.attack_profile(SPIDER)["reach_ft"]/10
    assert values["mouth_x"]==world.sockets[SPIDER]["bite"].origin[0]/10
    world.config.characters[PLAYER].character.current_hp-=1
    assert observation(world,SPIDER)["values"]==obs["values"]
    assert not any("legal" in field or "target_hp" in field or "target_id" in field for field in FIELDS)


def test_unseen_body_geometry_is_zeroed_and_feedback_is_observed():
    world=fixture(SPIDER,"no_contact_idle",930001,AccurateEncounter)
    obs=observation(world,SPIDER);values=dict(zip(FIELDS,obs["values"]))
    assert not obs["visible"]
    assert all(value==0 for key,value in values.items() if key.startswith("target_"))
    world=fixture(SPIDER,"healthy_close_idle",930002,AccurateEncounter)
    result=world.act(SPIDER,7,orientation=world.orientations[SPIDER])
    values=dict(zip(FIELDS,observation(world,SPIDER)["values"]))
    assert values["last_attack_admitted"]==float(result["admitted"])


def test_geometry_dataset_has_both_labels_and_keeps_engine_outcome_out_of_features():
    from experiments.learn_bite_geometry import collect
    rows=collect(48,940000)
    assert {r["admitted"] for r in rows}=={0,1}
    assert all(len(r["accurate"])==57 and len(r["coarse"])==41 for r in rows)


def test_visible_box_contract_reconstructs_native_mouth_to_body_distance():
    import math
    import pytest
    from models.body_geometry import Orientation
    from engine.sensory import floor_relative_position
    world=fixture(SPIDER,"healthy_close_idle",930003,AccurateEncounter)
    world.orientations[SPIDER]=Orientation(10,25,30)
    world.orientations[PLAYER]=Orientation(-15,20,80)
    obs=observation(world,SPIDER);v=dict(zip(FIELDS,obs["values"]))
    assert obs["visible"]
    mouth=[v["mouth_"+a]*10 for a in ("x","y","z")]
    center=[v["target_center_"+a]*10 for a in ("forward","left","up")]
    axes=[[v["target_"+axis+"_"+a] for a in ("x","y","z")] for axis in ("forward","left","up")]
    half=[v["target_depth"]*5,v["target_width"]*5,v["target_height"]*5]
    delta=[a-b for a,b in zip(mouth,center)]
    coordinates=[sum(a*b for a,b in zip(delta,axis)) for axis in axes]
    clipped=[max(-h,min(h,c)) for h,c in zip(half,coordinates)]
    nearest=[center[k]+sum(axes[j][k]*clipped[j] for j in range(3)) for k in range(3)]
    reconstructed=math.sqrt(sum((a-b)**2 for a,b in zip(nearest,mouth)))
    source=floor_relative_position(world.geometry,world.positions[SPIDER],world.heights[SPIDER])
    target=floor_relative_position(world.geometry,world.positions[PLAYER],world.heights[PLAYER])
    actual=world.sockets[SPIDER]["bite"].evaluate((source.x,source.y,source.z),world.orientations[SPIDER],
        world.bodies[PLAYER],(target.x,target.y,target.z),world.orientations[PLAYER])
    assert reconstructed==pytest.approx(actual["distance_ft"],abs=1e-10)
