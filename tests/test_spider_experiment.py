"""Checks against the actual cavern loader and activation/perception code."""
from spider_sim.spider_experiment import SpiderObservationBaseline, run_approach


def test_loads_place_resident_and_preserves_ceiling_position():
    experiment = SpiderObservationBaseline()
    spider = experiment.config.characters["giant_spider"]
    assert spider.activation == "zone_entry:Secret Chamber"
    assert experiment.geometry.starting_positions["giant_spider"].z == 10
    assert "giant_spider" not in experiment.active


def test_actual_activation_waits_for_chamber_entry():
    rows = run_approach()
    outside = rows[4]
    inside = rows[5]
    assert not outside["baseline"]["spider_active"]
    assert outside["baseline"]["spider_turn_perception"] is None
    assert inside["baseline"]["spider_active"]
    assert inside["baseline"]["activation_events"] == [
        ["giant_spider", "zone_entry:Secret Chamber"]
    ]
    assert rows[6]["baseline"]["activation_events"] == []


def test_hidden_observer_can_be_sampled_without_revealing_it():
    experiment = SpiderObservationBaseline()
    row = experiment.sample("outside", 82, 195, 0)
    assert "giant_spider" not in experiment.active
    assert "the giant spider" not in row["baseline"]["player_perception"].lower()
    assert row["observer_input"]["perception"]
    assert "player_position_xy" not in row["observer_input"]


def test_repeated_runs_have_identical_records():
    assert run_approach() == run_approach()

from copy import deepcopy
import pytest

from spider_sim.spider_experiment import ResolvedCavernExperiment, run_resolved_approach
from engine.sensory import observe_contacts
from engine.session import _build_perception


def test_contacts_do_not_expose_unobserved_player():
    experiment = ResolvedCavernExperiment()
    assert experiment.observe()["contacts"] == []
    record = experiment.step("gilthaness", "move", (110, 195))
    contact = record["observer_after"]["contacts"][0]
    assert set(contact) == {"track_id", "modality", "range_ft", "bearing_deg", "elevation_angle_deg"}
    assert contact["track_id"] == "gilthaness"


def test_structured_contacts_match_existing_perception():
    experiment = ResolvedCavernExperiment()
    for x in (150, 130, 110, 82, 80, 70):
        experiment.positions["gilthaness"] = (x, 195, "Rocky Passage")
        structured = experiment.observe()["contacts"]
        prose = _build_perception(
            "giant_spider", experiment.geometry, experiment.positions,
            experiment.active, experiment.config.characters,
        )
        handle = experiment.config.characters["gilthaness"].scene_handle
        assert bool(structured) == bool(prose)
        if structured:
            assert handle in prose


def test_real_clamp_limits_attempted_long_move():
    experiment = ResolvedCavernExperiment()
    row = experiment.step("gilthaness", "move", (70, 195))
    assert row["outcome"]["distance_moved_ft"] == 30
    assert row["ground_truth"]["actor_after_xy"] == [120, 195]
    assert not row["baseline"]["spider_active"]


def test_wait_preserves_state_and_old_observation_snapshot():
    experiment = ResolvedCavernExperiment()
    row = experiment.step("giant_spider", "wait")
    snapshot = deepcopy(row)
    assert row["outcome"]["distance_moved_ft"] == 0
    assert row["observer_before"] == row["observer_after"]
    experiment.step("gilthaness", "move", (110, 195))
    assert row == snapshot


def test_rock_blocks_move_off_passage():
    experiment = ResolvedCavernExperiment()
    experiment.step("gilthaness", "move", (120, 195))
    experiment.step("gilthaness", "move", (110, 195))
    row = experiment.step("gilthaness", "move", (110, 150))
    assert row["outcome"]["resolver_log"]
    assert row["ground_truth"]["actor_after_xy"][1] >= 190
    assert row["outcome"]["distance_moved_ft"] <= 30.08


def test_resolved_trace_is_deterministic_and_activates_only_on_entry():
    first = run_resolved_approach()
    assert first == run_resolved_approach()
    assert first[2]["observer_after"]["contacts"]
    assert not first[2]["baseline"]["spider_active"]
    assert first[5]["baseline"]["activation_events"] == [
        ["giant_spider", "zone_entry:Secret Chamber"]
    ]


def test_hidden_targets_are_excluded():
    experiment = ResolvedCavernExperiment()
    experiment.step("gilthaness", "move", (110, 195))
    experiment.active.remove("gilthaness")
    assert experiment.observe()["contacts"] == []


@pytest.mark.parametrize("action,destination", [
    ("attack", None), ("move", None), ("move", (float("nan"), 195)),
    ("wait", (100, 195)),
])
def test_invalid_actions_do_not_advance_state(action, destination):
    experiment = ResolvedCavernExperiment()
    original = dict(experiment.positions)
    with pytest.raises(ValueError):
        experiment.step("gilthaness", action, destination)
    assert experiment.elapsed_seconds == 0
    assert experiment.positions == original


from spider_sim.spider_experiment import run_encounters
from engine.game_rng import rng


def in_reach(seed=0):
    experiment = ResolvedCavernExperiment(seed=seed, spider_height_ft=0.0)
    for destination in ((120, 195), (90, 195), (69, 195)):
        experiment.step("gilthaness", "move", destination)
    return experiment


def test_attack_applies_actual_damage_without_auto_movement():
    experiment = in_reach()
    target = experiment.config.characters["gilthaness"].character
    hp_before = target.current_hp
    row = experiment.step("giant_spider", "attack", target_id="gilthaness")
    assert row["outcome"]["status"] == "hit"
    assert row["outcome"]["combat"]["damage"] > 0
    assert target.current_hp == max(0, hp_before-row["outcome"]["combat"]["damage"])
    assert row["outcome"]["distance_moved_ft"] == 0
    assert row["outcome"]["combat"]["weapon"] == "spider_bite"


def test_distant_attack_neither_moves_nor_damages():
    experiment = ResolvedCavernExperiment()
    experiment.step("gilthaness", "move", (120, 195))
    target = experiment.config.characters["gilthaness"].character
    hp_before = target.current_hp
    row = experiment.step("giant_spider", "attack", target_id="gilthaness")
    assert row["outcome"]["status"] == "out_of_range_or_unarmed"
    assert row["outcome"]["combat"]["attempted"] is False
    assert row["outcome"]["distance_moved_ft"] == 0
    assert target.current_hp == hp_before


@pytest.mark.parametrize("distance,attempted", [(7.5, True), (8, True), (8.1, False)])
def test_attack_uses_engine_reach_tolerance(distance, attempted):
    experiment = in_reach()
    experiment.step("gilthaness", "move", (64+distance, 195))
    row = experiment.step("giant_spider", "attack", target_id="gilthaness")
    assert row["outcome"]["combat"]["attempted"] is attempted


def test_nonvisual_only_contact_does_not_allow_aimed_attack():
    experiment = in_reach()
    experiment.geometry.visibility_rules[("SECRET CHAMBER", "SECRET CHAMBER")] = 0
    assert experiment.observe()["contacts"][0]["modality"] == "nonvisual"
    row = experiment.step("giant_spider", "attack", target_id="gilthaness")
    assert row["outcome"]["status"] == "target_not_visible"
    assert not row["outcome"]["combat"]["attempted"]


def test_defeated_targets_cannot_be_attacked_or_act():
    experiment = in_reach()
    experiment.config.characters["gilthaness"].character.current_hp = 0
    row = experiment.step("giant_spider", "attack", target_id="gilthaness")
    assert row["outcome"]["status"] == "target_defeated"
    time_before = experiment.elapsed_seconds
    with pytest.raises(ValueError, match="defeated"):
        experiment.step("gilthaness", "wait")
    assert experiment.elapsed_seconds == time_before


def test_seeded_combat_preserves_global_rng_and_record_history():
    previous = rng()
    state = previous.getstate()
    experiment = in_reach()
    row = experiment.step("giant_spider", "attack", target_id="gilthaness")
    frozen = deepcopy(row)
    experiment.step("giant_spider", "attack", target_id="gilthaness")
    assert row == frozen
    assert rng() is previous
    assert previous.getstate() == state
    assert run_encounters(0) == run_encounters(0)


def test_encounters_include_hits_misses_and_safe_retreats():
    records = [r for seed in range(5) for r in run_encounters(seed)]
    statuses = {r["outcome"]["status"] for r in records}
    assert {"hit", "miss", "out_of_range_or_unarmed"} <= statuses
    retreat = [r for r in records if r["episode"]["name"] == "retreat_before_entry"]
    assert all(not r["baseline"]["spider_active"] for r in retreat)


from engine.sensory import floor_relative_position, classify_contact
from models.space_time import Position


def test_authored_height_is_above_local_floor():
    experiment = ResolvedCavernExperiment()
    spider = floor_relative_position(
        experiment.geometry, experiment.positions["giant_spider"],
        experiment.heights["giant_spider"],
    )
    floor = experiment.geometry.floor_z_at(Position(64, 195))
    assert floor == -5
    assert spider.z == 5
    ground = floor_relative_position(experiment.geometry, (64, 195), 0)
    assert spider.distance_to(ground) == 10


def test_ceiling_bite_behind_fangs_requires_reorientation():
    experiment = ResolvedCavernExperiment()
    for destination in ((120, 195), (90, 195), (64, 195)):
        experiment.step("gilthaness", "move", destination)
    assert experiment.heights["giant_spider"] == 10
    contact = experiment.observe()["contacts"][0]
    assert contact["range_ft"] == 10
    assert contact["elevation_angle_deg"] == -90
    row = experiment.step("giant_spider", "attack", target_id="gilthaness")
    assert not row["outcome"]["combat"]["attempted"]
    assert row["outcome"]["status"] == "outside_attack_arc"


def test_grounded_creatures_follow_different_floor_elevations():
    experiment = ResolvedCavernExperiment()
    upper = floor_relative_position(experiment.geometry, (150, 195), 0)
    lower = floor_relative_position(experiment.geometry, (64, 195), 0)
    assert upper.z == -2
    assert lower.z == -5
    assert upper.distance_to(lower) == pytest.approx((86**2 + 3**2)**0.5)


def test_3d_diagonal_distance_and_range_filter():
    experiment = ResolvedCavernExperiment()
    a = Position(64, 195, -5)
    b = Position(67, 199, 7)
    assert a.distance_to(b) == 13
    sees, sensed = classify_contact(a, b, experiment.geometry, 12, 0)
    assert not sees and not sensed
    assert classify_contact(a, b, experiment.geometry, 13, 0)[0]


def test_height_survives_wait_and_horizontal_spider_move():
    experiment = ResolvedCavernExperiment()
    experiment.step("giant_spider", "wait")
    row = experiment.step("giant_spider", "move", (74, 195))
    assert row["ground_truth"]["heights_above_floor_ft"]["giant_spider"] == 10
    assert row["ground_truth"]["actor_after_xyz"] == [74, 195, 5]


def test_ceiling_and_grounded_controls_have_different_attack_outcomes():
    records = run_encounters(0)
    ceiling = [r for r in records if r["episode"]["name"] == "enter_and_attack"
               and r["action"]["type"] == "attack"]
    ground = [r for r in records if r["episode"]["name"] == "grounded_attack"
              and r["action"]["type"] == "attack"]
    ceiling_attack = next(r for r in ceiling if r["outcome"]["combat"]["attempted"])
    ground_attack = next(r for r in ground if r["outcome"]["combat"]["attempted"])
    assert ceiling_attack["outcome"]["combat"]["geometry"]["distance_ft"] > ground_attack["outcome"]["combat"]["geometry"]["distance_ft"]
    assert any(r["outcome"]["combat"]["attempted"] for r in ground)


def test_combat_resolver_checks_vertical_reach_even_with_preferred_weapon():
    import copy
    from engine.combat_resolver import CombatResolver
    experiment = in_reach()
    session = copy.copy(experiment.config.characters["giant_spider"])
    session.character = copy.copy(session.character)
    session.character.speed = "0 ft"
    adj, _ = CombatResolver().resolve(
        character_id="giant_spider", turn=1, attacker_session=session,
        target_id="gilthaness", target_session=experiment.config.characters["gilthaness"],
        attacker_pos=experiment.positions["giant_spider"],
        target_pos=experiment.positions["gilthaness"],
        scene_geometry=experiment.geometry, preferred_weapon=session.weapons[0],
        attacker_height_ft=10, target_height_ft=0,
    )
    assert not adj.attacks
    assert not adj.hp_damage


from models.body_geometry import Body, Orientation


def test_reorienting_changes_bite_without_moving_body_anchor():
    experiment = in_reach()
    experiment.step("gilthaness","move",(67,195))
    before = dict(experiment.positions)
    experiment.step("giant_spider", "orient", orientation=Orientation(heading=180))
    row = experiment.step("giant_spider", "attack", target_id="gilthaness")
    assert row["outcome"]["status"] == "outside_attack_arc"
    assert not row["outcome"]["combat"]["attempted"]
    assert experiment.positions == before
    experiment.step("giant_spider", "orient", orientation=Orientation())
    row = experiment.step("giant_spider", "attack", target_id="gilthaness")
    assert row["outcome"]["combat"]["attempted"]


def test_downward_aim_can_bite_tall_target_below_ceiling_anchor():
    experiment = ResolvedCavernExperiment()
    experiment.bodies["gilthaness"] = Body(2,1,6,"explicit six-foot test fixture")
    for destination in ((120,195),(90,195),(64,195)):
        experiment.step("gilthaness","move",destination)
    experiment.step("giant_spider","orient",orientation=Orientation(roll=180,pitch=90))
    row = experiment.step("giant_spider","attack",target_id="gilthaness")
    assert row["outcome"]["combat"]["attempted"]
    assert row["outcome"]["combat"]["geometry"]["distance_ft"] <= 5


def test_authored_body_provenance_is_recorded():
    row = in_reach().step("giant_spider","wait")
    profile = row["ground_truth"]["body_profiles"]["gilthaness"]
    assert "IDENTITY.md" in profile["source"]
    assert "provisional" in profile["source"]
    assert profile["height_ft"] == 5.5


def test_orient_requires_pose_and_has_no_position_side_effect():
    experiment = ResolvedCavernExperiment()
    positions = dict(experiment.positions)
    with pytest.raises(ValueError):
        experiment.step("giant_spider","orient")
    assert experiment.positions == positions
    assert experiment.elapsed_seconds == 0
