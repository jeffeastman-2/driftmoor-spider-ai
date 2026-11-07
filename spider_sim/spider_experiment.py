"""Offline Echoing Caverns observation baseline; no providers or session loop."""
from spider_sim.runtime import game_root
game_root()

from engine.activation import check_activations, initial_dormant_set
from engine.md_loader import configure_from_md
from engine.session import _build_perception
from models.space_time import Position

ROOT = game_root()
SPIDER = "giant_spider"
PLAYER = "gilthaness"

# Scripted placements, not movement commands. Door traversal is assumed open.
APPROACH = (
    ("echo_chamber", 150.0, 195.0),
    ("before_secret_opening", 131.0, 195.0),
    ("rocky_passage_entry", 129.0, 195.0),
    ("rocky_passage_middle", 110.0, 195.0),
    ("outside_spider_chamber", 82.0, 195.0),
    ("inside_spider_chamber", 80.0, 195.0),
    ("near_vault", 70.0, 195.0),
)


class SpiderObservationBaseline:
    """Keeps privileged state separate from observer-filtered perception."""

    def __init__(self):
        self.config = configure_from_md(ROOT / "scenes/echoing_caverns.md")
        self.geometry = self.config.scene_geometry
        if self.geometry is None:
            raise RuntimeError("Echoing Caverns geometry did not load")
        self.positions = {
            cid: (p.x, p.y, self.geometry.zone_at(p) or "")
            for cid, p in self.geometry.starting_positions.items()
        }
        self.active = set(self.config.characters) - initial_dormant_set(
            self.config.characters
        )

    def sample(self, label, x, y, elapsed_seconds):
        """Place one player and evaluate real engine activation/perception."""
        point = Position(x, y)
        zone = self.geometry.zone_at(point)
        if not zone:
            raise ValueError(f"Scripted point outside scene zones: {x}, {y}")
        self.positions[PLAYER] = (x, y, zone)
        activated = check_activations(
            characters=self.config.characters,
            active_characters=self.active,
            current_positions=self.positions,
        )
        # Calling perception for a dormant observer is experimental; the normal
        # session does not give that creature a turn. No activation is forced.
        sensed = _build_perception(
            SPIDER, self.geometry, self.positions, self.active,
            self.config.characters,
        )
        player_view = _build_perception(
            PLAYER, self.geometry, self.positions, self.active,
            self.config.characters,
        )
        return {
            "elapsed_seconds": elapsed_seconds,
            "label": label,
            "observer_input": {"perception": sensed},
            "baseline": {
                "spider_active": SPIDER in self.active,
                "spider_turn_perception": sensed if SPIDER in self.active else None,
                "activation_events": [list(event) for event in activated],
                "player_perception": player_view,
            },
            "ground_truth": {
                "player_id": PLAYER,
                "player_position_xy": [x, y],
                "player_zone": zone,
                "spider_position_xyz": [
                    self.geometry.starting_positions[SPIDER].x,
                    self.geometry.starting_positions[SPIDER].y,
                    self.geometry.starting_positions[SPIDER].z,
                ],
            },
        }


def run_approach():
    experiment = SpiderObservationBaseline()
    return [
        experiment.sample(label, x, y, step * 10)
        for step, (label, x, y) in enumerate(APPROACH)
    ]


class ResolvedCavernExperiment(SpiderObservationBaseline):
    """Bounded move/wait adapter using session geometry primitives.

    One step = one six-second combat movement budget. This is not the full
    narrated session. Doors are assumed open. Ceiling z is retained as authored
    height above the local floor; sensory and attack ranges use 3D distance.
    """

    def __init__(self, seed=0, *, spider_height_ft=None):
        import random
        super().__init__()
        import math
        if spider_height_ft is not None and (
            not math.isfinite(spider_height_ft) or spider_height_ft < 0
        ):
            raise ValueError("Spider height must be finite and nonnegative")
        # Authored z is a height ABOVE the local floor, not scene elevation.
        self.heights = {
            cid: p.z for cid, p in self.geometry.starting_positions.items()
        }
        if spider_height_ft is not None:
            self.heights[SPIDER] = spider_height_ft
        from models.body_geometry import Body, Orientation, body_sockets, body_from_identity
        # Explicit fixture assumptions: these are NOT facts read from identity.
        self.bodies = {
            SPIDER: Body(4, 4, 2, "experimental spider proxy; dimensions unauthored"),
            PLAYER: Body(2, 1, 5.5, "experimental humanoid fallback; Gilthaness height unauthored"),
        }
        self.bodies = {
            cid: body_from_identity(self.config.characters[cid].identity, body)
            for cid, body in self.bodies.items()
        }
        self.orientations = {
            SPIDER: Orientation(roll=180 if self.heights[SPIDER] > 0 else 0),
            PLAYER: Orientation(heading=180),
        }
        self.sockets = {cid: body_sockets(body) for cid, body in self.bodies.items()}
        self.random = random.Random(seed)
        self.elapsed_seconds = 0
        # One explicit initial placement; every subsequent change is resolved.
        self.positions[PLAYER] = (150.0, 195.0, "Echo Chamber")

    def observe(self):
        from engine.sensory import observe_contacts
        observation = observe_contacts(
            SPIDER, self.geometry, self.positions, self.active, self.config.characters, heights=self.heights,
        )
        pose = self.orientations[SPIDER]
        observation["self"]["orientation_roll_pitch_heading_deg"] = [pose.roll,pose.pitch,pose.heading]
        observation["self"]["anchor_xy"] = list(self.positions[SPIDER][:2])
        body = self.bodies[SPIDER]
        observation["self"]["body_dimensions_ft"] = [body.width,body.depth,body.height]
        from engine.loadout import wielded_weapons
        weapons = wielded_weapons(self.config.characters[SPIDER])
        observation["self"]["bite_reach_ft"] = max(
            (w.reach+0.5 for w in weapons if not w.is_ranged),default=0,
        )
        return observation

    def step(self, actor_id, action_type, destination=None, *, target_id=None, orientation=None):
        import math
        from engine.geometry import (
            _clamp_position, _check_boundary_crossing, _check_impassable,
        )
        from engine.session import _movement_budget_feet, _clamp_to_authored_home

        if actor_id not in {PLAYER, SPIDER}:
            raise ValueError("Experiment only controls the player and spider")
        if action_type not in {"move", "wait", "attack", "orient"}:
            raise ValueError("Only move, wait, orient and stationary attack are implemented")
        if action_type == "move":
            if destination is None or len(destination) != 2 or not all(
                math.isfinite(value) for value in destination
            ):
                raise ValueError("move requires two finite destination coordinates")
        elif destination is not None:
            raise ValueError("wait/attack takes no destination")
        if action_type == "attack" and target_id not in {PLAYER, SPIDER} - {actor_id}:
            raise ValueError("attack requires the opposing experiment actor")
        if action_type != "attack" and target_id is not None:
            raise ValueError("target_id is only valid for attack")
        if self.config.characters[actor_id].character.current_hp <= 0:
            raise ValueError("A defeated actor cannot act")
        from models.body_geometry import Orientation
        if action_type == "orient":
            if not isinstance(orientation, Orientation):
                raise ValueError("orient requires an Orientation")
        elif orientation is not None:
            raise ValueError("orientation is only valid for orient")
        before = self.observe()
        origin = self.positions[actor_id]
        char = self.config.characters[actor_id].character
        budget = _movement_budget_feet(char.speed, in_combat=True)
        log = []
        status = "waited"
        combat = None
        if action_type == "move":
            proposed = _clamp_position(
                (destination[0], destination[1], ""), origin, budget,
            )
            proposed = _clamp_to_authored_home(
                actor_id, proposed, char, self.geometry,
            )
            allowed, damage, notes, stop = _check_boundary_crossing(
                actor_id, origin, proposed, char,
                self.geometry.boundaries, self.config.game_time,
            )
            log.extend(notes)
            char.current_hp = max(0, char.current_hp - damage)
            candidate = proposed if allowed else (
                (stop[0], stop[1], "") if stop else origin
            )
            clear, notes, imp_stop = _check_impassable(
                actor_id, origin, candidate,
                self.geometry.impassable, self.geometry.zones,
            )
            log.extend(notes)
            if not clear:
                candidate = (imp_stop[0], imp_stop[1], "") if imp_stop else origin
            zone = self.geometry.zone_at(Position(candidate[0], candidate[1]))
            if zone is None:
                candidate = origin
                zone = origin[2]
            self.positions[actor_id] = (candidate[0], candidate[1], zone)
            status = "moved" if self.positions[actor_id][:2] != origin[:2] else "blocked"
        if action_type == "orient":
            self.orientations[actor_id] = orientation
            status = "oriented"
        if action_type == "attack":
            status, combat, notes = self._stationary_attack(actor_id, target_id)
            log.extend(notes)
        self.elapsed_seconds += 6
        activated = check_activations(
            characters=self.config.characters, active_characters=self.active,
            current_positions=self.positions,
        )
        landed = self.positions[actor_id]
        from engine.sensory import floor_relative_position
        origin_3d = floor_relative_position(self.geometry, origin, self.heights[actor_id])
        landed_3d = floor_relative_position(self.geometry, landed, self.heights[actor_id])
        return {
            "elapsed_seconds": self.elapsed_seconds,
            "observer_before": before,
            "action": {
                "actor_id": actor_id, "type": action_type,
                "destination_xy": list(destination) if destination is not None else None,
                "duration_seconds": 6,
                "target_id": target_id,
                "orientation_roll_pitch_heading_deg": (
                    [orientation.roll, orientation.pitch, orientation.heading]
                    if orientation is not None else None
                ),
            },
            "observer_after": self.observe(),
            "baseline": {
                "spider_active": SPIDER in self.active,
                "activation_events": [list(event) for event in activated],
            },
            "outcome": {
                "status": status,
                "distance_moved_ft": math.hypot(landed[0]-origin[0], landed[1]-origin[1]),
                "movement_budget_ft": budget,
                "resolver_log": log,
                "combat": combat,
            },
            "ground_truth": {
                "actor_before_xyz": [origin_3d.x, origin_3d.y, origin_3d.z],
                "actor_after_xyz": [landed_3d.x, landed_3d.y, landed_3d.z],
                "heights_above_floor_ft": dict(self.heights),
                "body_profiles": {
                    cid: {
                        "width_ft": body.width, "depth_ft": body.depth, "height_ft": body.height,
                        "source": body.source,
                        "orientation_roll_pitch_heading_deg": [
                            self.orientations[cid].roll, self.orientations[cid].pitch,
                            self.orientations[cid].heading,
                        ],
                    } for cid, body in self.bodies.items()
                },
                "actor_before_xy": list(origin[:2]),
                "actor_after_xy": list(landed[:2]),
                "actor_zone": landed[2],
                "hp": {
                    cid: self.config.characters[cid].character.current_hp
                    for cid in (SPIDER, PLAYER)
                },
            },
        }


    def _stationary_attack(self, actor_id, target_id):
        """Resolve an in-place weapon attack, with explicit experiment guards.

        CombatResolver normally auto-approaches targets. This action deliberately
        does not: planning movement and attacking are separate action slots.
        It uses the resolver's weapon selection/range and actual dice/damage.
        """
        import copy
        import math
        from engine.combat_resolver import CombatResolver, _best_weapon
        from engine.game_rng import rng, use_rng
        from engine.loadout import wielded_weapons
        from engine.sensory import observe_contacts

        target = self.config.characters[target_id].character
        feedback = {"attempted": False, "hit": None, "damage": 0}
        if target.current_hp <= 0:
            return "target_defeated", feedback, []
        visible = observe_contacts(
            actor_id, self.geometry, self.positions, self.active, self.config.characters, heights=self.heights,
        )["contacts"]
        if not any(c["track_id"] == target_id and c["modality"] == "vision" for c in visible):
            return "target_not_visible", feedback, []
        origin, target_pos = self.positions[actor_id], self.positions[target_id]
        from engine.sensory import floor_relative_position
        source_point = floor_relative_position(self.geometry, origin, self.heights[actor_id])
        target_point = floor_relative_position(self.geometry, target_pos, self.heights[target_id])
        shape = self.sockets[actor_id]["bite"].evaluate(
            (source_point.x,source_point.y,source_point.z), self.orientations[actor_id],
            self.bodies[target_id], (target_point.x,target_point.y,target_point.z),
            self.orientations[target_id],
        )
        distance = shape["distance_ft"]
        feedback["geometry"] = shape
        session = self.config.characters[actor_id]
        weapon = _best_weapon(wielded_weapons(session), distance)
        if weapon is None:
            return "out_of_range_or_unarmed", feedback, []
        if not shape["facing"]:
            return "outside_attack_arc", feedback, []
        stationary = copy.copy(session)
        stationary.character = copy.copy(session.character)
        stationary.character.speed = "0 ft"
        previous_rng = rng()
        use_rng(self.random)
        try:
            adj, log = CombatResolver().resolve(
                character_id=actor_id, turn=self.elapsed_seconds // 6 + 1,
                attacker_session=stationary, target_id=target_id,
                target_session=self.config.characters[target_id],
                attacker_pos=origin, target_pos=target_pos,
                scene_geometry=self.geometry, game_time=self.config.game_time,
                preferred_weapon=weapon,
                attacker_height_ft=self.heights[actor_id],
                target_height_ft=self.heights[target_id],
                stationary_contact_distance_ft=distance,
            )
        finally:
            use_rng(previous_rng)
        for cid, damage in adj.hp_damage.items():
            char = self.config.characters[cid].character
            char.current_hp = max(0, char.current_hp - damage)
            if char.current_hp == 0:
                self.active.discard(cid)
        feedback = {
            "attempted": bool(adj.attacks),
            "hit": bool(adj.hp_damage) if adj.attacks else None,
            "damage": adj.hp_damage.get(target_id, 0),
            "weapon": weapon.id,
            "geometry": shape,
        }
        return ("hit" if feedback["hit"] else "miss"), feedback, log


def run_resolved_approach():
    experiment = ResolvedCavernExperiment()
    # Scripted commands; all changes after initialization go through step().
    commands = [
        (PLAYER, "move", (130.0, 195.0)),
        (SPIDER, "wait", None),
        (PLAYER, "move", (110.0, 195.0)),
        (PLAYER, "move", (82.0, 195.0)),
        (SPIDER, "wait", None),
        (PLAYER, "move", (70.0, 195.0)),
        (SPIDER, "move", (74.0, 195.0)),
        (PLAYER, "move", (110.0, 195.0)),
    ]
    return [experiment.step(actor, action, destination) for actor, action, destination in commands]


def run_encounters(seed=0):
    """Matched scripted encounters; no learned decision-maker is claimed."""
    records = []
    for name in ("retreat_before_entry", "enter_and_wait", "enter_and_attack", "offset_and_attack", "grounded_attack", "turn_away", "aim_down"):
        experiment = ResolvedCavernExperiment(
            seed=seed, spider_height_ft=0.0 if name in {"grounded_attack", "turn_away"} else None,
        )
        commands = [
            (PLAYER, "move", (120.0, 195.0), None),
            (SPIDER, "attack", None, PLAYER),  # visible, but far beyond bite reach
            (PLAYER, "move", (90.0, 195.0), None),
        ]
        if name == "retreat_before_entry":
            commands += [(PLAYER, "move", (120.0, 195.0), None),
                         (SPIDER, "wait", None, None)]
        else:
            commands += [(PLAYER, "move", (69.0, 195.0), None)]
            if name == "offset_and_attack":
                commands += [(PLAYER, "move", (69.0, 205.0), None)]
            if name == "turn_away":
                commands += [(PLAYER, "move", (67.0,195.0),None),
                             (SPIDER, "orient", (0,0,180),None)]
            if name == "aim_down":
                commands += [(PLAYER, "move", (64.0,195.0),None),
                             (SPIDER, "orient", (180,90,0),None)]
            if name == "enter_and_wait":
                commands += [(SPIDER, "wait", None, None)]
            else:
                commands += [(SPIDER, "attack", None, PLAYER)]
            commands += [(PLAYER, "move", (100.0, 195.0), None),
                         (SPIDER, "attack", None, PLAYER)]
        for step, (actor, action, destination, target) in enumerate(commands):
            if action == "orient":
                from models.body_geometry import Orientation
                record = experiment.step(actor, action, orientation=Orientation(*destination))
            else:
                record = experiment.step(actor, action, destination, target_id=target)
            record["episode"] = {"name": name, "seed": seed, "step": step}
            records.append(record)
    return records
