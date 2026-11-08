"""Disengage, Dash and automatic opportunity reactions for the cavern fixture."""
import math

from spider_sim.runtime import game_root
game_root()
from engine.geometry import _clamp_position, _check_impassable
from engine.session import _clamp_to_authored_home, _movement_budget_feet
from engine.sensory import floor_relative_position
from engine.loadout import wielded_weapons
from models.space_time import Position


class WithdrawalRules:
    def speed_ft(self, actor):
        return _movement_budget_feet(self.config.characters[actor].character.speed,in_combat=True)

    def _preview_endpoint(self, actor, destination, budget):
        origin = self.positions[actor]
        char = self.config.characters[actor].character
        point = _clamp_position((*destination,""),origin,budget)
        point = _clamp_to_authored_home(actor,point,char,self.geometry)
        clear,_,stop = _check_impassable(actor,origin,point,self.geometry.impassable,self.geometry.zones)
        if not clear:
            point = (*stop,"") if stop else origin
        zone = self.geometry.zone_at(Position(*point[:2]))
        return (*point[:2],zone) if zone else origin

    def _reach_distance(self, reactor, mover, point):
        source = floor_relative_position(self.geometry,self.positions[reactor],self.heights[reactor])
        target = floor_relative_position(self.geometry,point,self.heights[mover])
        return self.sockets[reactor]["bite"].evaluate(
            (source.x,source.y,source.z),self.orientations[reactor],
            self.bodies[mover],(target.x,target.y,target.z),self.orientations[mover])["distance_ft"]

    def _first_reach_exit(self, reactor, mover, endpoint):
        """Find the first inside->outside crossing, including pass-through moves.

        Quarter-foot sampling brackets a crossing; bisection places the reaction
        just inside reach. This uses the experiment's existing socket/box reach
        and +0.5-foot combat tolerance rather than grid-square reach.
        """
        reach = max((w.reach+0.5 for w in wielded_weapons(self.config.characters[reactor])
                     if not w.is_ranged),default=0)
        if not reach:
            return None
        origin = self.positions[mover]
        length = math.dist(origin[:2],endpoint[:2])
        if length<1e-9:
            return None
        def point(t):
            x = origin[0]+(endpoint[0]-origin[0])*t
            y = origin[1]+(endpoint[1]-origin[1])*t
            return (x,y,self.geometry.zone_at(Position(x,y)) or origin[2])
        count = max(1,math.ceil(length/0.25))
        previous_t = 0.
        previous_inside = self._reach_distance(reactor,mover,origin)<=reach
        for index in range(1,count+1):
            t = index/count
            inside = self._reach_distance(reactor,mover,point(t))<=reach
            if previous_inside and not inside:
                lo,hi = previous_t,t
                for _ in range(32):
                    mid = (lo+hi)/2
                    if self._reach_distance(reactor,mover,point(mid))<=reach-1e-7:
                        lo = mid
                    else:
                        hi = mid
                return point(lo)
            previous_t,previous_inside = t,inside
        return None

    def _move_with_reactions(self, actor, destination, budget, disengage, step):
        other = "gilthaness" if actor=="giant_spider" else "giant_spider"
        endpoint = self._preview_endpoint(actor,destination,budget)
        crossing = None
        if (not disengage and self.reaction_available[other]
            and self.config.characters[other].character.current_hp>0):
            crossing = self._first_reach_exit(other,actor,endpoint)
        char = self.config.characters[actor].character
        original_speed = char.speed
        # The shared adapter enforces its supplied speed cap. Dash grants an
        # additional Speed of movement without permanently changing the stat.
        char.speed = f"{budget} ft"
        events = []
        elapsed = self.elapsed_seconds
        try:
            if crossing is None:
                result = step(actor,"move",endpoint[:2])
            else:
                result = step(actor,"move",crossing[:2])
                self.elapsed_seconds = elapsed
                arrived = math.dist(self.positions[actor][:2],crossing[:2])<1e-5
                if arrived and char.current_hp>0:
                    visible = self.observation_for(other)["contact"]
                    if visible and visible["source"]=="vision":
                        hp_before = char.current_hp
                        status,combat,_ = self._stationary_attack(other,actor)
                        if combat.get("attempted"):
                            self.reaction_available[other] = False
                            event = {"reactor":other,"mover":actor,"status":status,
                                     "hit":bool(combat.get("hit")),
                                     "damage":hp_before-char.current_hp,
                                     "rolled_damage":combat.get("damage",0),
                                     "position_xy":list(self.positions[actor][:2]),
                                     "rear_exposed":abs(((math.degrees(math.atan2(
                                         self.positions[other][1]-self.positions[actor][1],
                                         self.positions[other][0]-self.positions[actor][0]))
                                         -self.orientations[actor].heading+180)%360)-180)>=120}
                            events.append(event)
                            self.feedback[other][:2] = [float(event["hit"]),event["rolled_damage"]/20]
                    if char.current_hp>0:
                        prefix_distance = result["outcome"]["distance_moved_ft"]
                        prefix_log = result["outcome"]["resolver_log"]
                        result = step(actor,"move",endpoint[:2])
                        result["outcome"]["distance_moved_ft"] += prefix_distance
                        result["outcome"]["resolver_log"] = prefix_log+result["outcome"]["resolver_log"]
                if char.current_hp<=0:
                    result["outcome"]["status"] = "defeated_by_opportunity_attack"
            result["outcome"]["movement_budget_ft"] = budget
            result["outcome"]["opportunity_attacks"] = events
            return result
        finally:
            char.speed = original_speed
