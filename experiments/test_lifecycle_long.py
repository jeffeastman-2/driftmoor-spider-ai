import random
from experiments.run_lifecycle_long import cache_static_geometry
from spider_sim.ecology import EcologyEncounter
from models.space_time import Position


def test_static_geometry_acceleration_preserves_exact_results():
    world=EcologyEncounter(42)
    geometry=world.geometry
    rng=random.Random(42)
    points=[Position(rng.uniform(0,150),rng.uniform(0,180),rng.choice([0,1,-54])) for _ in range(120)]
    points += [Position(v[0],v[1]) for polygon in geometry.zones.values() for v in polygon.vertices]
    expected_zones=[geometry.zone_at(p) for p in points]
    expected_floors=[geometry.floor_z_at(p) for p in points]
    pairs=list(zip(points,points[1:]))
    expected_visibility=[geometry.can_see(a,b) for a,b in pairs]
    cache_static_geometry(geometry)
    for _ in range(2):
        assert [geometry.zone_at(p) for p in points]==expected_zones
        assert [geometry.floor_z_at(p) for p in points]==expected_floors
        assert [geometry.can_see(a,b) for a,b in pairs]==expected_visibility
