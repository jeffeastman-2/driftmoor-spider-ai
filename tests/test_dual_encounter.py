from spider_sim.dual_encounter import DualEncounter,ContactMemory,demonstration,ACTORS,ACTIONS,SPIDER,PLAYER

def test_private_observations_and_anonymous_contacts():
    world=DualEncounter(70000)
    for role in ACTORS:
        obs=world.observation_for(role)
        assert len(obs["values"])==41
        assert "ground_truth" not in obs
        if obs["contact"]:
            assert "id" not in obs["contact"]
            assert "species" not in obs["contact"]
    assert world.observation_for(PLAYER)["values"][2]==0

def test_teacher_low_health_and_memory():
    world=DualEncounter(70000)
    obs=world.observation_for(SPIDER)
    obs["self"]["hp_fraction"]=.1
    memory=ContactMemory()
    assert ACTIONS[memory.choose(SPIDER,obs)]=="wait"  # No observed threat bearing yet.
    obs["contact"]={"range_ft":20,"bearing_deg":45,"elevation_deg":None,"source":"vibration"}
    assert memory.update(obs)[0]==1
    obs["contact"]=None
    assert memory.update(obs)[0]==1
    for _ in range(7):
        memory.update(obs)
    assert memory.contact is None

def test_demonstrations_reproducible_and_next_actor_turn_targets():
    a=demonstration(70001,rounds=3)
    assert a==demonstration(70001,rounds=3)
    for rows in a["actors"].values():
        assert rows[0]["next_observation"]==rows[1]["observation"]
