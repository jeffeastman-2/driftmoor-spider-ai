import torch
from experiments.audit_jepa_planning import fixture,reference_future,analyze
from experiments.recurrent_agents import Agent
from spider_sim.dual_encounter import PLAYER,SPIDER,ACTIONS


def test_reference_branches_are_repeatable_and_do_not_mutate_source():
    world=fixture(SPIDER,"healthy_close_idle",230000)
    before=world.observation_for(SPIDER)
    positions=dict(world.positions)
    rng=world.random.getstate()
    first=reference_future(world,SPIDER,ACTIONS.index("attack"),[0]*5,240000,"idle")
    assert first==reference_future(world,SPIDER,ACTIONS.index("attack"),[0]*5,240000,"idle")
    assert world.observation_for(SPIDER)==before
    assert world.positions==positions and world.random.getstate()==rng
    # An idle opponent cannot injure a stationary attacker.
    assert first[0][16]==before["values"][16]
    assert first[0][41]==float(first[1]["admitted"])
    assert first[0][42]==first[1]["damage"]/20


def test_no_contact_fixture_has_no_contact_for_either_actor():
    for role in (SPIDER,PLAYER):
        world=fixture(role,"no_contact_idle",230005)
        assert world.observation_for(role)["contact"] is None


def test_audit_reports_forecast_and_cost_comparisons_separately():
    torch.manual_seed(2)
    world=fixture(SPIDER,"healthy_close_idle",230000)
    world.audit_case="healthy_close_idle"
    result=analyze(world,SPIDER,Agent().eval(),2)
    assert len(result["actions"])==8
    assert result["forecast_ranking_regret"]>=0
    assert set(result["offline_forecast_replacements"])=={
        "next_hp","attack_feedback","contact_range","blocked_flight"}
    assert all(row["reference_next_hp"]==1 for row in result["actions"]
               if row["action"] in ("wait","face","attack"))
