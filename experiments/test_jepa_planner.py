import torch
from experiments.recurrent_agents import Agent
from experiments.jepa_planner import Planner,CostWeights,predicted_cost


def test_planner_scores_all_actions_and_updates_memory_once():
    torch.manual_seed(4)
    agent=Agent().eval()
    planner=Planner(agent).session()
    ordinary=agent.session()
    values=[0.]*41
    values[16]=1
    action,belief=planner.choose(values)
    ordinary.choose(values)
    assert torch.equal(planner.hidden,ordinary.hidden)
    assert len(planner.last_plan["predictions"])==8
    assert action==min(range(8),key=lambda a:planner.last_plan["scores"][a])
    assert len(belief)==5
    assert Planner(agent).session().hidden is None


def test_cost_penalizes_injury_and_failed_attack():
    obs=torch.zeros(41); obs[16]=1
    future=torch.zeros(8,43); future[:,16]=1
    safe=predicted_cost(obs,future)
    future[1,16]=.5
    hurt=predicted_cost(obs,future)
    assert hurt[1]>safe[1]
    future[7,41]=1; future[7,42]=.5
    assert predicted_cost(obs,future)[7]<safe[7]


def test_escape_and_approach_have_opposite_distance_objectives():
    obs=torch.zeros(41); obs[5]=1; obs[6]=.5; obs[16]=1
    future=torch.zeros(8,43); future[:,16]=1; future[:,5]=1; future[:,6]=1
    approach=predicted_cost(obs,future)[1]
    obs[16]=.2; future[:,16]=.2
    escape=predicted_cost(obs,future)[1]
    assert approach>0 and escape<0
    future[4,29]=1
    assert predicted_cost(obs,future)[4]>escape


def test_zero_cost_weights_recover_policy_argmax():
    agent=Agent().eval()
    weights=CostWeights(0,0,0,0,0,1)
    values=[0.]*41
    assert Planner(agent,weights).session().choose(values)==agent.session().choose(values)


def test_half_health_is_more_defensive_without_a_threshold_jump():
    future=torch.zeros(8,43); future[:,5]=1; future[:,6]=.25
    obs=torch.zeros(41); obs[5]=1; obs[6]=.5
    obs[16]=1; future[:,16]=.9
    healthy=predicted_cost(obs,future)[1]
    obs[16]=.5; future[:,16]=.4
    wounded=predicted_cost(obs,future)[1]
    assert wounded>healthy
    obs[16]=.50001; future[:,16]=.40001
    assert abs(float(predicted_cost(obs,future)[1]-wounded))<.001


def test_conditions_modulate_goals_relative_to_baseline_and_absence():
    from experiments.jepa_planner import goal_modulation
    baseline=goal_modulation()
    context={"present":["dopamine","cortisol"],"deltas":{"dopamine":.4,"cortisol":.4}}
    changed=goal_modulation(context)
    assert changed["injury_multiplier"]>baseline["injury_multiplier"]
    assert changed["attack_multiplier"]>baseline["attack_multiplier"]
    assert changed["defensive_shift"]>0
    assert goal_modulation({"present":[],"deltas":context["deltas"]})==baseline
    assert goal_modulation({"present":context["present"],"deltas":{}})==baseline


def test_planner_uses_own_condition_without_expanding_network_or_double_memory_update():
    agent=Agent().eval()
    session=Planner(agent).session()
    values=[0.]*41; values[16]=.5
    # No contact means orientation uses own pose; no game state is needed.
    obs={"values":values,"contact":None,"self":{
        "orientation_roll_pitch_heading_deg":[0,0,0],
        "condition":{"present":["cortisol"],"deltas":{"cortisol":.4}}}}
    session.choose_observation("actor",obs)
    ordinary=agent.session(); ordinary.choose(values)
    assert torch.equal(session.hidden,ordinary.hidden)
    assert session.last_plan["goal_modulation"]["injury_multiplier"]==1.2
