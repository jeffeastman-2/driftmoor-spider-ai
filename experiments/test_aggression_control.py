import torch
from experiments.jepa_planner import predicted_cost
from experiments.run_aggression_control import SETTINGS


def test_wounded_costs_unchanged_for_all_aggression_settings():
    torch.manual_seed(1)
    futures=torch.rand(8,43)
    for hp in (.1,.25,.35):
        current=torch.rand(41);current[16]=hp
        baseline=predicted_cost(current,futures,SETTINGS["baseline"])
        for weights in SETTINGS.values():
            assert torch.equal(baseline,predicted_cost(current,futures,weights))


def test_healthy_attack_more_valuable_but_invalid_attack_still_penalized():
    current=torch.zeros(41);current[16]=1
    futures=torch.zeros(8,43);futures[:,16]=1
    for weights in SETTINGS.values():
        assert predicted_cost(current,futures,weights)[7]==.5
    futures[7,41]=1;futures[7,42]=.2;futures[7,16]=.95
    costs=[float(predicted_cost(current,futures,weights)[7]) for weights in SETTINGS.values()]
    assert costs[0]>costs[1]>costs[2]
