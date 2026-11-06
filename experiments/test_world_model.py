"""Tests for the trained predictor's artifact and inference boundary."""
import torch
from experiments.world_model import Network, Predictor, load_predictor


def test_saved_weights_roundtrip_and_probabilities(tmp_path):
    torch.manual_seed(7)
    model = Network()
    rows = [[0.0]*44,[1.0]*44]
    expected = Predictor(model)(rows)
    artifact = tmp_path/"predictor.pt"
    torch.save({"state_dict":model.state_dict(),"inputs":44,"observations":17},artifact)
    actual = load_predictor(artifact)(rows)
    assert actual == expected
    assert all(len(row)==18 and 0 <= row[-1] <= 1 for row in actual)


def test_training_changes_prediction_weights():
    torch.manual_seed(7)
    model = Network()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
    x = torch.zeros(8,44)
    y = torch.ones(8,18)
    before = model(x).detach().clone()
    for _ in range(10):
        loss = ((model(x)-y)**2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert ((model(x)-y)**2).mean() < ((before-y)**2).mean()


def test_persistence_baseline_uses_only_current_observation():
    from experiments.run_world_model import persistence_predictions
    x = torch.arange(88,dtype=torch.float32).reshape(2,44)
    y = torch.zeros(3,18)
    y[:,-1] = torch.tensor([1,0,0])
    result = persistence_predictions(x,y)
    assert torch.equal(result[:,:17],x[:,17:34])
    assert torch.allclose(result[:,-1],torch.full((2,),1/3))
