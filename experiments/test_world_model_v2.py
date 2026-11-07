"""V2 checkpoint and separate admission heads."""
import torch
from experiments.world_model_v2 import Network, Predictor, load_predictor


def test_v2_roundtrip_preserves_both_probability_heads(tmp_path):
    torch.manual_seed(17)
    model = Network()
    before = Predictor(model)([[0.0]*44,[1.0]*44])
    assert all(len(row)==19 and all(0<=p<=1 for p in row[-2:]) for row in before)
    path = tmp_path/"v2.pt"
    torch.save({"version":2,"state_dict":model.state_dict()},path)
    assert load_predictor(path)([[0.0]*44,[1.0]*44])==before


def test_wrong_checkpoint_version_is_rejected(tmp_path):
    import pytest
    path = tmp_path/"wrong.pt"
    torch.save({"version":1},path)
    with pytest.raises(ValueError,match="v2"):
        load_predictor(path)
