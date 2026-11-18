import torch
from experiments.train_recurrent_meal_forecast import RecurrentOutcome
from experiments.train_meal_forecast import loss
from spider_sim.pursuit import PursuitModel


def test_44_input_initialization_preserves_original_representation_and_source():
    torch.manual_seed(0);source=PursuitModel(21).eval().requires_grad_(False)
    model=RecurrentOutcome(source,44).eval();history=torch.randn(2,4,44);lengths=torch.tensor([4,4])
    with torch.no_grad():
        expected,_=source.represent(history[:,:,:36])
        actual=model.represent(history,lengths)
    torch.testing.assert_close(actual,expected[:,-1],atol=1e-6,rtol=1e-6)
    assert model.encoder[0].weight.shape==(64,44)
    assert torch.count_nonzero(model.encoder[0].weight[:,36:])==0
    assert model.head.layers[0].in_features==51  # No direct context bypass.


def test_padding_cannot_change_predictions_for_a_shorter_history():
    torch.manual_seed(0);model=RecurrentOutcome(PursuitModel(21),44).eval()
    history=torch.randn(2,4,44);lengths=torch.tensor([1,4]);changed=history.clone();changed[0,1:]=100
    torch.testing.assert_close(model(history,lengths),model(changed,lengths))


def test_training_updates_extra_inputs_and_gru_without_mutating_source_actor():
    torch.manual_seed(0);source=PursuitModel(21).eval().requires_grad_(False)
    original={k:v.clone() for k,v in source.state_dict().items()};model=RecurrentOutcome(source,44)
    old_gru={k:v.clone() for k,v in model.memory.state_dict().items()}
    history=torch.rand(3,4,44);lengths=torch.tensor([4,3,2]);target=torch.zeros(9,3,4)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    objective=loss(model(history,lengths).reshape(-1,3,4),target)
    optimizer.zero_grad();objective.backward()
    assert model.encoder[0].weight.grad[:,36:].abs().sum()>0
    optimizer.step()
    assert model.encoder[0].weight[:,36:].abs().sum()>0
    assert any(not torch.equal(v,model.memory.state_dict()[k]) for k,v in old_gru.items())
    assert all(torch.equal(v,source.state_dict()[k]) for k,v in original.items())


def test_history_variant_can_use_prior_hunger_but_snapshot_cannot():
    torch.manual_seed(0);source=PursuitModel(21)
    history=RecurrentOutcome(source,44).eval();snapshot=RecurrentOutcome(source,44,snapshot=True).eval()
    with torch.no_grad():history.encoder[0].weight[:,36]=.1
    snapshot.load_state_dict(history.state_dict())
    a=torch.rand(1,3,44);b=a.clone();a[0,0,36]=0;b[0,0,36]=1
    lengths=torch.tensor([3])
    assert not torch.allclose(history.represent(a,lengths),history.represent(b,lengths),atol=1e-7,rtol=0)
    torch.testing.assert_close(snapshot.represent(a,lengths),snapshot.represent(b,lengths))


def test_saved_recurrent_predictor_roundtrip_and_source_contract(tmp_path):
    import hashlib
    import pytest
    from spider_sim.pursuit import save
    from experiments.train_recurrent_meal_forecast import load_predictor
    source=PursuitModel(21);source_path=tmp_path/'source.pt';save(source,source_path)
    model=RecurrentOutcome(source,44).eval()
    sha=hashlib.sha256(source_path.read_bytes()).hexdigest();path=tmp_path/'predictor.pt'
    torch.save({'schema':'recurrent-meal-forecast-v1','variant':'retrained44','inputs':44,'snapshot':False,
                'state':model.state_dict(),'source_model_sha256':sha},path)
    restored=load_predictor(path,source_path)
    history=torch.rand(2,4,44);lengths=torch.tensor([2,4])
    torch.testing.assert_close(model(history,lengths),restored(history,lengths))
    assert not any(p.requires_grad for p in restored.parameters())
    other=tmp_path/'other.pt';save(PursuitModel(21),other)
    with pytest.raises(ValueError,match='source checkpoint mismatch'):load_predictor(path,other)
