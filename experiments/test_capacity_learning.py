import torch
from experiments.recurrent_agents import Agent,save,load
from experiments.capacity_learning import initialize,sequence_batch,loss,train_variant


def toy():
    obs=[.1]*41;obs[16]=1
    future=obs.copy();future[16]=.8
    return [{"rows":[{"observation":obs,"action":7,"next_observation":future,"feedback":[1,.2]}]*3}]*2


def test_deeper_initial_encoder_preserves_original_function_and_checkpoint(tmp_path):
    torch.manual_seed(1);prior=Agent().eval()
    deep=initialize(prior,"deeper",42)
    obs=torch.randn(4,7,41)
    assert torch.allclose(prior.encoder(obs),deep.encoder(obs),atol=1e-6)
    assert torch.allclose(prior(obs)[0],deep(obs)[0],atol=1e-6)
    assert deep.encoder[0].weight[32:].abs().sum()>0
    path=tmp_path/"deep.pt";save(deep,path)
    assert load(path).encoder_depth==2
    assert torch.allclose(load(path)(obs)[0],deep(obs)[0],atol=1e-6)
    assert all(torch.equal(value,load(path).state_dict()[key]) for key,value in deep.state_dict().items())


def test_padding_does_not_contribute_and_joint_gradients_reach_memory():
    prior=Agent().eval();agent=initialize(prior,"unfrozen",1)
    episodes=toy();episodes[1]={"rows":episodes[1]["rows"][:1]}
    data=sequence_batch(episodes)
    before=loss(agent,data)
    changed=list(data);changed[2]=data[2].clone();changed[2][~data[3]]=1000
    assert torch.equal(before,loss(agent,changed))
    before.backward()
    assert agent.encoder[0].weight.grad.abs().sum()>0
    assert agent.memory.weight_ih_l0.grad.abs().sum()>0
    assert agent.policy.weight.grad is None


def test_training_scopes_and_deep_added_units_can_learn():
    torch.manual_seed(1);prior=Agent().eval()
    for variant in ("predictor_only","unfrozen","deeper"):
        initial=initialize(prior,variant,42)
        agent,result=train_variant(prior,variant,toy(),toy(),epochs=4,seed=42)
        assert torch.equal(agent.policy.weight,prior.policy.weight)
        if variant=="predictor_only":
            assert torch.equal(agent.encoder[0].weight,prior.encoder[0].weight)
        else:
            assert not torch.equal(agent.memory.weight_ih_l0,prior.memory.weight_ih_l0)
        if variant=="deeper":
            assert agent.encoder[2].weight[:,32:].abs().sum()>0
