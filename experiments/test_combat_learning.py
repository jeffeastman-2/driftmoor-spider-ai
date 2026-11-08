import torch
from experiments.recurrent_agents import Agent,save,load
from experiments.combat_learning import residual_predictor,collect_experience,replay_tensors,fit_predictor,prediction_loss
from spider_sim.dual_encounter import SPIDER


def test_residual_initialization_preserves_prior_except_hp_row_and_freezes_scope(tmp_path):
    prior=Agent().eval();agent=residual_predictor(prior)
    for name,param in agent.named_parameters():
        assert param.requires_grad==name.startswith("predictor.")
        if name not in ("predictor.2.weight","predictor.2.bias"):
            assert torch.equal(param,prior.state_dict()[name])
    obs=torch.zeros(1,1,41);obs[:,:,16]=.7
    _,_,future,_=agent(obs,torch.zeros(1,1,dtype=torch.long))
    assert future[0,0,16]==obs[0,0,16]
    path=tmp_path/"experience.pt";save(agent,path)
    restored=load(path)
    assert restored.prediction_mode=="hp_delta"
    assert torch.equal(restored(obs,torch.zeros(1,1,dtype=torch.long))[2],future)
    assert prior.prediction_mode=="absolute"


def test_experience_keeps_sequential_horizon_and_explores_repeatably():
    torch.manual_seed(1);agent=Agent().eval()
    a=collect_experience(agent,SPIDER,[300000],1.,rounds=3)
    assert a==collect_experience(agent,SPIDER,[300000],1.,rounds=3)
    rows=a[0]["rows"]
    assert rows and all(row["exploration"] for row in rows)
    for first,second in zip(rows,rows[1:]):
        assert first["next_observation"]==second["observation"]
    assert "terminal" in rows[-1]


def test_replay_training_updates_only_predictor_and_reduces_error():
    torch.manual_seed(1);prior=Agent().eval();agent=residual_predictor(prior)
    obs=[0.]*41;obs[16]=1
    future=obs.copy();future[16]=.8
    episode={"rows":[{"observation":obs,"action":7,"next_observation":future,"feedback":[1.,.2]}]*4}
    data=replay_tensors(agent,[episode]);before=float(prediction_loss(agent,data))
    fit_predictor(agent,[episode],[episode],epochs=30)
    assert float(prediction_loss(agent,data))<before
    for name,param in agent.state_dict().items():
        if not name.startswith("predictor."):
            assert torch.equal(param,prior.state_dict()[name])


def test_terminal_transition_is_retained_instead_of_discarding_dead_actor(monkeypatch):
    from spider_sim.dual_encounter import DualEncounter
    original=DualEncounter.act
    def fatal_transition(world,actor,*args,**kwargs):
        result=original(world,actor,*args,**kwargs)
        # Exercise recorder termination independently of stochastic combat rolls.
        world.config.characters[actor].character.current_hp=0
        return result
    monkeypatch.setattr(DualEncounter,"act",fatal_transition)
    episodes=collect_experience(Agent().eval(),SPIDER,[300000],rounds=3)
    rows=episodes[0]["rows"]
    assert len(rows)==1 and rows[0]["terminal"]
    assert rows[0]["next_observation"][16]==0
