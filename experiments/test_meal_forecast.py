import copy
import torch
from experiments.compare_meal_defense import setup,scenarios
from experiments.meal_forecast_data import fork_outcome,private_context,begin_round,SPIDER
from experiments.train_meal_forecast import inputs,loss,OutcomeHead
from spider_sim.pursuit import PursuitModel


def test_counterfactual_forks_are_repeatable_and_do_not_mutate_parent_or_weights():
    model=PursuitModel(21).eval().requires_grad_(False)
    case=list(scenarios([81000]))[0]
    world,policies=setup(case,model,'guard')
    obs=begin_round(world,policies)
    spider=policies[SPIDER];beast=policies['freight_shadowbeast']
    before=(copy.deepcopy(world.positions),world.random.getstate(),copy.deepcopy(world.food),copy.deepcopy(world.hunger),world.config.characters[SPIDER].character.current_hp)
    weights={k:v.clone() for k,v in model.state_dict().items()}
    a=fork_outcome(world,spider,beast,'guard',901)
    b=fork_outcome(world,spider,beast,'guard',901)
    assert a==b and len(a)==3 and all(len(h)==4 for h in a)
    after=(world.positions,world.random.getstate(),world.food,world.hunger,world.config.characters[SPIDER].character.current_hp)
    assert before==after and spider.hunter.hidden is None
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in weights.items())
    assert not any(p.requires_grad for p in model.parameters())
    assert all(a[i][0]<=a[i+1][0] and a[i][2]<=a[i+1][2] and a[i][3]<=a[i+1][3] for i in range(2))


def test_predictor_inputs_exclude_metadata_and_hidden_opponent_state():
    model=PursuitModel(21);case=list(scenarios([81000]))[0]
    world,policies=setup(case,model,'guard');obs=begin_round(world,policies)
    context=private_context(obs,policies[SPIDER].meal,'wrap')
    assert len(context)==8 and all(isinstance(v,(int,float)) for v in context)
    record={'latent':[0.]*48,'context':context,'condition':case}
    before=inputs(record,1,'frozen_memory_context')
    record['condition']={'behavior':'returner','enemy_hp':999,'return_delay':999,'seed':999}
    assert inputs(record,1,'frozen_memory_context')==before
    assert len(before)==59 and before[-3:]==[0.,1.,0.]


def test_outcome_head_learns_only_its_own_parameters_from_future_labels():
    torch.manual_seed(0);head=OutcomeHead(59)
    x=torch.randn(12,59);targets=torch.zeros(12,3,4);targets[:6,:,2]=1
    optimizer=torch.optim.Adam(head.parameters(),lr=.01)
    first=float(loss(head(x),targets).detach())
    for _ in range(30):
        objective=loss(head(x),targets);optimizer.zero_grad();objective.backward();optimizer.step()
    assert float(loss(head(x),targets).detach())<first*.8
