from copy import deepcopy
import torch
import pytest
from spider_sim.learned_lifecycle import LearnedLifecyclePolicy
from spider_sim.lifecycle_model import LifecycleModel,GOALS
from spider_sim.feeding import FeedingPolicy
from spider_sim.pursuit import PursuitModel,PursuitPolicy
from spider_sim.ecology import EcologyEncounter,SPIDER,FEMALE_SPIDER
from spider_sim.djepa_ecology import senses


class ForcedGoal:
    goal='hunt'
    def step(self,x,hidden=None):
        logits=torch.full((1,1,len(GOALS)),-100.);logits[0,0,GOALS.index(self.goal)]=100.
        return logits,torch.ones(1,1,64)


def test_streamed_gru_matches_full_sequence_and_newborn_has_fresh_memory():
    torch.manual_seed(42);model=LifecycleModel(True).eval();x=torch.randn(1,8,99);hidden=None;stream=[]
    with torch.no_grad():
        for t in range(8):
            logits,hidden=model.step(x[:,t:t+1],hidden);stream.append(logits)
        assert torch.allclose(torch.cat(stream,1),model(x,torch.tensor([8])),atol=1e-6)
    p=LearnedLifecyclePolicy(PursuitPolicy(PursuitModel(21)),model,'male');p.hidden=hidden
    child=p.spawn(PursuitPolicy(PursuitModel(21)),'female')
    assert child.model is p.model and child.hidden is None and child.hunter is not p.hunter
    assert child.sex=='female'


def test_learned_choice_never_calls_teacher_or_silently_changes_goal(monkeypatch):
    world=EcologyEncounter(42,feeding_cycle=True,spider_pair=True)
    model=ForcedGoal();model.goal='court'
    p=LearnedLifecyclePolicy(PursuitPolicy(PursuitModel(21)),model,'male')
    def forbidden(*args):raise AssertionError('teacher called')
    monkeypatch.setattr(FeedingPolicy,'choose',forbidden)
    command=p.choose(senses(world,SPIDER))
    assert p.goal=='court' and command=={'action':'wait'} and p.unexecutable==1
    assert p.last_plan['selection_reason']=='learned_lifecycle_goal'


def test_response_court_motor_uses_visible_target_without_teacher_decision():
    w=EcologyEncounter(42,feeding_cycle=True,spider_pair=True)
    w.positions[SPIDER]=(103,140,'Fungal Stores');w.positions[FEMALE_SPIDER]=(107,140,'Fungal Stores')
    w.hunger[SPIDER]=.15;w.hunger[FEMALE_SPIDER]=.05
    model=ForcedGoal();model.goal='court';p=LearnedLifecyclePolicy(PursuitPolicy(PursuitModel(21)),model,'male')
    cmd=p.choose(senses(w,SPIDER));assert cmd=={'action':'court','target':w.tokens[SPIDER][FEMALE_SPIDER]}
    assert w.act(SPIDER,cmd)['status']=='courting'


def test_motor_execution_matches_teacher_when_given_same_goals(tmp_path):
    from experiments.watch_level2 import LiveRun
    from experiments.run_lifecycle_long import cache_static_geometry
    from spider_sim.ecology import ecology_round
    from spider_sim.pursuit import save
    weights=tmp_path/'pursuit.pt';model=PursuitModel(21);model.value_kind='negative_action_logit';save(model,weights)
    run=LiveRun(tmp_path/'run',42,100,spider_model=weights,spider_local_motion=True,replenish_prey=True,
        feeding_cycle=True,random_prey=True,spider_pair=True,lifecycle=True)
    cache_static_geometry(run.world.geometry);wrapped=set()
    for turn in range(100):
        run.set_hunger_rate(turn)
        for actor in run.world.spiders:
            teacher=run.policies[actor]
            if actor in wrapped:continue
            forced=ForcedGoal();shadow=LearnedLifecyclePolicy(deepcopy(teacher.hunter),forced,run.world.spiders[actor])
            original=teacher.choose
            def choose(obs,teacher=teacher,original=original,forced=forced,shadow=shadow):
                command=original(obs);forced.goal=teacher.goal
                actual=shadow.choose(obs)
                assert actual==command,(teacher.goal,actual,command)
                return command
            teacher.choose=choose;wrapped.add(actor)
        ecology_round(run.world,run.policies,turn)


@pytest.mark.parametrize('previous_goal',['seek_mate','approach_mate','court'])
def test_learned_executor_can_defend_against_attack_received_during_mating(previous_goal):
    w=EcologyEncounter(42,feeding_cycle=True,spider_pair=True)
    w.positions[SPIDER]=(103,140,'Fungal Stores')
    w.positions['freight_shadowbeast']=(107,140,'Fungal Stores')
    target=w.tokens[SPIDER]['freight_shadowbeast']
    obs=senses(w,SPIDER);obs['incoming_damage']=[{'damage_hp':1,'attacker':target}]
    model=ForcedGoal();model.goal='defend'
    p=LearnedLifecyclePolicy(PursuitPolicy(PursuitModel(21)),model,'male');p.goal=previous_goal
    assert p.choose(obs)=={'action':'attack','target':target}
    assert p.goal=='defend' and p.defense_track==target
