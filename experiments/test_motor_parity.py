from experiments.motor_parity import MotorMatchedBaseline
from spider_sim.ecology import OBSERVATION_SCHEMA
from spider_sim.djepa_ecology import candidates


def observation(x,y):
    import math
    return {'schema':OBSERVATION_SCHEMA,'self':{'hp_fraction':1.,'heading_deg':0.,'body_dimensions_ft':[4,4,2]},
            'contacts':[{'track':'opaque','modality':'vision','range_ft':math.hypot(x,y),'relative_position_ft':[x,y,0]}],
            'own_webs_in_view':[],'last_result':None}


def test_matched_script_uses_available_step_lengths_and_headings():
    policy=MotorMatchedBaseline()
    assert policy.choose(observation(20,20))=={'action':'move','heading_deg':45.,'distance_ft':10.}
    # The continuous script would request a three-foot step here.
    assert policy.choose(observation(8,0))=={'action':'move','heading_deg':0.,'distance_ft':5.}
    obs=observation(20,20);obs['self']['heading_deg']=17
    assert policy.choose(obs) in candidates(obs)


def test_matching_does_not_replace_attack_or_wait_decisions():
    policy=MotorMatchedBaseline()
    assert policy.choose(observation(5,0))=={'action':'attack','target':'opaque'}
    obs=observation(20,20);obs['contacts']=[]
    assert policy.choose(obs)=={'action':'wait'}


def test_precise_approach_adds_a_command_without_removing_existing_choices():
    from spider_sim.pursuit import actions
    obs=observation(8,0);goal=obs['contacts'][0]
    original=actions(obs,goal);augmented=actions(obs,goal,approach=True)
    assert all(c in augmented for c in original)
    assert {'action':'move','heading_deg':0.,'distance_ft':3.} in augmented
    assert augmented==actions(obs,goal,approach=True)


def test_precise_approach_is_not_a_forced_scripted_fallback():
    import torch
    from experiments.train_single_prey import fixture
    from spider_sim.ecology import SPIDER
    from spider_sim.djepa_ecology import senses
    from spider_sim.pursuit import PursuitPolicy
    class PreferWait:
        action_size=21
        def represent(self,obs,hidden):return torch.zeros(1,1,48),torch.zeros(1,1,48)
        def values(self,state,choices):return torch.arange(len(choices),dtype=torch.float32)
    world,_,_=fixture(500000,'stationary')
    policy=PursuitPolicy(PreferWait(),approach=True)
    assert policy.choose(senses(world,SPIDER))=={'action':'wait'}
    assert policy.last_plan['approach_candidate']


def test_imitation_checkpoint_reports_action_scores_not_cost_forecasts(tmp_path):
    from experiments.train_single_prey import fixture
    from spider_sim.ecology import SPIDER
    from spider_sim.djepa_ecology import senses
    from spider_sim.pursuit import PursuitModel,PursuitPolicy,save,load
    model=PursuitModel(21);model.value_kind='negative_action_logit'
    path=tmp_path/'actor.pt';save(model,path);restored=load(path)
    policy=PursuitPolicy(restored,approach=True)
    world,_,_=fixture(500000,'stationary');policy.choose(senses(world,SPIDER))
    assert policy.last_plan['value_kind']=='negative_action_logit'
    assert 'action_scores' in policy.last_plan and 'predicted_goal_costs' not in policy.last_plan
