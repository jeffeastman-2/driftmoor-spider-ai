from types import SimpleNamespace
from experiments.compare_meal_defense import ComparisonPolicy,episode,scenarios,setup
from spider_sim.pursuit import PursuitModel
from spider_sim.ecology import SPIDER


class Hunter:
    def __init__(self):
        self.task=SimpleNamespace(previous=None,track=None,last=None)
        self.locomotion=None;self.last_plan={}
    def choose(self,obs):return {'action':'wait'}


def observation(x=14,feedback=None,incoming=None):
    return {'self':{'hunger':.8,'hp_fraction':.9},'contacts':[
        {'track':'threat','modality':'vision','range_ft':x,'relative_position_ft':[x,0,0]}],
        'food_in_view':[{'track':'food','range_ft':4,'wrapped':True,'remaining':1}],
        'last_result':feedback,'incoming_damage':incoming or []}


def controller(mode):
    policy=ComparisonPolicy(Hunter(),mode);policy.goal='defend';policy.defense_track='threat';policy.meal='food'
    policy.last_sighting={'track':'threat','relative_position_ft':[4,0,0]}
    return policy


def test_guard_releases_after_measured_withdrawal_but_pursuit_keeps_chasing():
    guard=controller('guard');pursue=controller('pursue')
    assert guard.choose(observation())['action']=='feed'
    assert guard.last_plan['defense_release']=='observed_withdrawal'
    assert pursue.choose(observation())['action']=='move' and pursue.goal=='defend'


def test_guard_does_not_mistake_own_retreat_for_opponent_withdrawal():
    policy=controller('guard');policy.hunter.task.previous={'action':'move','heading_deg':180}
    policy.choose(observation(feedback={'action':'move','distance_ft':10}))
    assert policy.goal=='defend' and policy.releases==0


def test_resume_requires_successful_retaliation_and_no_new_damage():
    policy=controller('resume')
    assert policy.choose(observation(4,{'action':'attack','target':'threat','damage':1}))['action']=='feed'
    policy=controller('resume')
    policy.choose(observation(4,{'action':'attack','target':'threat','damage':0}))
    assert policy.goal=='defend'
    policy=controller('resume')
    policy.choose(observation(4,{'action':'attack','target':'threat','damage':1},[{'damage_hp':1,'attacker':'threat'}]))
    assert policy.goal=='defend'


def test_engine_episode_reproducible_and_scenario_grid_is_balanced():
    cases=list(scenarios([73000]));assert len(cases)==48
    model=PursuitModel(21);model.value_kind='negative_action_logit'
    a=episode(cases[0],model,'guard',16);b=episode(cases[0],model,'guard',16)
    assert a==b
    assert a['food_consumed']<=1.2+1e-8 and a['damage_taken']>=0
    world,_=setup(cases[0],model,'guard')
    assert world.active=={SPIDER,'freight_shadowbeast'}
    assert all(session.character.current_hp==0 for cid,session in world.config.characters.items() if cid not in world.active)


def test_guard_remembers_first_attacker_sighting_and_stops_at_first_withdrawal():
    policy=ComparisonPolicy(Hunter(),'guard');policy.goal='wrap';policy.meal='food'
    first=observation(4,incoming=[{'damage_hp':1,'attacker':'threat'}])
    assert policy.choose(first)['action']=='attack'
    assert policy.last_sighting['relative_position_ft']==[4,0,0]
    policy.hunter.task.previous={'action':'attack','target':'threat'}
    assert policy.choose(observation(14,{'action':'attack','target':'threat','damage':1}))['action']=='feed'


def test_experience_alignment_includes_next_attacker_response_and_masks_episode_end(tmp_path):
    import gzip,io,json
    from experiments.align_meal_experience import align
    model=PursuitModel(21);model.value_kind='negative_action_logit'
    case=list(scenarios([73000]))[1];case['health']=1.
    captured=io.StringIO();result=episode(case,model,'guard',8,captured,0)
    raw=[json.loads(line) for line in captured.getvalue().splitlines()]
    assert len(raw)>1
    (tmp_path/'episodes.jsonl').write_text(json.dumps(result)+'\n')
    with gzip.open(tmp_path/'transitions.jsonl.gz','wt') as f:f.write(captured.getvalue())
    stats=align(tmp_path)
    with gzip.open(tmp_path/'decision_transitions.jsonl.gz','rt') as f:aligned=[json.loads(line) for line in f]
    assert stats['transitions']==len(raw) and stats['episodes']==1
    assert aligned[0]['next_observation']==raw[1]['observation']
    assert aligned[-1]['next_observation'] is None
    assert aligned[-1]['terminal_outcome']['hp']==result['final_hp']
    assert aligned[-1]['terminal_outcome']['hunger']==result['final_hunger']
    assert aligned[-1]['terminal']!=aligned[-1]['truncated']
