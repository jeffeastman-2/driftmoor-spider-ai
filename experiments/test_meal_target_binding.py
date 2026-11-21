from copy import deepcopy
from types import SimpleNamespace
import gzip,json
import pytest
import torch
from spider_sim.meal_target import select_meal
from spider_sim.lifecycle_model import FEATURE_NAMES_V2,FEATURE_NAMES_V3,features_v2,features_v3,GOALS,LifecycleModel,feature_names
from spider_sim.learned_lifecycle import LearnedLifecyclePolicy,load
from experiments.train_lifecycle_model import load_corpus


def obs():
    return {'self':{'hunger':.8,'hp_fraction':1,'heading_deg':0},'contacts':[],
        'food_in_view':[{'track':'near','range_ft':5,'relative_position_ft':[0,5,0],'wrapped':True,'remaining':.2},
            {'track':'middle','range_ft':7,'relative_position_ft':[0,-7,0],'wrapped':False,'remaining':.5},
            {'track':'far','range_ft':30,'relative_position_ft':[30,0,0],'wrapped':False,'remaining':2.5}]}


class Hunter:
    locomotion=None
    def __init__(self):self.task=SimpleNamespace(previous=None,track=None,last=None);self.last_plan={}
    def choose(self,observation):return {'action':'wait'}


class Capture:
    feature_version='v3'
    def __init__(self,goal='wrap'):self.goal=goal;self.during_inference=lambda:None
    def step(self,x,hidden=None):
        self.seen=x.clone();self.during_inference()
        logits=torch.full((1,1,len(GOALS)),-10.);logits[0,0,GOALS.index(self.goal)]=10
        return logits,None


def test_retained_meal_must_be_currently_perceived_and_can_be_outside_nearest_two():
    o=obs();assert select_meal(o,'far')['track']=='far'
    assert select_meal(o,'lost')['track']=='near'
    assert select_meal({'food_in_view':[]},'far') is None


def test_actor_and_motor_share_one_selection_even_if_memory_changes_during_inference(monkeypatch):
    import spider_sim.learned_lifecycle as runtime
    calls=[];original=runtime.select_meal
    def counted(o,track):calls.append(track);return original(o,track)
    monkeypatch.setattr(runtime,'select_meal',counted)
    model=Capture();policy=LearnedLifecyclePolicy(Hunter(),model,'male');policy.meal='far'
    model.during_inference=lambda:setattr(policy,'meal','near')
    command=policy.choose(obs())
    assert calls==['far']
    assert model.seen[0,0,FEATURE_NAMES_V3.index('selected_meal_range_60ft')]==.5
    assert model.seen[0,0,FEATURE_NAMES_V3.index('meal_0_1')]==pytest.approx(5/60)
    assert command=={'action':'wrap','target':'far'}  # No feasibility mask or teacher override.
    assert policy.meal=='far'


def test_approach_uses_selected_meal_direction_and_missing_selection_stays_missing():
    model=Capture('approach_meal');policy=LearnedLifecyclePolicy(Hunter(),model,'male');policy.meal='far'
    command=policy.choose(obs());assert command['heading_deg']==0 and command['distance_ft']==5
    empty={'self':{'hunger':.8,'hp_fraction':1},'contacts':[],'food_in_view':[]}
    model.goal='wrap';model.during_inference=lambda:empty['food_in_view'].extend(obs()['food_in_view'])
    assert policy.choose(empty)=={'action':'wait'}


def test_bound_meal_encoder_is_versioned_private_and_requires_context(tmp_path):
    o=obs();o['selected_meal']=select_meal(o,'far');expected=features_v3(o)
    assert len(expected)==len(FEATURE_NAMES_V3)==152 and expected[:145]==features_v2(o)
    renamed=deepcopy(o);renamed['selected_meal']['track']='different-private-token'
    renamed.update(teacher_goal='feed',after_round={'hunger':0})
    assert features_v3(renamed)==expected
    with pytest.raises(ValueError,match='selected-meal'):features_v3(obs())
    for version in ('v1','v2','v3'):
        model=LifecycleModel(True,feature_version=version);path=tmp_path/f'{version}.pt'
        torch.save({'state_dict':model.state_dict(),'recurrent':True,'features':feature_names(version),'goals':GOALS},path)
        assert load(path).feature_version==version


def test_old_recordings_reconstruct_selected_target_causally_across_other_activities(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps({'splits':{'train':[11]}}));root=tmp_path/'seed-11';root.mkdir()
    first=obs();first['food_in_view']=[first['food_in_view'][2]]
    rows=[{'seed':11,'split':'train','actor':'a','turn':1,'observation':first,'teacher_goal':'approach_meal','executed_goal':'approach_meal','command':{'action':'move'}},
        {'seed':11,'split':'train','actor':'a','turn':2,'observation':obs(),'teacher_goal':'approach_meal','executed_goal':'hunt','command':{'action':'wait'}},
        {'seed':11,'split':'train','actor':'a','turn':3,'observation':obs(),'teacher_goal':'approach_meal','executed_goal':'wrap','command':{'action':'wrap','target':'far'}}]
    def write():
        with gzip.open(root/'decisions.jsonl.gz','wt') as f:
            for r in rows:f.write(json.dumps(r)+'\n')
    write();seq=load_corpus(tmp_path,'v3')['train'][0]
    index=FEATURE_NAMES_V3.index('selected_meal_range_60ft')
    assert seq['x'][:,index].tolist()==[.5,.5,.5]
    before=seq['x'].clone();rows[1]['teacher_goal']='court';rows[2]['teacher_goal']='defend';write()
    assert torch.equal(before,load_corpus(tmp_path,'v3')['train'][0]['x'])
    rows[2]['command']['target']='near';write()
    with pytest.raises(ValueError,match='motor target'):load_corpus(tmp_path,'v3')
