from copy import deepcopy
import math
import torch
from spider_sim.lifecycle_model import features,FEATURE_NAMES,action_features,ACTION_SIZE,LifecycleModel
from experiments.train_lifecycle_model import classification,load_corpus


def observation():
    return {'self':{'hunger':.3,'hp_fraction':.8,'heading_deg':30,'sex':'male','local_clearance_ft':[10]*8},
        'contacts':[{'track':'contact_1','modality':'vision','range_ft':12,'bearing_deg':60,'body_dimensions_ft':[1,2,1]}],
        'food_in_view':[{'track':'contact_2','range_ft':3,'relative_position_ft':[3,0,0],'wrapped':True,'remaining':.2}],
        'last_result':{'action':'attack','hit':None},'incoming_damage':[]}


def test_features_use_private_inputs_not_teacher_or_global_metadata():
    obs=observation();before=deepcopy(obs);expected=features(obs)
    obs.update(teacher_goal='feed',actor='secret_id',audit={'target_kind':'rat'},after_round={'hunger':0})
    obs['self'].update(seeking_mate=True,goal='court')
    assert features(obs)==expected and len(expected)==len(FEATURE_NAMES)==99
    assert all(math.isfinite(v) for v in expected)
    assert before['last_result']['hit'] is None
    renamed=deepcopy(before);renamed['contacts'][0]['track']='unrelated'
    assert features(renamed)==expected


def test_bearings_are_relative_to_own_heading():
    obs=observation();turned=deepcopy(obs);turned['self']['heading_deg']+=90
    turned['contacts'][0]['bearing_deg']+=90;turned['food_in_view'][0]['relative_position_ft']=[0,3,0]
    assert torch.allclose(torch.tensor(features(obs)),torch.tensor(features(turned)),atol=1e-6)
    command={'action':'move','heading_deg':60,'distance_ft':5}
    changed={**command,'heading_deg':150}
    assert len(action_features(obs,command))==ACTION_SIZE
    assert torch.allclose(torch.tensor(action_features(obs,command)),torch.tensor(action_features(turned,changed)),atol=1e-6)


def test_gru_is_causal_and_padding_does_not_change_valid_states():
    torch.manual_seed(42);model=LifecycleModel(True).eval()
    x=torch.randn(2,7,99);lengths=torch.tensor([7,3])
    with torch.no_grad():
        y=model(x,lengths)
        assert torch.allclose(y[1,:3],model(x[1:2,:3],torch.tensor([3]))[0],atol=1e-6)
        changed=x.clone();changed[:,4:]+=100
        assert torch.allclose(y[:,:4],model(changed,lengths)[:,:4],atol=1e-6)
        assert y.shape==(2,7,9)


def test_absent_validation_classes_are_reported_not_perfect_scores():
    matrix=torch.zeros(9,9,dtype=torch.long);matrix[0,0]=8;matrix[0,1]=2
    result=classification(matrix)
    assert result['accuracy']==.8
    assert result['per_goal']['court']['support']==0
    assert result['per_goal']['court']['f1']==0
    assert result['balanced_accuracy_present']==.8


def test_forecast_masks_unknown_next_context_but_keeps_terminal_body_targets():
    from experiments.train_lifecycle_predictor import loss
    target=torch.zeros(2,66);valid=torch.tensor([True,False]);scale=torch.ones(66)
    pred=target.clone();pred[1,:64]=100
    assert loss(pred,target,valid,scale)==0
    pred[1,64:]=1
    assert loss(pred,target,valid,scale)==.5
    assert torch.isfinite(loss(pred,target,torch.zeros(2,dtype=torch.bool),scale))


def test_sequence_encoder_does_not_observe_future_or_teacher_labels():
    from experiments.train_lifecycle_predictor import transitions
    obs=observation();rows=[{'observation':obs,'command':{'action':'wait'},
        'after_round':{'hp_fraction':.7,'hunger':.31},'teacher_goal':'feed'}]
    sequence={'x':torch.tensor([features(obs)]),'y':torch.tensor([3]),'rows':rows}
    model=LifecycleModel(True).eval()
    state,action,target,valid=transitions(model,[sequence])
    assert valid.tolist()==[False]
    assert torch.all(target[:,:64]==0)
    assert torch.allclose(target[0,64:],torch.tensor([-.1,.01]),atol=1e-6)
    rows[0]['after_round']={'hp_fraction':0,'hunger':1};rows[0]['teacher_goal']='survive'
    state2,action2,target2,_=transitions(model,[sequence])
    assert torch.equal(state,state2) and torch.equal(action,action2)
    assert not torch.equal(target,target2)
