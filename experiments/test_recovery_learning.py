from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from spider_sim.lifecycle_model import features,features_v2,FEATURE_NAMES_V2,LifecycleModel
from spider_sim.learned_lifecycle import LearnedLifecyclePolicy,load
from spider_sim.recovery_advisor import RecoveryAdvisor
from spider_sim.feeding import FeedingPolicy
from experiments.test_lifecycle_model import observation
from experiments.test_learned_lifecycle import ForcedGoal
from spider_sim.pursuit import PursuitModel,PursuitPolicy


def test_v2_preserves_refusal_and_boundary_measurement_without_future_labels():
    o=observation();o['last_result']={'action':'wrap','status':'food_not_reachable'}
    o['food_in_view'][0]['range_ft']=7
    vals=dict(zip(FEATURE_NAMES_V2,features_v2(o)))
    assert len(vals)==145 and vals['last_status_food_not_reachable']==1
    assert vals['meal_0_interaction_margin_6ft']==pytest.approx(-1/6)
    assert features_v2(o)[:99]==features(o)
    changed=deepcopy(o);changed.update(teacher_goal='feed',executed_goal='wrap',advisor_command={'action':'feed'},after_round={'hp_fraction':0})
    assert features_v2(changed)==features_v2(o)
    changed['last_result']['status']='wrapped'
    assert features(changed)==features(o) and features_v2(changed)!=features_v2(o)


def test_v1_and_v2_checkpoint_loading_and_streamed_v2_memory(tmp_path):
    for version,n in [('v1',99),('v2',145)]:
        from spider_sim.lifecycle_model import feature_names,GOALS
        m=LifecycleModel(True,feature_version=version).eval();path=tmp_path/f'{version}.pt'
        torch.save({'state_dict':m.state_dict(),'recurrent':True,'features':feature_names(version),'goals':GOALS},path)
        restored=load(path);assert restored.feature_version==version
        x=torch.randn(1,4,n);hidden=None;outputs=[]
        with torch.no_grad():
            for t in range(4):
                y,hidden=restored.step(x[:,t:t+1],hidden);outputs.append(y)
            assert torch.allclose(torch.cat(outputs,1),restored(x,torch.tensor([4])),atol=1e-6)


def test_advisor_labels_actual_failed_wrap_state_without_executing_or_mutating_student():
    model=ForcedGoal();model.goal='wrap'
    student=LearnedLifecyclePolicy(PursuitPolicy(PursuitModel(21)),model,'male')
    student.goal='wrap';student.meal='contact_2'
    obs=observation();obs['self']['hunger']=.8;obs['food_in_view'][0].update(range_ft=12,relative_position_ft=[12,0,0],wrapped=False)
    obs['last_result']={'action':'wrap','status':'food_not_reachable','target':'contact_2'}
    original=deepcopy(obs);before=(student.goal,student.meal,student.hidden,student.hunter.hidden)
    goal,proposal=RecoveryAdvisor().recommend(obs,student)
    assert goal=='approach_meal' and proposal['action']=='move'
    assert obs==original and before==(student.goal,student.meal,student.hidden,student.hunter.hidden)


def test_label_only_teacher_matches_real_teacher_goal_over_private_history():
    real=FeedingPolicy(PursuitPolicy(PursuitModel(21),approach=True,local_motion=True))
    advisor=RecoveryAdvisor();fake=SimpleNamespace(goal='hunt',meal=None,mate_track=None,defense_track=None,defense_age=0,
        mate_hunger_threshold=.1,resume_hunting=.3,mate_hunger_limit=.3,courtship_ready=True)
    for hunger,rng,wrapped in [(.8,12,False),(.81,4,False),(.82,4,True),(.05,4,True),(.15,20,True),(.3,20,True)]:
        obs=observation();obs['schema']='ecology-djepa-v1';obs['self'].update(hunger=hunger,speed_ft=30,reach_ft=5,max_hp=26,height_above_floor_ft=0,body_dimensions_ft=[4,4,2])
        obs['food_in_view'][0].update(range_ft=rng,relative_position_ft=[rng,0,0],wrapped=wrapped)
        for key in ('goal','meal','mate_track','defense_track','defense_age','mate_hunger_threshold','resume_hunting','mate_hunger_limit','courtship_ready'):
            setattr(real,key,getattr(fake,key))
        advised,_=advisor.recommend(obs,fake);real.choose(obs)
        assert advised==real.goal
        for key in ('goal','meal','mate_track','defense_track','defense_age'):setattr(fake,key,getattr(real,key))


def test_recovery_dataset_uses_advice_as_label_and_actual_observation_as_input(tmp_path):
    import json,gzip
    from experiments.train_lifecycle_model import load_corpus
    from spider_sim.lifecycle_model import GOALS
    (tmp_path/'manifest.json').write_text(json.dumps({'splits':{'train':[11]}}))
    root=tmp_path/'seed-11';root.mkdir();obs=observation()
    row={'seed':11,'split':'train','actor':'a','turn':1,'observation':obs,
        'teacher_goal':'approach_meal','executed_goal':'wrap','command':{'action':'wrap'},
        'advisor_command':{'action':'move'},'result':{'status':'food_not_reachable'}}
    with gzip.open(root/'decisions.jsonl.gz','wt') as f:f.write(json.dumps(row)+'\n')
    sequence=load_corpus(tmp_path,'v2')['train'][0]
    assert sequence['y'].tolist()==[GOALS.index('approach_meal')]
    assert torch.allclose(sequence['x'],torch.tensor([features_v2(obs)]))


def test_advice_on_damage_during_actual_mating_activity_is_defense():
    student=SimpleNamespace(goal='seek_mate',meal=None,mate_track=None,defense_track=None,defense_age=0,
        mate_hunger_threshold=.1,resume_hunting=.3,mate_hunger_limit=.3,courtship_ready=True)
    obs=observation();obs['self']['hunger']=.15;obs['contacts'][0]['relative_position_ft']=[12,0,0];obs['incoming_damage']=[{'damage_hp':1,'attacker':'contact_1'}]
    goal,_=RecoveryAdvisor().recommend(obs,student)
    assert goal=='defend' and student.goal=='seek_mate'


def test_previous_intent_uses_past_executed_student_goal_not_teacher_advice(tmp_path):
    import gzip,json
    from experiments.train_lifecycle_model import load_corpus
    (tmp_path/'manifest.json').write_text(json.dumps({'splits':{'train':[11]}}))
    root=tmp_path/'seed-11';root.mkdir()
    rows=[{'seed':11,'split':'train','actor':'a','turn':t,'observation':observation(),
        'teacher_goal':'approach_meal','executed_goal':'wrap'} for t in (1,2)]
    def write():
        with gzip.open(root/'decisions.jsonl.gz','wt') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
    write();seq=load_corpus(tmp_path,'v2')['train'][0];x=seq['x'].clone()
    assert x[0,FEATURE_NAMES_V2.index('previous_intent_hunt')]==1
    assert x[1,FEATURE_NAMES_V2.index('previous_intent_wrap')]==1
    assert x[1,FEATURE_NAMES_V2.index('previous_intent_approach_meal')]==0
    rows[0]['teacher_goal']='court';rows[1]['teacher_goal']='defend';write()
    assert torch.equal(x,load_corpus(tmp_path,'v2')['train'][0]['x'])


def test_runtime_supplies_own_previous_selected_intent_to_v2_actor():
    from spider_sim.ecology import EcologyEncounter,SPIDER
    from spider_sim.djepa_ecology import senses
    class Capture(ForcedGoal):
        feature_version='v2'
        def step(self,x,hidden=None):
            self.seen=x.clone()
            return super().step(x,hidden)
    model=Capture();model.goal='feed'
    p=LearnedLifecyclePolicy(PursuitPolicy(PursuitModel(21)),model,'male');p.goal='wrap'
    obs=senses(EcologyEncounter(42,feeding_cycle=True,spider_pair=True),SPIDER)
    p.choose(obs)
    assert model.seen[0,0,FEATURE_NAMES_V2.index('previous_intent_wrap')]==1
    p.choose(obs)
    assert model.seen[0,0,FEATURE_NAMES_V2.index('previous_intent_feed')]==1


def test_recovery_collector_executes_student_command_not_advisor(tmp_path,monkeypatch):
    import gzip,json
    import experiments.collect_recovery_experience as collector
    from spider_sim.pursuit import save
    from spider_sim.lifecycle_model import GOALS,feature_names
    from spider_sim.ecology import SPIDER,FEMALE_SPIDER
    pursuit=tmp_path/'pursuit.pt';motor=PursuitModel(21);motor.value_kind='negative_action_logit'
    save(motor,pursuit)
    model=LifecycleModel(True,feature_version='v2')
    for p in model.parameters():p.data.zero_()
    model.actor[-1].bias.data[GOALS.index('wrap')]=10
    weights=tmp_path/'lifecycle.pt';torch.save({'state_dict':model.state_dict(),'recurrent':True,'features':feature_names('v2'),'goals':GOALS},weights)
    real=collector.LiveRun
    def controlled(*args,**kwargs):
        run=real(*args,**kwargs);w=run.world;rat='cave_rat_1'
        w.positions[SPIDER]=(103,140,'Fungal Stores');w.positions[FEMALE_SPIDER]=(103,136,'Fungal Stores')
        w.positions[rat]=(120,140,'Fungal Stores');w.config.characters[rat].character.current_hp=0;w.active.discard(rat)
        w.food[rat]={'wrap_turns':0,'remaining':.5}
        return run
    monkeypatch.setattr(collector,'LiveRun',controlled)
    collector.collect_one(tmp_path/'out',str(pursuit),str(weights),11,2)
    with gzip.open(tmp_path/'out/decisions.jsonl.gz','rt') as f:rows=[json.loads(line) for line in f]
    assert rows
    assert all(r['executed_goal']=='wrap' and r['teacher_goal']=='approach_meal' for r in rows)
    assert all(r['command']['action']=='wrap' and r['advisor_command']['action']=='move' for r in rows)
    assert all(r['result']['status']=='food_not_reachable' for r in rows)


def test_evaluation_resume_reuses_completed_case_and_checks_parameters(tmp_path,monkeypatch):
    import json
    import experiments.evaluate_learned_lifecycle as evaluation
    out=tmp_path/'case';out.mkdir();saved={'seed':11,'mode':'demos','turns':1000,'current_profile':True,'living_spiders':3}
    (out/'summary.json').write_text(json.dumps(saved))
    def forbidden(*args,**kwargs):raise AssertionError('Completed case should not run again')
    monkeypatch.setattr(evaluation,'LiveRun',forbidden)
    assert evaluation.run_one(out,'unused','unused',11,1000,'demos',True,True)=={**saved,'cached':True}
    with pytest.raises(ValueError,match='mismatch'):evaluation.run_one(out,'unused','unused',11,500,'demos',True,True)


def test_evaluation_resume_preserves_partial_files_before_restart(tmp_path,monkeypatch):
    import experiments.evaluate_learned_lifecycle as evaluation
    out=tmp_path/'case';out.mkdir();(out/'partial-log').write_text('preserve me')
    class StopProbe(Exception):pass
    def stop(*args,**kwargs):raise StopProbe()
    monkeypatch.setattr(evaluation,'LiveRun',stop)
    with pytest.raises(StopProbe):evaluation.run_one(out,'unused','unused',11,1000,'demos',True,True)
    assert (tmp_path/'case.partial-1/partial-log').read_text()=='preserve me'
    assert not out.exists()
