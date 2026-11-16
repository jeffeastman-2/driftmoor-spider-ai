import copy
import json
import pytest
import torch
from spider_sim.ecology import EcologyEncounter, ecology_round
from spider_sim.djepa_ecology import (EcologyModel, NeuralPolicy, SPECIES, senses, encode,
    observation_tensors, save, load, policies, costs, candidates)

torch.set_num_threads(1)

def test_contact_order_and_track_names_do_not_change_latent():
    obs=senses(EcologyEncounter(42),'freight_spider')
    alternate=copy.deepcopy(obs)
    alternate['contacts'].reverse()
    for row in alternate['contacts']:row['track']='renamed-'+row['track']
    model=EcologyModel().eval()
    a,_=model.represent(*observation_tensors([obs],'spider'))
    b,_=model.represent(*observation_tensors([alternate],'spider'))
    assert torch.allclose(a,b,atol=1e-6)
    assert len(encode(obs,'spider')[0])==24


def test_species_weights_shared_but_individual_memories(tmp_path):
    for species in SPECIES:save(EcologyModel(),tmp_path/(species+'.pt'),species)
    world=EcologyEncounter(42);group=policies(world,tmp_path)
    rat1=group['cave_rat_1'];rat2=group['cave_rat_2']
    assert rat1.model is rat2.model
    assert rat1.model is not group['freight_spider'].model
    rat1.choose(senses(world,'cave_rat_1'))
    assert rat1.hidden.shape==(1,1,48) and rat2.hidden is None
    assert world.observation_for('cave_rat_1')['schema']!='ecology-djepa-v1'
    assert all(0<=v<=10.01 for v in senses(world,'cave_rat_1')['self']['local_clearance_ft'])
    with pytest.raises(ValueError):load(tmp_path/'rat.pt','spider')


def test_impossible_damage_is_suppressed_by_predicted_permission():
    obs=senses(EcologyEncounter(42),'freight_spider');choices=candidates(obs)
    attack=next(i for i,c in enumerate(choices) if c['action']=='attack')
    prediction=torch.zeros(len(choices),7);prediction[:,3]=-100
    prediction[attack,4]=10
    assert costs(obs,'spider',choices,prediction)[attack].item()==pytest.approx(0,abs=1e-6)
    prediction[attack,3]=100
    assert costs(obs,'spider',choices,prediction)[attack]<-100


def test_neural_viewer_resumes_and_rejects_changed_weights(tmp_path):
    from experiments.watch_level2 import LiveRun
    models=tmp_path/'models';models.mkdir()
    for species in SPECIES:save(EcologyModel(),models/(species+'.pt'),species)
    run=LiveRun(tmp_path/'live',42,3,model_dir=models);run.step()
    saved=json.loads((run.output/'day1_00001.json').read_text())
    rows=[r for r in saved['ecology_decisions'] if r.get('controller_kind')=='djepa']
    assert {r['actor'] for r in rows}=={'cave_rat_1','cave_rat_2','freight_spider','freight_shadowbeast'}
    assert all('Forecast:' in c['narration']['prose'] for c in saved['character_results'] if c['character_id'] in {r['actor'] for r in rows})
    restored=LiveRun(run.output,42,3,resume=True,model_dir=models)
    assert restored.round==1
    restored.restart()
    assert restored.policies['freight_spider'].hidden is None
    with pytest.raises(ValueError,match='controller/checkpoint'):LiveRun(run.output,42,3,resume=True)
    save(EcologyModel(),models/'spider.pt','spider')
    with pytest.raises(ValueError,match='controller/checkpoint'):LiveRun(run.output,42,3,resume=True,model_dir=models)


def test_clearance_candidate_feature_is_private_and_checkpoint_specific(tmp_path):
    from spider_sim.djepa_ecology import candidate_values,CANDIDATE_SIZE
    obs=senses(EcologyEncounter(42),'cave_rat_1')
    obs['self']['local_clearance_ft']=[0.,1.,2.,3.,4.,5.,6.,7.]
    choices=candidates(obs)
    forward=choices[1];reverse=choices[9]
    assert candidate_values(forward,obs,CANDIDATE_SIZE+1)[-1]==0
    assert candidate_values(reverse,obs,CANDIDATE_SIZE+1)[-1]==.4
    # Zero clearance is information, not a hard feasibility mask.
    assert forward in choices
    path=tmp_path/'rat.pt';save(EcologyModel(CANDIDATE_SIZE+1),path,'rat')
    model=load(path,'rat');assert model.candidate_size==19
    policy=NeuralPolicy(model,'rat');assert policy.choose(obs) in choices
    save(EcologyModel(),path,'rat');assert load(path,'rat').candidate_size==18
    with pytest.raises(ValueError):candidate_values(forward,obs,20)


def test_curriculum_has_close_far_starts_and_wall_labels_use_native_geometry():
    from experiments.train_ecology_djepa import training_world,FOCUS
    from experiments.evaluate_ecology_isolated import facing_wall
    near=[];wall_labels=[]
    for seed in range(100000,100004):
        world=training_world('spider',seed);obs=senses(world,FOCUS['spider'])
        assert len(world.active)==2
        assert len(obs['contacts'])>=1
        near.append(min(c['range_ft'] for c in obs['contacts']))
        if seed%2==0:wall_labels.append(facing_wall(world,FOCUS['spider']))
    assert min(near)<=5 and max(near)>=10
    # Sampling-box edges are not necessarily dungeon walls.
    assert wall_labels==[True,False]
