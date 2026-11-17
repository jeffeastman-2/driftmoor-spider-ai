import asyncio
import json
from aiohttp.test_utils import TestServer,TestClient
from experiments.watch_level2 import LiveRun,create_app,explanation
from spider_sim.ecology import game_root,SPIDER


def test_live_control_steps_real_world_and_publishes_native_checkpoints(tmp_path):
    async def check():
        run=LiveRun(tmp_path/'live',42,3)
        client=TestClient(TestServer(create_app(run,interval=.01)))
        await client.start_server()
        try:
            page=await (await client.get('/')).text()
            assert 'Run / resume' in page and 'Restart from beginning' in page and 'One round' in page and '__CONTROLS__' not in page
            state=await (await client.get('/simulation/status')).json()
            assert state['round']==0 and not state['running']
            state=await (await client.post('/simulation/step')).json()
            assert state['round']==1 and not state['running']
            snapshot=json.loads((run.output/'day1_00001.json').read_text())
            assert snapshot['ecology_decisions']
            assert all('[Scripted baseline]' in row['narration']['prose'] for row in snapshot['character_results'])
            page=await (await client.get('/map')).text()
            assert 'web-sheet' in page and 'FRAMES=' in page and 'current silk HP' in page
            assert (await client.post('/refresh-geometry')).status==200
            assert json.loads((run.output/'day1_00001.json').read_text())['ecology_decisions']==snapshot['ecology_decisions']
            await client.post('/simulation/play')
            for _ in range(100):
                state=await (await client.get('/simulation/status')).json()
                if state['round']==3:break
                await asyncio.sleep(.01)
            assert state['round']==3 and not state['running']
            assert (await client.post('/simulation/unknown')).status==404
            await client.post('/simulation/step')
            assert run.round==3
            old=json.loads((run.output/'day1_00001.json').read_text())
            state=await (await client.post('/simulation/restart')).json()
            assert state['round']==0 and state['episode']==2 and not state['running']
            assert not (run.output/'day1_00003.json').exists()
            assert (run.output/'previous_episodes/episode-0001/day1_00003.json').exists()
            assert not json.loads((run.output/'day1_00000.json').read_text())['ecology_decisions']
            await client.post('/simulation/step')
            new=json.loads((run.output/'day1_00001.json').read_text())
            assert new['ecology_decisions']==old['ecology_decisions']
        finally:
            await client.close()
    asyncio.run(check())


def test_explanation_describes_recorded_evidence_not_neural_reasoning(tmp_path):
    run=LiveRun(tmp_path/'live',42,1);run.step()
    snapshot=json.loads((run.output/'day1_00001.json').read_text())
    row=next(r for r in snapshot['ecology_decisions'] if r['actor']==SPIDER)
    text=explanation(row)
    assert 'goal feed' in text and 'own health' in text and 'silk contacts' in text
    assert 'Chose '+row['command']['action'] in text
    assert 'predicted' not in text.lower()


def test_restart_restores_creatures_webs_random_state_and_policy_memory(tmp_path):
    from engine.webs import damage_web
    run=LiveRun(tmp_path/'live',42,3)
    initial_positions=dict(run.world.positions)
    initial_observation=run.world.observation_for(SPIDER)
    run.step()
    run.world.config.characters[SPIDER].character.current_hp=1
    damage_web(run.world.config.items['store_web_west'],10)
    run.policies[SPIDER].goal='survive'
    run.running=True;run.error='old error'
    run.restart()
    assert run.world.positions==initial_positions
    assert run.world.observation_for(SPIDER)==initial_observation
    assert run.policies[SPIDER].goal=='rest' and run.policies[SPIDER].goal_turns==0
    assert not run.running and run.error is None


def test_creature_names_are_observer_annotations_not_policy_inputs(tmp_path):
    from experiments.watch_level2 import screen_names
    run=LiveRun(tmp_path/'live',42,1);run.step()
    snapshot=json.loads((run.output/'day1_00001.json').read_text())
    spider=next(c for c in snapshot['character_results'] if c['character_id']==SPIDER)
    assert spider['character_name']=='Giant Spider'
    assert 'freight_spider' not in spider['narration']['prose']
    row=next(r for r in snapshot['ecology_decisions'] if r['actor']==SPIDER)
    assert 'Cave Cricket 1' in spider['narration']['prose']
    assert 'Cave Cricket 1' not in json.dumps(row['observation'])
    token=run.world.tokens[SPIDER]['cave_rat_1']
    row['command']={'action':'attack','target':token}
    row['result']={'status':'hit','damage':4,'target_defeated':True}
    tracks={token:cid for cid,token in run.world.tokens[SPIDER].items()}
    prose=explanation(row,screen_names(run.world),tracks)
    assert 'Target Cave Rat 1' in prose and 'contact_' not in prose


def test_resume_preserves_the_current_encounter_and_refuses_divergent_replay(tmp_path):
    import pytest
    path=tmp_path/'live'
    original=LiveRun(path,42,4);original.step();original.step()
    resumed=LiveRun(path,42,4,resume=True)
    assert resumed.round==2 and not resumed.running
    assert resumed.world.positions==original.world.positions
    assert resumed.world.last_results==original.world.last_results
    assert resumed.world.random.getstate()==original.world.random.getstate()
    original.step();resumed.step()
    assert original.world.positions==resumed.world.positions
    checkpoint=path/'day1_00003.json'
    data=json.loads(checkpoint.read_text());data['ecology_hp'][SPIDER]=0
    checkpoint.write_text(json.dumps(data));before=checkpoint.read_bytes()
    with pytest.raises(ValueError,match='Replay differs'):
        LiveRun(path,42,4,resume=True)
    assert checkpoint.read_bytes()==before


def test_improved_spider_only_mode_annotations_memory_and_resume(tmp_path):
    import torch
    from spider_sim.pursuit import PursuitModel,save
    model=PursuitModel(21);model.value_kind='negative_action_logit'
    path=tmp_path/'spider.pt';save(model,path)
    run=LiveRun(tmp_path/'pursuit',42,3,spider_model=path)
    assert run.policies[SPIDER].approach
    assert all(not getattr(p,'neural',False) for cid,p in run.policies.items() if cid!=SPIDER)
    run.step()
    row=json.loads((run.output/'day1_00001.json').read_text())
    spider=next(r for r in row['ecology_decisions'] if r['actor']==SPIDER)
    assert spider['controller_kind']=='pursuit' and spider['plan']['value_kind']=='negative_action_logit'
    text=next(r for r in row['character_results'] if r['character_id']==SPIDER)['narration']['prose']
    assert '[Learned pursuit]' in text and 'Forecast' not in text
    assert 'other creatures scripted' in run.status()['message']
    restored=LiveRun(run.output,42,3,spider_model=path,resume=True)
    assert restored.round==1
    restored.restart()
    assert restored.policies[SPIDER].hidden is None and restored.policies[SPIDER].task.track is None
