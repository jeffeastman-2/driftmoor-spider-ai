"""Watch the Level-2 simulation in Driftmoor's native dungeon spectator.

Starts paused. Play/Pause/Step advance the actual simulation; the map's own
slider and Play button only replay recorded history. No model provider calls.
"""
import argparse
import asyncio
from contextlib import suppress
import json
import shutil
import hashlib
from pathlib import Path
from aiohttp import web
from experiments.game_runtime import game_root

CONTROLS = """<span style='font-weight:normal;margin-left:1em'>
<button onclick="control('play')">Run / resume</button>
<button onclick="control('restart')">Restart from beginning</button>
<button onclick="control('pause')">Pause</button>
<button onclick="control('step')">One round</button>
<span id='simstate'>Paused · scripted baseline</span>
<label><input type='checkbox' id='allactors' onchange="location.search=this.checked?'?all=1':''">All creatures</label></span>
<script>
window.spectatorFocus=new URLSearchParams(location.search).has('all')?null:'freight_spider';
document.getElementById('allactors').checked=!window.spectatorFocus;
async function control(action){
 const r=await fetch('/simulation/'+action,{method:'POST'});
 const s=await r.json();document.getElementById('simstate').textContent=s.message||JSON.stringify(s);
 if(action==='restart' && r.ok)location.reload();
}
let episode=null;
async function status(){try{const r=await fetch('/simulation/status');const s=await r.json();
 if(episode!==null && episode!==s.episode){location.reload();return;}episode=s.episode;
 document.getElementById('simstate').textContent=s.message;}catch(e){document.getElementById('simstate').textContent='Disconnected';}}
setInterval(status,1000);
</script>"""


def screen_names(world):
    """Researcher-facing labels only; never supplied to creature policies."""
    special={'freight_spider':'Mr Spider' if world.spider_pair else 'Giant Spider','freight_spider_female':'Ms Spider','freight_shadowbeast':'Shadowbeast',
             'freight_master_goblin':'Freight Master'}
    return {cid:special.get(cid,cid.replace('_',' ').title()) for cid in world.positions}


def explanation(row,names=None,tracks=None):
    """Describe recorded inputs and rules, never invent a neural chain of thought."""
    names=names or {};tracks=tracks or {}
    label=lambda token:names.get(tracks.get(token,token),token.replace('_',' ').title())
    obs=row['observation'];result=row['result'];command=row['command']
    visual={c['track'] for c in obs['contacts'] if c['modality']=='vision'}
    silk={c['track'] for c in obs['contacts'] if c['modality']=='web'}
    own=obs['self']['hp_fraction']
    births=' '.join(f"{names.get(b['actor'],b['actor'])} spawned, replacing {names.get(b['replaces'],b['replaces'])}." for b in row.get('births',[]))
    activated=' '.join(f"{names.get(a['actor'],a['actor'])} became active ({a['reason']})." for a in result.get('activations',[]))
    activated=' '.join(part for part in (activated,births) if part)
    goal=row['goal']
    if row.get('controller_kind')=='pursuit':
        plan=row['plan'];target=plan.get('goal_track')
        text=(f"[Learned pursuit] {names.get(row['actor'],row['actor'])}: own health {own:.0%}; "
              f"{len(visual)} visual and {len(silk)} silk contacts. "
              +(f"Pursuing {label(target)} (memory age {plan['goal_age']} turns). " if target else 'No committed prey contact. ')
              +("No pursuit task is active; rest." if plan.get('selection_reason')=='no_pursuit_goal' else "Authored motor: "+plan['selection_reason']+"." if plan.get('selection_reason')!='learned_actor' else f"Neural actor chose {command['action']} from {len(plan['candidates'])} candidates."))
        if 'hunger' in obs['self']:text+=f" Goal {goal}; hunger {obs['self']['hunger']:.0%}."
        if command['action'] in ('wrap','feed','court'):
            text+=f" {goal.title()} {label(command['target'])}: {result['status']}."
            if 'food_remaining' in result:text+=f" Food remaining {result['food_remaining']:.2f} meal units."
        if command['action']=='vibrate_web':text+=' Sends a patterned courtship vibration through the web.'
        if goal=='seek_mate':text+=' Searching for a receptive mate using local contacts.'
        if command['action']=='move':
            text+=f" Travel direction {command['heading_deg']:.1f} degrees; moved {result['distance_ft']:.1f} ft ({result['status']})."
            if 'body_heading_deg' in result:text+=f" Body heading {result['body_heading_deg']:.1f} degrees; "+(result['turn_reason'].replace('_',' ')+'.' if result.get('turn_reason') else 'facing retained.')
        elif command['action']=='attack':text+=f" Target {label(command['target'])}; engine {result['status']}; damage {result['damage']} HP."
        else:text+=f" Engine: {result['status']}."
        if result.get('target_defeated'):text+=' The target was defeated.'
        return text+(' '+activated if activated else '')
    if row.get('controller_kind')=='djepa':
        plan=row['plan']
        text=(f"[dJEPA] {names.get(row['actor'],row['actor'])}: goal {goal}; own health {own:.0%}. "
              f"Sensed {len(visual)} visual and {len(silk)} silk contacts. Chose {command['action']} "
              f"from {len(plan['candidates'])} candidates. Forecast: injury {plan['predicted_injury_hp']:.1f} HP")
        if command['action']=='attack':
            text+=f"; bite/strike permission {plan['predicted_admission']:.0%}; expected damage {plan['predicted_damage_hp']:.1f} HP. Target {label(command['target'])}; actual {result['status']}, damage {result['damage']} HP."
        elif command['action']=='move':
            text+=f". Heading {command['heading_deg']:.0f} degrees; actual movement {result['distance_ft']:.1f} ft ({result['status']})."
        else:text+=f". Engine: {result['status']}."
        if result.get('target_defeated'):text+=' The target was defeated.'
        return text+(' '+activated if activated else '')
    reason={'survive':'Health is critically low; withdraw from the nearest sensed contact.',
            'feed':'Appetite is high and a contact is available; pursue or bite.',
            'maintain_web':'No feeding contact takes priority; inspect or repair damaged local silk.',
            'rest':'No urgent sensed target or local repair; remain still.',
            'forage':'Wander, or move away from a visibly larger creature.'}[goal]
    text=(f"[Scripted baseline] {names.get(row['actor'],row['actor'].replace('_',' ').title())}: goal {goal}; own health {own:.0%}. "
          f"Sensed {len(visual)} visual and {len(silk)} silk contacts. {reason} "
          f"Chose {command['action']}.")
    if command['action']=='attack':
        text+=f" Target {label(command['target'])}; engine: {result['status']}; damage {result['damage']} HP."
        if result.get('target_defeated'):
            text+=' The target was defeated.'
    elif command['action']=='move':
        if goal=='feed' and obs['contacts']:
            contacts={c['track']:c for c in obs['contacts'] if c['modality']!='vision'}
            contacts.update({c['track']:c for c in obs['contacts'] if c['modality']=='vision'})
            nearest=min(contacts.values(),key=lambda c:c['range_ft'])
            text+=f" Pursuing {label(nearest['track'])}."
        text+=f" Heading {command['heading_deg']:.0f} degrees; moved {result['distance_ft']:.1f} ft ({result['status']})."
    else:
        text+=f" Engine: {result['status']}."
    return text+(' '+activated if activated else '')


class LiveRun:
    def __init__(self,output,seed=42,rounds=120,*,resume=False,model_dir=None,selection="predictive",spider_model=None,spider_facing="track",spider_local_motion=False,replenish_prey=False,feeding_cycle=False,random_prey=False,spider_pair=False):
        from spider_sim.ecology import EcologyEncounter,baseline_policies
        if model_dir and spider_model:raise ValueError('Choose species models or a single pursuit spider model')
        if feeding_cycle and not spider_model:raise ValueError("Feeding cycle requires a pursuit spider model")
        if spider_pair and not feeding_cycle:raise ValueError("Spider pair requires feeding cycle")
        self.random_prey=random_prey;self.spider_pair=spider_pair
        self.feeding_cycle=feeding_cycle
        self.spider_local_motion=spider_local_motion
        self.replenish_prey=replenish_prey
        self.spider_facing=spider_facing
        self.spider_model=Path(spider_model).resolve() if spider_model else None
        self.selection=selection
        self.model_dir=Path(model_dir).resolve() if model_dir else None
        self.controller_signature=({'mode':'djepa','selection':selection,'weights':{species:hashlib.sha256((self.model_dir/(species+'.pt')).read_bytes()).hexdigest()
                                    for species in ('rat','spider','shadowbeast')}} if self.model_dir else {'mode':'scripted'})
        self.mode=f'dJEPA ({selection}): rats, spider and Shadowbeast' if self.model_dir else 'scripted baseline'
        if self.spider_model:
            self.controller_signature={'mode':'learned_pursuit','approach':True,'facing':spider_facing,'rest_without_goal':True,'spider_sha256':hashlib.sha256(self.spider_model.read_bytes()).hexdigest()}
            self.mode='learned pursuit spider · goal-directed facing · rest without prey · other creatures scripted' if spider_facing=='track' else 'learned pursuit spider · legacy facing · rest without prey · other creatures scripted'
        if self.spider_model and spider_local_motion:
            self.controller_signature['local_motion']='roam-and-feedback-detour-v1'
            self.mode='learned pursuit spider · local recovery and roaming · other creatures scripted'
        if replenish_prey:
            self.controller_signature['replenish_prey']=True
            self.mode+=' · rats/crickets replaced'
        if feeding_cycle:
            self.controller_signature['feeding_cycle']='hunt-wrap-feed-seek-mate-v2'
            self.mode+=' · hunt/wrap/feed/seek mate'
        if random_prey:self.controller_signature['random_prey']='valid-floor-v1';self.mode+=' · random prey spawns'
        if spider_pair:self.controller_signature['spider_pair']='fed-courtship-web-signals-v1';self.mode+=' · Mr & Ms Spider'
        self.seed=seed;self.episode=1
        self.world=EcologyEncounter(seed,replenish_prey=self.replenish_prey,feeding_cycle=self.feeding_cycle,random_prey=self.random_prey,spider_pair=self.spider_pair)
        self.policies=self.make_policies(self.world)
        self.output=Path(output)
        if self.output.exists() and not resume:
            raise ValueError('Use a new output directory or --resume')
        self.output.mkdir(parents=True,exist_ok=resume)
        self.round=0;self.limit=rounds;self.running=False;self.error=None
        records=[]
        if resume:
            # Replay with matching weights and mode; validate before touching files.
            snapshots=[json.loads(p.read_text()) for p in self.output.glob('day1_*.json')]
            if not snapshots:raise ValueError('No episode checkpoints to resume')
            saved=max(snapshots,key=lambda row:row['turn'])
            if saved.get('seed')!=seed:raise ValueError('Resume seed mismatch')
            if saved.get('controller_signature',{'mode':'scripted'})!=self.controller_signature:
                raise ValueError('Resume controller/checkpoint mismatch')
            from spider_sim.ecology import ecology_round
            for turn in range(saved['turn']):records=ecology_round(self.world,self.policies,turn)
            positions={cid:{'x':p[0],'y':p[1],'zone':p[2]} for cid,p in self.world.positions.items()}
            hp={cid:cs.character.current_hp for cid,cs in self.world.config.characters.items()}
            webs={key:item.state for key,item in self.world.config.items.items() if 'web_bounds' in item.state}
            if (positions!=saved['current_positions'] or hp!=saved['ecology_hp'] or
                webs!=saved['ecology_web_state'] or records!=saved['ecology_decisions'] or
                (self.feeding_cycle and (self.world.food!=saved.get('ecology_food') or self.world.hunger!=saved.get('ecology_hunger') or self.world.mate_cooldown!=saved.get('ecology_mate_cooldown',{}) or self.world.mating_events!=saved.get('ecology_mating_events',[]) or self.world.web_signals!=saved.get('ecology_web_signals',{}) or {':'.join(k):v for k,v in self.world.courtship.items()}!=saved.get('ecology_courtship',{})))):
                raise ValueError('Replay differs from saved episode; resume refused without changing checkpoints')
            self.round=saved['turn'];self.episode=saved.get('episode',1)
        self.publish(records)

    def make_policies(self,world):
        if self.spider_model:
            from spider_sim.ecology import baseline_policies,SPIDER
            from spider_sim.pursuit import load,PursuitPolicy
            model=load(self.spider_model)
            if model.value_kind!='negative_action_logit':raise ValueError('Expected the improved imitation pursuit checkpoint')
            result=baseline_policies(world,self.seed)
            result[SPIDER]=PursuitPolicy(model,approach=True,facing=self.spider_facing,rest_without_goal=True,local_motion=self.spider_local_motion,seed=self.seed)
            if self.feeding_cycle:
                from spider_sim.feeding import FeedingPolicy
                result[SPIDER]=FeedingPolicy(result[SPIDER])
                if self.spider_pair:
                    from spider_sim.ecology import FEMALE_SPIDER
                    result[FEMALE_SPIDER]=FeedingPolicy(PursuitPolicy(model,approach=True,facing=self.spider_facing,rest_without_goal=True,local_motion=True,seed=self.seed+1))
            return result
        if self.model_dir:
            from spider_sim.djepa_ecology import policies
            return policies(world,self.model_dir,self.seed,self.selection)
        from spider_sim.ecology import baseline_policies
        return baseline_policies(world,self.seed)

    def status(self):
        mode='Failed' if self.error else 'Complete' if self.round>=self.limit else 'Running' if self.running else 'Paused'
        return {'episode':self.episode,'seed':self.seed,'round':self.round,'limit':self.limit,'running':self.running,'error':self.error,
                'message':f'{mode} · episode {self.episode} · round {self.round}/{self.limit} · {self.mode}'+(f' · {self.error}' if self.error else '')}

    def restart(self):
        """Reset world, RNG and policy memory, preserving the previous episode."""
        from spider_sim.ecology import EcologyEncounter,baseline_policies
        world=EcologyEncounter(self.seed,replenish_prey=self.replenish_prey,feeding_cycle=self.feeding_cycle,random_prey=self.random_prey,spider_pair=self.spider_pair)
        policies=self.make_policies(world)
        archive=self.output/'previous_episodes'/f'episode-{self.episode:04d}'
        archive.mkdir(parents=True,exist_ok=False)
        self.running=False
        for path in list(self.output.glob('day1_*.json'))+[self.output/'echoing_caverns_level2_map.html']:
            if path.exists():shutil.move(str(path),archive/path.name)
        self.world=world;self.policies=policies
        self.round=0;self.episode+=1;self.error=None
        self.publish([])

    def step(self):
        from spider_sim.ecology import ecology_round
        if self.round>=self.limit or self.error:
            return
        records=ecology_round(self.world,self.policies,self.round)
        self.round+=1
        self.publish(records)
        if self.round>=self.limit:
            self.running=False

    def publish(self,records):
        from engine.map_renderer import render_turn_slider_html
        world=self.world
        names=screen_names(world)
        snapshot={'ecology_web_signals':world.web_signals,'ecology_courtship':{':'.join(k):v for k,v in world.courtship.items()},'ecology_mate_cooldown':world.mate_cooldown,'ecology_mating_events':world.mating_events,'ecology_food':world.food,'ecology_hunger':world.hunger,'ecology_births':world.last_births,'controller_signature':self.controller_signature,'episode':self.episode,'seed':self.seed,'turn':self.round,'scene_place_path':world.geometry.place_path,
            'active_characters':sorted(world.active),
            'current_positions':{cid:{'x':p[0],'y':p[1],'zone':p[2]} for cid,p in world.positions.items()},
            'character_results':[{'character_id':r['actor'],'character_name':names[r['actor']],
                'narration':{'prose':explanation(r,names,{token:cid for cid,token in world.tokens[r['actor']].items()})}} for r in records],
            'ecology_decisions':records,
            'ecology_hp':{cid:cs.character.current_hp for cid,cs in world.config.characters.items()},
            'ecology_web_state':{key:item.state for key,item in world.config.items.items() if 'web_bounds' in item.state}}
        path=self.output/f'day1_{self.round:05d}.json'
        temp=path.with_suffix('.tmp');temp.write_text(json.dumps(snapshot,indent=2)+'\n');temp.replace(path)
        path=self.output/'echoing_caverns_level2_map.html';temp=path.with_suffix('.tmp')
        # Observer map: all placed residents are visible to the researcher, not to policies.
        render_turn_slider_html(world.config.geometry_path,temp,snapshot_dir=self.output,
            title='Level-2 ecology · observer map · '+self.mode,live_refresh_secs=2,
            start_characters=set(world.positions),item_details=world.config.items,
            start_positions={cid:p[:2] for cid,p in world.positions.items()},scene_geometry=world.geometry,
            visited_places={world.geometry.place_path},
            char_colors={'freight_spider':'#e64b28','freight_spider_female':'#9c4dcc'},
            character_symbols={cid:('🕷' if cid in world.spiders else '🐀' if 'rat_' in cid else '🦗' if 'cricket_' in cid
                                    else '🦎' if 'lizard_' in cid else '◆') for cid in world.positions})
        temp.replace(path)


def create_app(run,interval=2.):
    from web_spectator import create_app as spectator_app
    app=spectator_app(run.output,controls_html=CONTROLS)
    async def status(request):
        return web.json_response(run.status())
    async def control(request):
        action=request.match_info['action']
        if action=='play':
            run.running=run.round<run.limit and not run.error
        elif action=='pause':run.running=False
        elif action=='restart':run.restart()
        elif action=='step':
            run.running=False
            try:run.step()
            except Exception as exc:
                run.error=str(exc);raise
        else:raise web.HTTPNotFound()
        return web.json_response(run.status())
    async def refresh(request):
        from models.geometry_loader import load_geometry
        from models.space_time import Position
        geometry=load_geometry(run.world.config.geometry_path)
        if any(not geometry.zone_at(Position(*p[:2])) for p in run.world.positions.values()):
            raise web.HTTPBadRequest(text="Edited geometry excludes an existing resident")
        geometry.starting_positions=run.world.geometry.starting_positions
        run.world.geometry=geometry
        run.world.config.scene_geometry=geometry
        current=run.output/f'day1_{run.round:05d}.json'
        records=json.loads(current.read_text()).get('ecology_decisions',[])
        run.publish(records)
        return web.json_response({'refreshed':True})
    async def loop():
        while True:
            await asyncio.sleep(interval)
            if run.running:
                try:run.step()
                except Exception as exc:
                    run.error=str(exc);run.running=False
    async def lifecycle(app):
        task=asyncio.create_task(loop())
        yield
        task.cancel()
        with suppress(asyncio.CancelledError):await task
    app.add_routes([web.get('/simulation/status',status),web.post('/simulation/{action}',control),
                    web.post('/refresh-geometry',refresh)])
    app.cleanup_ctx.append(lifecycle)
    return app


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--driftmoor-root',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--rounds',type=int,default=120)
    parser.add_argument('--interval',type=float,default=2.)
    parser.add_argument('--port',type=int,default=8081)
    parser.add_argument('--selection',choices=['predictive','proposal'],default='predictive',help='Neural forecast scoring or proposal-only diagnostic')
    parser.add_argument('--random-prey',action='store_true',help='Spawn replacement prey in random valid zones and floor positions')
    parser.add_argument('--spider-pair',action='store_true',help='Add Ms Spider with independent hunger and fed-only courtship')
    parser.add_argument('--feeding-cycle',action='store_true',help='Authored hunt/wrap/feed/seek mate spider goals')
    parser.add_argument('--spider-local-motion',action='store_true',help='Authored gentle roaming and local blocked-movement recovery')
    parser.add_argument('--replenish-prey',action='store_true',help='Replace each dead rat/cricket with one newborn at its authored home')
    parser.add_argument('--spider-facing',choices=['track','legacy'],default='track',help='Separate pursuit travel direction and body heading, or reproduce legacy coupling')
    parser.add_argument('--spider-model',type=Path,help='Improved learned pursuit checkpoint for spider only; other residents scripted')
    parser.add_argument('--models',type=Path,help='Directory containing rat.pt, spider.pt and shadowbeast.pt; default remains scripted')
    parser.add_argument('--resume',action='store_true',help='Validate deterministic replay of the existing episode, then pause at its last round')
    args=parser.parse_args()
    if not 1<=args.rounds<=1000 or not .1<=args.interval<=60:
        parser.error('Use 1..1000 rounds and a finite interval from .1 to 60 seconds')
    if args.models and args.spider_model:parser.error('Choose --models or --spider-model, not both')
    game_root(args.driftmoor_root)
    run=LiveRun(args.output,args.seed,args.rounds,resume=args.resume,model_dir=args.models,selection=args.selection,spider_model=args.spider_model,spider_facing=args.spider_facing,spider_local_motion=args.spider_local_motion,replenish_prey=args.replenish_prey,feeding_cycle=args.feeding_cycle,random_prey=args.random_prey,spider_pair=args.spider_pair)
    print(f'Open http://localhost:{args.port}/ — starts paused; {run.mode}')
    web.run_app(create_app(run,args.interval),host='127.0.0.1',port=args.port)

if __name__=='__main__':main()
