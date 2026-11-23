/* Embedded synthetic records for the original interface. No network or model execution. */
(function (root) {
  'use strict';

  const TASKS = ['extraction', 'registration', 'segmentation', 'classification'];
  const INITIAL_STEPS = 256;
  const STEPS_PER_EPOCH = 16;
  const EXPERIMENT_EPOCHS = 20;
  const MAX_STEPS = STEPS_PER_EPOCH * EXPERIMENT_EPOCHS;
  const ORDER = ['denoising', 'bias', 'extraction', 'normalization', 'registration', 'parcellation', 'graph', 'segmentation', 'classification', 'surface'];
  const MODULES = {denoising:['denoise_net'], bias:['bias_net'], extraction:['ext_net'], normalization:['norm_net'], registration:['reg_net'], segmentation:['seg_net'], parcellation:['par_net'], graph:['feature_net','graph_gen'], classification:['gnn_pred'], surface:['surface_net']};
  const PROVENANCE = {mode:'simulated', synthetic:true, source:'Embedded deterministic illustration', model_executed:false, llm_called:false, measured:false};
  const clone = value => JSON.parse(JSON.stringify(value));
  const round = value => Number(value.toFixed(6));
  const mapTasks = fn => Object.fromEntries(TASKS.map((task, i) => [task, fn(task, i)]));

  function createDemoBackend(initialConfig, clock = () => Date.now()) {
    let atlas;
    let config, started = clock(), serial = 0, running = null, latest = null;
    const builds = new Map(), jobs = new Map();
    const stamp = value => new Date(value ?? clock()).toISOString();
    const response = (data, status = 200) => new Response(JSON.stringify({...data, provenance:PROVENANCE}), {status, headers:{'Content-Type':'application/json', 'Cache-Control':'no-store'}});
    const defaults = () => clone(config.catalog.defaults);
    const isOriginal = selection => config.catalog.stages.every(stage => selection[stage.id] === config.catalog.defaults[stage.id]);
    const activeModules = selection => ORDER.filter(id => selection[id] && selection[id] !== 'none').flatMap(id => MODULES[id]);

    function configure(value) {
      if (!value?.catalog?.stages || !value.catalog.defaults) throw new Error('The demo requires the bundled UniBrain catalog.');
      config = clone(value);
      atlas = typeof document === 'undefined' ? require('./mni_preview.json') :
        JSON.parse(document.getElementById('demoAtlas').textContent);
      api.active = true;
      return api;
    }

    function validateSelection(value) {
      if (!value || Object.keys(value).length !== config.catalog.stages.length) throw new Error('Select one catalog option for every stage.');
      config.catalog.stages.forEach(stage => {
        if (!stage.options.some(option => option.id === value[stage.id])) throw new Error('Unknown option for ' + stage.id);
      });
      if ((value.extraction === 'M03') !== (value.registration === 'M03')) throw new Error('ERNet requires extraction and registration together.');
      if (value.extraction === 'M03' && value.normalization !== 'none') throw new Error('The ERNet pair requires normalization to be skipped.');
      if (value.surface === 'M15') throw new Error('Cortical surface requires mesh inputs that this synthetic demonstration does not contain.');
      return clone(value);
    }

    // Indexed noise keeps replay, live polling and the objective on the same trace.
    function noise(step, channel) {
      let bits = Math.imul(step + 1, 0x85ebca6b) ^ Math.imul(channel + 1, 0xc2b2ae35);
      bits = Math.imul(bits ^ (bits >>> 16), 0x27d4eb2d);
      return ((bits ^ (bits >>> 15)) >>> 0) / 2147483648 - 1;
    }

    function losses(step) {
      const difficulty = noise(step, 0);
      return mapTasks((_, i) => {
        const trend = [.08,.15,.22,.38][i] + [.55,.22,1.1,.40][i] * Math.exp(-step / [18,30,23,50][i]);
        const variation = .26 * difficulty + [.28,.14,.33,.82][i] * noise(step, i + 1);
        const surge = difficulty > .7 ? [.12,.06,.26,.48][i] : 0;
        return round(trend * Math.exp(variation) + surge);
      });
    }

    function policy(step) {
      const batchLoss = losses(step);
      const vectors = TASKS.map((task,i) => TASKS.map((_,j) =>
        Math.sqrt(batchLoss[task]) * ((i === j ? 1 : 0) + .86 * noise(step, 12 + 4*i + j))));
      const scale = 1 + .08 * Math.sin(step / 7);
      const gram = vectors.map(a => vectors.map(b => round(a.reduce((s, v, k) => s + v * b[k], 0) * scale)));
      const gradient_norms = mapTasks((_, i) => Math.sqrt(gram[i][i]));
      const exploration = .09 + .91 * Math.exp(-step / 67);
      const action_logits = mapTasks((task, i) =>
        [.18,-.15,.09,-.12][i] + exploration * 1.55 * Math.tanh(
          .6 * Math.sin(step / 3 + i) + .8 * noise(step, 32+i) + .5 * batchLoss[task]));
      const denominator = Object.values(action_logits).reduce((s, x) => s + Math.exp(x), 0);
      const actions = mapTasks(task => Math.exp(action_logits[task]) / denominator);
      const accepted = round(.68 + .30 * noise(step, 40));
      const weights = mapTasks((task, i) => round([1,.1,1,1][i] *
        (1 + accepted * .15 * Math.max(-1, Math.min(1, 4 * actions[task] - 1)))));
      const improvement = round(.012 + .12 * noise(step, 41));
      return {profile:'unibrain_anchored', gradient_normalization:'none', reward_scope:'other_training_subject_before_after_update',
        trust_radius:.15, accepted_action_fraction:accepted, gradient_norms, gram,
        cosine_similarity:gram.map((row,i) => row.map((value,j) => value / Math.sqrt(gram[i][i] * gram[j][j]))),
        observations:mapTasks((task,i) => [...gram[i].map(value => round(value / (gram[i][i] + 1e-8))), round(Math.log1p(losses(step)[task]))]),
        task_losses:losses(step), action_logits, actions, weights,
        critic_td_losses:mapTasks((_,i) => round(.08 / (1 + step / 10) + .002 * i)),
        policy_updates:Math.max(0, step - 4), actor_optimizer_steps:Math.max(0, step - 4), critic_optimizer_steps:Math.max(0, step - 4),
        replay_size:step, reward:round(improvement - .005),
        reward_terms:{training_loss_improvement:improvement, gradient_norm_penalty:.005},
        action_bound:2, provenance:PROVENANCE};
    }

    function objective(selection, coordination = 'single', step = null) {
      const packet = clone(config.objective || {});
      const base = [['extraction','Brain extraction','BCE','Subject brain mask',1], ['registration','Image registration','GCC','Fixed MRI template',.1], ['segmentation','Tissue segmentation','CE','Tissue reference',1], ['classification','Classification','CE','Subject class label',1]];
      packet.terms = base.map(([id,name,definition,supervision,weight]) => ({id,name,definition,supervision,weight}));
      if (['M06','M07'].includes(selection.registration)) packet.terms.push({id:'regularization', name:'Deformation regularization', definition:'Deformation smoothness', supervision:'Deformation field', weight:.001, weight_group:'registration', relative_weight:.01});
      if (['M09','M10'].includes(selection.parcellation)) packet.terms.push({id:'parcellation', name:'AAL parcellation', definition:'CE', supervision:'Subject AAL reference', weight:.2, weight_group:'segmentation', relative_weight:.2});
      if (selection.segmentation === 'M08') packet.terms.find(term => term.id === 'segmentation').definition = 'CE + 0.5 CE per auxiliary head';
      packet.stages = ORDER.filter(id => selection[id] && selection[id] !== 'none').map(id => {
        const stage = config.catalog.stages.find(item => item.id === id), option = stage.options.find(item => item.id === selection[id]);
        const loss_ids = packet.terms.filter(term => (term.id === 'regularization' ? 'registration' : term.id) === id).map(term => term.id);
        return {id, name:stage.name, method:option.method || option.label, loss_ids, branch:['segmentation','surface'].includes(id), note:loss_ids.length ? '' : 'N/A: no independent loss.'};
      });
      const keys = activeModules(selection);
      Object.assign(packet, {version:1, selection:clone(selection), coordination, profile:'Simulated UniBrain illustration',
        status:step === null ? 'declared' : 'observed', step, modules:keys.map(id => ({id,name:packet.stages.find(stage => MODULES[stage.id].includes(id)).method+' ('+id+')'})), connections:[],
        unassigned_loss_ids:[], provenance:PROVENANCE,
        connection_evidence:'Synthetic illustrative connections. No autograd computation was executed.',
        explanation:'Simulated objective. Optional preprocessing receives downstream losses; the selected module names update with this pipeline.'});
      const upstream = ['denoise_net','bias_net','ext_net','norm_net','reg_net'];
      const reached = {
        extraction:['denoise_net','bias_net','ext_net'], registration:upstream,
        segmentation:[...upstream,'seg_net'], regularization:upstream,
        parcellation:[...upstream,'par_net'],
        classification:isOriginal(selection) ? ['feature_net','graph_gen','gnn_pred'] : [...upstream,'par_net','feature_net','graph_gen','gnn_pred']
      };
      if (step !== null) {
        const values = {...losses(step), regularization:.04, parcellation:.45}, coefficients = coordination === 'single' || coordination === 'none' ? null : policy(step).weights;
        packet.terms.forEach(term => {
          if (coefficients) term.weight = coefficients[term.weight_group || term.id] * (term.relative_weight || 1);
          term.value = values[term.id]; term.weighted_value = term.value * term.weight; term.requires_grad = true;
          reached[term.id].filter(id => keys.includes(id)).forEach(id => packet.connections.push({loss:term.id,module:id,status:'autograd_path',simulated:true}));
        });
        packet.total = packet.reconstructed_total = packet.terms.reduce((s, term) => s + term.weighted_value, 0);
      }
      delete packet.forward_edges;
      return packet;
    }

    function frame(step, selection, agent = true, coordination = agent ? 'taskforce' : 'single') {
      const observed = objective(selection, coordination, step);
      return {step, epoch:Math.ceil(step / STEPS_PER_EPOCH), losses:losses(step), weighted_loss:observed.total, total_loss:observed.total,
        objective:observed, ...(agent ? {gradient_policy:policy(step)} : {}), provenance:PROVENANCE};
    }

    function metrics(epoch, fold = 1) {
      const rise = 1 - Math.exp(-epoch / 3), shift = .026 * noise(epoch, 50 + fold);
      const auc = Math.round(144 * (.57 + .23 * rise + .068 * noise(epoch, 60 + fold))) / 144;
      const correct = (rate, channel) => Math.max(0, Math.min(12,
        Math.round(12 * (rate + .13 * noise(epoch, channel + fold)))));
      const tp = correct(.5 + .24 * rise, 70), tn = correct(.58 + .18 * rise, 80);
      const accuracy = (tp + tn) / 24;
      const errors = 24 - tp - tn;
      return {brain_dice:round(.75 + .19 * rise + shift), extraction_dice:round(.75 + .19 * rise + shift), brain_jaccard:round(.64 + .24 * rise + shift),
        registration_mi:round(.43 + .17 * rise + shift), registration_cc:round(.8 + .13 * rise + shift), registration_dice:round(.6 + .12 * rise + shift),
        tissue_dice:round(.29 + .27 * rise + shift), tissue_jaccard:round(.19 + .20 * rise + shift),
        atlas_dice:round(.38 + .29 * rise + shift), atlas_jaccard:round(.25 + .25 * rise + shift),
        accuracy, auc:round(auc), balanced_accuracy:accuracy,
        sensitivity:tp/12, specificity:tn/12, macro_f1:round(tp/(2*tp+errors) + tn/(2*tn+errors)),
        reference_loss:round(1.5 - .7 * rise), count:24, simulated:true};
    }

    function foldSteps(fold) {
      if (fold !== 1) return MAX_STEPS;
      const elapsedSteps = Math.floor(Math.max(0, clock() - started) / 4000);
      return Math.min(MAX_STEPS, INITIAL_STEPS + elapsedSteps);
    }

    function trainingPacket(id) {
      const original = id.includes('original'), fold = Number(id.match(/fold(\d)/)?.[1] || 1);
      const count = foldSteps(fold);
      const selection = running && id === 'current' ? running.selection : defaults();
      const history = Array.from({length:count}, (_, i) => frame(i + 1, selection, !original));
      return {run:id, history, objective:history.at(-1).objective, coordination:original ? 'none' : 'taskforce',
        validation:Array.from({length:Math.floor(count / STEPS_PER_EPOCH)}, (_,i) => ({epoch:i+1, step:(i+1)*STEPS_PER_EPOCH, validation:metrics(i+1,fold), training_probe:{auc:round(.59 + .3 * (1 - Math.exp(-(i+1)/3)) + .04*noise(i+1,90+fold))}, provenance:PROVENANCE})),
        training_probe_size:24, reported_status:count < MAX_STEPS ? 'running' : 'complete', observed_at:stamp(),
        health:{phase:'Simulated monitoring', recovery_count:0, supervisor:{status:'running',restarts:0,max_restarts:3}, diagnosis:{warnings:['Synthetic illustration; these values are not experiment results.'],targets:{}}},
        provenance:PROVENANCE};
    }

    function experiment(id) {
      const original = id === 'demo_original';
      const folds = Array.from({length:5}, (_, i) => {
        const fold = i + 1, steps = foldSteps(fold), epoch = Math.floor(steps / STEPS_PER_EPOCH);
        return {fold, run:id+'_fold'+fold, status:steps < MAX_STEPS ? 'running' : 'complete',
          epoch, epochs:EXPERIMENT_EPOCHS, validation:metrics(epoch,fold), family_disjoint_verified:false, simulated:true};
      });
      const summary = {validation:{}};
      ['auc','accuracy','brain_dice','tissue_dice','atlas_dice','registration_mi','registration_cc'].forEach(key => {
        const values=folds.map(fold=>fold.validation[key]), mean=values.reduce((s,x)=>s+x,0)/values.length;
        summary.validation[key]={mean,sd:Math.sqrt(values.reduce((s,x)=>s+(x-mean)**2,0)/(values.length-1)),n:values.length};
      });
      return {id,label:original ? 'Synthetic cohort · Original UniBrain' : 'Synthetic cohort · Gradient aware agent',server:'Embedded browser demo',protocol:'synthetic_illustration',expected_folds:5,completed_folds:folds.filter(fold => fold.status === 'complete').length,folds,summary,provenance:PROVENANCE};
    }

    function slice(stage, index, axis) {
      const plane = atlas.planes[stage]?.[axis]?.[index];
      if (!plane) throw new Error('No reference image for this stage.');
      return plane;
    }

    function live(run) {
      const elapsed = Math.max(0,(clock()-run.started)/1000), fraction = Math.min(1,elapsed/16);
      const count = Math.min(run.steps, Math.floor(fraction*run.steps));
      const history = Array.from({length:count},(_,i)=>frame(i+1,run.selection,run.coordination!=='single',run.coordination));
      const phase = elapsed<1?'preparing':elapsed<12?'training':elapsed<13?'audit':elapsed<15?'validation':elapsed<16?'saving':'complete';
      const keys = activeModules(run.selection);
      return {phase, message:phase==='complete'?'Simulation complete. No model, GPU, or LLM was used.':'Playing synthetic '+phase+' events in this browser.',
        steps:run.steps,completed_steps:count,history,objective:history.at(-1)?.objective,elapsed_seconds:elapsed,
        resources:{device:'Browser simulation',threads:0,cpu_percent:0,rss_mb:0,simulated:true},
        modules:keys.map((key,i)=>({module:key,label:key,calls:count,seconds:round(count*(i+1)*.025),status:phase==='complete'?'completed':'started'})),
        current_module:phase==='complete'?null:keys[Math.floor(elapsed)%keys.length],
        agents:TASKS.map(role=>({role,status:phase==='complete'?'completed':'running',seconds:elapsed,usage:{input_tokens:0,output_tokens:0},decision:{summary:'Simulated local policy observation and gradient coordination; no LLM request.'}})),
        events:[{kind:'demo',message:'Simulation started. Training values are simulated. The viewer uses a public MRI template.',seconds:0},
          ...history.map(row=>({kind:'training',message:'Simulated optimizer update '+row.step,seconds:row.step*16/run.steps,losses:row.losses,weights:row.gradient_policy?.weights||{extraction:1,registration:.1,segmentation:1,classification:1},weighted_loss:row.weighted_loss,simulated:true}))],
        validation:{completed:phase==='complete'?24:0,total:24},provenance:PROVENANCE};
    }

    function report(run = null) {
      const selection=run?.selection||defaults(), steps=run?.steps||INITIAL_STEPS, keys=activeModules(selection);
      const history=Array.from({length:steps},(_,i)=>frame(i+1,selection,run ? run.coordination!=='single' : true,run?.coordination||'taskforce'));
      const planes=Object.fromEntries(Object.keys(atlas.planes).map(stage=>[stage,[0,1,2].map(axis=>[slice(stage,48,axis)])]));
      const matrix=Object.fromEntries(TASKS.map(task=>[task,Object.fromEntries(keys.map(key=>[key,{state:history.at(-1).objective.connections.some(edge=>edge.loss===task&&edge.module===key)?'active':'disconnected',parameter_tensors:8,connected_tensors:8,nonzero_tensors:8,norm:.08,simulated:true}]))]));
      return {status:'complete',profile:'original',mode:'single',created:stamp(run?.started||started),run_dir:'demo/'+(run?.id||'initial'),epochs:Math.ceil(steps / STEPS_PER_EPOCH),seed:0,
        upstream_commit:'simulated display, not an executed source revision',elapsed_seconds:16,
        build:run?{id:run.build_id,selection}:null, objective:history.at(-1).objective,
        data:{splits:{train:{count:96},val:{count:24},test:{count:24}}},environment:{device:'Browser simulation',threads:0},
        previews:{candidate:{reference_atlas:atlas.source,shape:[96,96,96],slice_indices:[48],planes,run_id:run?.id||'initial'}},
        native:{parameters:1280000,updates:history.map(row=>({step:row.step,modules_with_observed_updates:keys})),audit:{checks:{synthetic_fixture:true}},provenance:PROVENANCE},
        candidate:{steps,history,training_seconds:16,validation:metrics(Math.ceil(steps / STEPS_PER_EPOCH)),model:{config:{stages:selection}},
          gradients:{matrix,updated_parameters:Object.fromEntries(keys.map(key=>[key,100]))},spatial:{synthetic_geometry:true},checkpoint:'In-memory simulated record',
          coordination:run?.coordination==='single'?null:{mode:'taskforce',policy_updates:Math.max(0,steps-4),latest:policy(steps)},provenance:PROVENANCE},
        live:live({id:'initial',selection,steps,coordination:run?.coordination||'taskforce',started:clock()-17000}),provenance:PROVENANCE};
    }

    function code(build) {
      const text = '# Simulated review artifact. No model is executed.\n'+
        'selection = '+JSON.stringify(build.selection,null,2).replaceAll('"','\'')+'\n'+
        'pipeline = compose_stages(selection)\n'+
        'objective = collect_task_losses(pipeline)\n'+
        'check_gradient_paths(objective, pipeline)\n';
      return {scope:'Simulated code review',files:[{name:'pipeline_preview.py',kind:'illustration',code:text,sha256:'not computed: synthetic preview'}],diff:'Simulated selection update\n'+JSON.stringify(build.selection,null,2),provenance:PROVENANCE};
    }

    function buildRecord(record) {
      const elapsed=(clock()-record.started)/1000, phases=['contracts','planning','training','audit','ready'];
      const index=Math.min(4,Math.floor(elapsed));
      return {...record,status:index===4?'ready':index===3?'validating':'preparing',profile:isOriginal(record.selection)?'original':'compact',
        created:stamp(record.started),code:code(record).files[0].code,objective:objective(record.selection,record.coordination),
        events:phases.slice(0,index+1).map((phase,i)=>({phase,at:stamp(record.started+i*1000),message:['Simulated catalog contract check','Simulated agent plan for selected modules','Simulated training checks','Simulated gradient path inspection','Simulation ready for playback'][i]})),
        progress:{phase:phases[index],message:'Synthetic build preview. No source files or models are changed.'},provenance:PROVENANCE};
    }

    function assistantJob(job) {
      const done=clock()-job.started>=1800;
      const common={...job,status:done?'complete':'running',history:[],events:[{kind:'demo',message:'Simulated assistant reasoning from the bundled catalog. No LLM call.'}],sources:[],search:{status:'disabled'},context:{run_id:'demo',diagnostic:{task_norms:policy(24).gradient_norms}},provenance:PROVENANCE};
      if(!done)return common;
      if(job.mode==='roles') return {...common,result:{roles:(config.roles||[]).filter(role=>job.role==='all'||job.role===role.id).map(role=>({role:role.id,summary:'Simulated '+role.name+' review.',evidence:['Synthetic gradients are finite.','This is a scripted illustration, not an executed audit.'],recommendation:'Inspect task alignment before applying an update.',uncertainty:'No real model evidence in this demonstration.'})),action:'keep_weights',rationale:'Illustrative coordination decision. No training weights were changed.',weights_before:policy(24).weights,weights_proposed:policy(24).weights}};
      const stage=config.catalog.stages.find(item=>item.id===job.stage)||config.catalog.stages.find(item=>item.id==='registration');
      const option=stage.options.find(item=>item.research&&item.id!=='M15');
      const source=option?{source_id:option.id,publication:option.label,origin:'Bundled project catalog',verification:'Metadata from the catalog; not newly verified by this simulation.',paper:option.research.paper,code:option.research.code}:null;
      return {...common,sources:source?[source]:[],result:{answer:'Simulated response: first inspect the stage input and output shapes, then retain differentiable tensor operations. Verify the classification loss reaches the chosen module before joint training. This reply is a fixed demonstration, not live research.',source_ids:source?[source.source_id]:[],recommendations:option?[{stage:stage.id,tool_id:option.id,source_id:option.id,reason:'Example from the bundled catalog for demonstrating the selection workflow.',verification_needed:'Run real spatial, gradient, and parameter update checks on your backend.'}]:[],next_question:''}};
    }

    async function fetch(path, options = {}) {
      if (!config) return response({error:'Demo backend is not configured.'},503);
      if(options.signal?.aborted) throw new DOMException('The operation was aborted.','AbortError');
      const url=new URL(String(path),'https://demo.invalid'), route=url.pathname, method=(options.method||'GET').toUpperCase();
      let body={};
      try {if(options.body)body=JSON.parse(options.body);}catch(_){return response({error:'Expected JSON.'},400);}
      try {
        if(method==='GET'&&route==='/api/catalog')return response(config.catalog);
        if(method==='GET'&&route==='/api/runtime')return response({native_cuda:false,original_available:false,device:'Browser simulation',demo:true});
        if(method==='GET'&&route==='/api/status'){
          const value=running?live(running):null, completed=value?.phase==='complete';
          if(completed)latest=report(running);
          return response({app:'unibrain_agent',api_version:2,job:running?'run':null,running:!!running&&!completed,completed:!!completed,run_dir:running?'demo/'+running.id:null,build_id:running?.build_id,live:value});
        }
        if(method==='GET'&&route==='/api/report')return response({report:latest||report()});
        if(method==='POST'&&route==='/api/objective/preview')return response(objective(validateSelection(body.selection),body.coordination));
        if(method==='POST'&&route==='/api/prepare'){
          const selection=validateSelection(body.selection),id='demo-build-'+(++serial);
          builds.set(id,{id,selection,coordination:body.coordination||'single',started:clock()});
          return response({build_id:id});
        }
        const buildMatch=route.match(/^\/api\/builds\/([^/]+)(?:\/(code|run))?$/);
        if(buildMatch){
          const record=builds.get(buildMatch[1]);if(!record)return response({error:'This simulated build is not in this browser session.'},404);
          if(method==='GET'&&buildMatch[2]==='code')return response(code(record));
          if(method==='GET'&&!buildMatch[2])return response(buildRecord(record));
          if(method==='POST'&&buildMatch[2]==='run'){
            if(buildRecord(record).status!=='ready')return response({error:'Wait for the simulated build checks.'},409);
            if(running&&live(running).phase!=='complete')return response({error:'A simulation is already playing.'},409);
            const steps=Math.max(2,Math.min(MAX_STEPS,Math.floor(Number(body.steps))||12));
            running={id:'demo-run-'+(++serial),build_id:record.id,selection:clone(record.selection),coordination:record.coordination,steps,started:clock()};
            return response({status:'running',run_id:running.id,steps,simulated:true});
          }
        }
        if(method==='GET'&&route==='/api/volume'){
          const index=Number(url.searchParams.get('index')||48),stage=url.searchParams.get('stage')||'input';
          if(!Number.isInteger(index)||index<0||index>95)return response({error:'Slice must be between 0 and 95.'},400);
          return response({index,shape:[96,96,96],planes:[0,1,2].map(axis=>slice(stage,index,axis))});
        }
        if(method==='GET'&&route==='/api/training-experiments')return response({experiments:['demo_agent','demo_original'].map(id=>experiment(id))});
        if(method==='GET'&&route.startsWith('/api/training-experiments/')){
          const id=route.split('/').at(-1);return ['demo_agent','demo_original'].includes(id)?response(experiment(id)):response({error:'Unknown synthetic experiment.'},404);
        }
        if(method==='GET'&&route.startsWith('/api/training-runs/')){
          const id=route.split('/').at(-1);return /^(demo_(agent|original)_fold[1-5]|current)$/.test(id)?response(trainingPacket(id)):response({error:'Unknown synthetic training run.'},404);
        }
        if(method==='GET'&&route==='/api/assistant')return response({roles:config.roles||[],research_prompt:config.research_prompt||'Demonstrate a catalog review.',latest:{},context:{run_id:'demo',diagnostic:{task_norms:policy(24).gradient_norms}}});
        if(method==='POST'&&route==='/api/assistant'){
          const id='demo-assistant-'+(++serial);jobs.set(id,{id,mode:body.mode||'research',role:body.role||'all',stage:body.stage,message:String(body.message||'').slice(0,2000),started:clock()});return response({id});
        }
        if(method==='GET'&&route.startsWith('/api/assistant/jobs/')){const job=jobs.get(route.split('/').at(-1));return job?response(assistantJob(job)):response({error:'Unknown simulated job.'},404);}
        if(method==='GET'&&route==='/api/ai')return response({provider:'demo',providers:[{id:'demo',name:'Embedded simulation',installed:false,credential:'not supplied',env:'None'}],profiles:{demo:{model:'Scripted illustration',base_url:''}},test:null});
        if(route.startsWith('/api/ai/'))return response({error:'AI authentication and external calls are disabled in the embedded simulation.'},403);
        if(method==='GET'&&route==='/api/data')return response({selected:{path:'embedded/synthetic_phantom',audit:{splits:{train:{count:96},val:{count:24},test:{count:24}},warnings:['Synthetic phantom and illustrative counts. No research subjects or files are loaded.']}},job:null});
        if(method==='POST'&&route==='/api/data/jobs'){
          const id='demo-data-'+(++serial);jobs.set(id,{id,mode:body.mode||'inspect',started:clock()});return response({id});
        }
        if(method==='GET'&&route.startsWith('/api/data/jobs/')){
          const job=jobs.get(route.split('/').at(-1));if(!job)return response({error:'Unknown simulated data job.'},404);
          const audit={splits:{train:{count:96},val:{count:24},test:{count:24}},warnings:['Synthetic illustration.']};
          return response({...job,status:'complete',events:[],inventory:{items:[{path:'embedded/synthetic_phantom',kind:'Synthetic phantom',status:'ready',audit}]},result:{dataset:'embedded/synthetic_phantom',audit,answer:'The demo contains synthetic 96 cubed images and illustrative labels. Real conversion requires your own backend and explicit source definitions.',blockers:[]}});
        }
        if(method==='POST'&&route==='/api/data/use')return response({path:'embedded/synthetic_phantom',audit:{splits:{train:{count:96},val:{count:24},test:{count:24}},warnings:['Synthetic illustration.']}});
        if(method==='GET'&&route==='/api/research-cycles')return response({defaults:{source:'embedded catalog',data:'synthetic phantom',gpu:0,stages:['extraction','registration','segmentation']},job:null});
        if(route.startsWith('/api/research-cycles'))return response({error:'The full research cycle needs a real backend. Use Apply changes and Confirm and Run for the embedded simulation.'},501);
        return response({error:'This action is not available in the embedded simulation.'},501);
      } catch(error) {return response({error:error.message},400);}
    }

    const api={configure,fetch,enable(){api.active=true;return api;},disable(){api.active=false;return api;},active:false};
    if(initialConfig)configure(initialConfig);
    return api;
  }

  root.DemoBackend=createDemoBackend();
  if(typeof module!=='undefined'&&module.exports)module.exports={createDemoBackend};
})(typeof globalThis!=='undefined'?globalThis:window);
