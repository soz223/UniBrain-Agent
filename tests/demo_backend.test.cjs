const {test} = require('node:test');
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const vm = require('node:vm');
const {createDemoBackend} = require('../web/demo.js');

function fixture() {
  const definitions = [
    ['denoising','Denoising',['none','M14']], ['bias','Bias correction',['none','M01']],
    ['extraction','Brain extraction',['unibrain','M02','M03']], ['normalization','Intensity normalization',['none','M04','M05']],
    ['registration','Registration',['unibrain','M06','M07','M03']], ['segmentation','Tissue segmentation',['unibrain','M08']],
    ['parcellation','Brain parcellation',['unibrain','soft_atlas','M09','M10']], ['graph','Network generation',['unibrain','continuous','M11']],
    ['classification','Classification',['unibrain','M12','M13']], ['surface','Cortical surface',['none','M15']]
  ];
  return {catalog:{defaults:Object.fromEntries(definitions.map(([id,,choices])=>[id,choices[0]])),stages:definitions.map(([id,name,choices])=>({id,name,options:choices.map(choice=>({id:choice,label:choice==='M14'?'RED WGAN':choice==='M06'?'VoxelMorph':choice==='M09'?'FastSurferCNN':name+' '+choice}))}))},objective:{terms:[],stages:[]},roles:[],research_prompt:'Bundled prompt'};
}

function harness() {
  let now = Date.UTC(2026,0,1);
  const config=fixture(),backend=createDemoBackend(config,()=>now);
  return {config,backend,advance:ms=>now+=ms,get:async path=>(await backend.fetch(path)).json(),post:async(path,body)=>(await backend.fetch(path,{method:'POST',body:JSON.stringify(body)})).json()};
}

test('demo status is identifiable and idle while original pages receive complete synthetic records',async()=>{
  const h=harness(),status=await h.get('/api/status'),data=await h.get('/api/report');
  assert.equal(status.app,'unibrain_agent');assert.equal(status.api_version,2);assert.equal(status.running,false);
  assert.equal(h.backend.active,true);assert.equal(data.provenance.model_executed,false);
  const report=data.report;
  assert.equal(report.previews.candidate.shape.join(','),'96,96,96');
  for(const stage of ['input','mask','extracted'])assert.equal(report.previews.candidate.planes[stage].length,3);
  assert.ok(report.candidate.history.length>1);assert.equal(report.candidate.history.at(-1).gradient_policy.gram.length,4);
  assert.equal(report.native.updates.length,report.candidate.steps);
  assert.equal(report.provenance.synthetic,true);
});

test('pipeline selection changes methods, optional stages, objective terms and connections together',async()=>{
  const h=harness(),selection={...h.config.catalog.defaults,denoising:'M14',normalization:'M04',registration:'M06',parcellation:'M09',graph:'continuous'};
  const objective=await h.post('/api/objective/preview',{selection,coordination:'roles'});
  assert.equal(objective.stages.length,8);
  assert.equal(objective.stages.find(stage=>stage.id==='denoising').method,'RED WGAN');
  assert.deepEqual(objective.stages.find(stage=>stage.id==='denoising').loss_ids,[]);
  assert.ok(objective.terms.some(term=>term.id==='regularization'));
  assert.ok(objective.terms.some(term=>term.id==='parcellation'));
  assert.equal(objective.stages.find(stage=>stage.id==='parcellation').method,'FastSurferCNN');
  const context=vm.createContext({});vm.runInContext(readFileSync('web/objective.js','utf8'),context);
  const formula=context.objectiveFormula(objective);
  assert.match(formula.find(term=>term.id==='registration').symbol,/VoxelMorph/);
  assert.match(formula.find(term=>term.id==='parcellation').symbol,/FastSurferCNN/);
  assert.equal(h.config.catalog.defaults.denoising,'none');
});

test('Apply, review, Confirm, updates and completion follow the original API contracts',async()=>{
  const h=harness(),selection={...h.config.catalog.defaults,denoising:'M14',parcellation:'soft_atlas',graph:'continuous'};
  const prepared=await h.post('/api/prepare',{selection,coordination:'roles'}),path='/api/builds/'+prepared.build_id;
  assert.equal((await h.get(path)).status,'preparing');
  const early=await h.backend.fetch(path+'/run',{method:'POST',body:JSON.stringify({steps:5})});assert.equal(early.status,409);
  h.advance(4200);const ready=await h.get(path);assert.equal(ready.status,'ready');assert.equal(ready.events.length,5);
  const review=await h.get(path+'/code');assert.match(review.files[0].code,/Simulated/);assert.match(review.files[0].code,/M14/);
  await h.post(path+'/run',{steps:5});assert.equal((await h.get('/api/status')).running,true);
  h.advance(8000);const mid=await h.get('/api/status');assert.equal(mid.live.phase,'training');assert.ok(mid.live.history.length>0);
  assert.equal(mid.live.history.at(-1).objective.coordination,'roles');
  assert.ok(mid.live.history.at(-1).objective.connections.some(edge=>edge.module==='denoise_net'));
  h.advance(9000);const done=await h.get('/api/status');assert.equal(done.running,false);assert.equal(done.completed,true);assert.equal(done.live.completed_steps,5);
  const saved=(await h.get('/api/report')).report;assert.equal(saved.build.id,prepared.build_id);assert.equal(saved.candidate.steps,5);
  const objective=saved.candidate.history.at(-1).objective;
  assert.ok(Math.abs(objective.total-objective.terms.reduce((sum,term)=>sum+term.weight*term.value,0))<1e-12);
});

test('both experiments expose five folds, sample SD, policies and epoch outcomes to the original Training renderer',async()=>{
  const h=harness(),list=await h.get('/api/training-experiments');assert.equal(list.experiments.length,2);
  const exp=await h.get('/api/training-experiments/demo_agent');assert.equal(exp.folds.length,5);
  const values=exp.folds.map(fold=>fold.validation.auc),mean=values.reduce((a,b)=>a+b)/5;
  assert.equal(exp.summary.validation.auc.mean,mean);
  assert.ok(exp.summary.validation.auc.sd>0);
  const packet=await h.get('/api/training-runs/demo_agent_fold1'),row=packet.history.at(-1),policy=row.gradient_policy;
  assert.equal(packet.validation.at(-1).epoch,exp.folds[0].epoch);
  assert.equal(packet.validation.at(-1).validation.auc,exp.folds[0].validation.auc);
  assert.equal(policy.gram.length,4);assert.equal(policy.observations.classification.length,5);
  assert.ok(Math.abs(Object.values(policy.actions).reduce((a,b)=>a+b)-1)<1e-12);
  for(let i=0;i<4;i++)for(let j=0;j<4;j++)assert.equal(policy.gram[i][j],policy.gram[j][i]);
  assert.ok(policy.policy_updates>0);
  assert.ok(Math.abs(policy.reward - policy.reward_terms.training_loss_improvement + policy.reward_terms.gradient_norm_penalty)<1e-6);
  const trace=packet.history.map(row=>row.gradient_policy);
  assert.ok(trace.some(p=>p.reward<0) && trace.some(p=>p.reward>0));
  const spread = rows => {
    const shares = rows.map(p => p.actions.classification);
    return Math.max(...shares) - Math.min(...shares);
  };
  assert.ok(spread(trace.slice(-32)) < .25 * spread(trace.slice(0,32)));
  const alignment=trace.map(p=>p.cosine_similarity[0][1]);
  assert.ok(Math.min(...alignment)<-.2 && Math.max(...alignment)>.2);
  const context=vm.createContext({escapeHtml:String,titleCase:value=>value});vm.runInContext(readFileSync('web/training.js','utf8'),context);
  const frames=context.trainingFrames(packet);
  assert.equal(frames.length,256);
  assert.deepEqual([frames[0].step,frames.at(-1).step],[1,256]);
  assert.match(context.trainingDiagram(row.objective,'classification'),/backward-path/);
  assert.match(context.trainingCritics(policy),/Critic 4/);
  h.advance(12000);assert.ok((await h.get('/api/training-runs/demo_agent_fold1')).history.length>packet.history.length);
  const original=await h.get('/api/training-runs/demo_original_fold2');assert.equal(original.coordination,'none');assert.equal(original.history[0].gradient_policy,undefined);
});

test('the public atlas supplies different anatomical slices and planes',async()=>{
  const h=harness(),a=await h.get('/api/volume?stage=input&index=48'),b=await h.get('/api/volume?stage=input&index=49');
  assert.equal(a.planes.length,3);assert.ok(a.planes.every(uri=>uri.startsWith('data:image/png;base64,')));
  assert.notEqual(a.planes[0],b.planes[0]);assert.notEqual(a.planes[0],a.planes[1]);
  for(const index of [0,95])assert.equal((await h.backend.fetch('/api/volume?index='+index)).status,200);
  assert.equal((await h.backend.fetch('/api/volume?index=96')).status,400);
});

test('unsupported routes and credentials fail closed and no API path can fall back to the network',async()=>{
  const originalFetch=global.fetch;let calls=0;
  global.fetch=async()=>{calls++;throw new Error('Network access is forbidden');};
  try {
    const h=harness();
    assert.equal((await h.backend.fetch('/api/arbitrary')).status,501);
    assert.equal((await h.backend.fetch('/api/ai/connect',{method:'POST',body:JSON.stringify({api_key:'must-not-be-stored'})})).status,403);
    assert.equal((await h.backend.fetch('https://another.invalid/api/status')).status,200);
    await h.get('/api/report');await h.get('/api/training-runs/demo_agent_fold1');
    assert.equal(calls,0);
    assert.doesNotMatch(JSON.stringify(await h.get('/api/ai')),/must-not-be-stored/);
    const controller=new AbortController();controller.abort();
    await assert.rejects(h.backend.fetch('/api/status',{signal:controller.signal}),{name:'AbortError'});
  } finally {global.fetch=originalFetch;}
});

test('invalid catalog choices and unsupported surface inputs are refused',async()=>{
  const h=harness();
  for(const selection of [{...h.config.catalog.defaults,denoising:'unknown'},{...h.config.catalog.defaults,extraction:'M03'},{...h.config.catalog.defaults,surface:'M15'}]) {
    const response=await h.backend.fetch('/api/prepare',{method:'POST',body:JSON.stringify({selection})});assert.equal(response.status,400);
  }
});

test('research is explicitly scripted and simulated jobs never claim an LLM invocation',async()=>{
  const h=harness(),job=await h.post('/api/assistant',{mode:'research',message:'Review registration',stage:'registration'});
  assert.equal((await h.get('/api/assistant/jobs/'+job.id)).status,'running');
  h.advance(2000);const complete=await h.get('/api/assistant/jobs/'+job.id);
  assert.equal(complete.status,'complete');assert.match(complete.result.answer,/fixed demonstration/);assert.equal(complete.provenance.llm_called,false);
  assert.deepEqual((await h.get('/api/assistant')).latest,{});
});
