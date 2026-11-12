const {test}=require('node:test');
const assert=require('node:assert/strict');
const {readFileSync}=require('node:fs');
const vm=require('node:vm');
const {execFileSync}=require('node:child_process');
const script=readFileSync('web/connection.js','utf8');

function context(saved='',token='',manual=false) {
  const calls=[];
  const sandbox={URL,Headers,AbortSignal,location:{protocol:'https:',origin:'https://soz223.github.io'},
    document:{getElementById:()=>({textContent:'{"standalone":true}'})},
    localStorage:{getItem:()=>saved},sessionStorage:{getItem:key=>key==='unibrainManualConnect'?(manual?saved:''):token,removeItem:()=>{}},
    fetch:async(url,options)=>{calls.push({url,options});return {ok:true,json:async()=>({app:'unibrain_agent',api_version:2})};}};
  vm.createContext(sandbox);vm.runInContext(script,sandbox);return {sandbox,calls};
}

test('a public frontend makes no backend request before configuration',async()=>{
  const {sandbox,calls}=context();assert.equal(await vm.runInContext('backendReady',sandbox),false);assert.equal(calls.length,0);
});
test('configured requests use the backend prefix and bearer token, without cookies or redirects',async()=>{
  const {sandbox,calls}=context('https://gpu.example/workspace','test-token',true);
  assert.equal(await vm.runInContext('backendReady',sandbox),true);
  assert.equal(calls[0].url,'https://gpu.example/workspace/api/status');
  assert.equal(calls[0].options.headers.get('Authorization'),'Bearer test-token');
  assert.equal(calls[0].options.credentials,'omit');assert.equal(calls[0].options.redirect,'error');
});
test('credentials cannot be sent to an arbitrary URL or an insecure remote endpoint',async()=>{
  const {sandbox}=context();
  assert.throws(()=>vm.runInContext("normalizeBackend('http://gpu.example')",sandbox));
  assert.throws(()=>vm.runInContext("normalizeBackend('https://key@gpu.example')",sandbox));
  await assert.rejects(vm.runInContext("backendFetch('https://other.example',{},{address:'https://gpu.example',token:'secret'})",sandbox));
});

test('public pages ignore old backend credentials until a manual connection',async()=>{
  const {sandbox,calls}=context('https://gpu.example','previous-session-token');
  assert.equal(await vm.runInContext('backendReady',sandbox),false);
  assert.equal(calls.length,0);assert.equal(vm.runInContext('aiSession',sandbox),'');
});

const library=JSON.parse(execFileSync('python',['-c','import json; from unibrain_agent.catalog import catalog; print(json.dumps(catalog()))'],{encoding:'utf8'}));
const objectiveContext=vm.createContext({});
vm.runInContext(readFileSync('web/objective.js','utf8'),objectiveContext);
const describe=objectiveContext.describeObjective;
const objective=(changes={},options={})=>describe({selection:{...library.defaults,...changes},defaults:library.defaults,stages:library.stages,...options});

test('the original objective names its four losses, weights, supervision and discrete boundaries',()=>{
  const model=objective();
  assert.equal(model.original,true);
  assert.deepEqual(Array.from(model.terms,t=>t.short),['BCE','GCC','CE','CE']);
  assert.deepEqual(Array.from(model.terms,t=>t.weight),[1,0.1,1,1]);
  assert.match(model.terms[0].detail,/each subject/);
  assert.match(model.terms[2].detail,/template tissue labels warped/);
  assert.match(model.boundary,/discrete operations/);
  assert.equal(model.terms.some(t=>t.id==='parcellation'),false);
});

test('compact parcellation CE shares the tissue task weight; full parcellation CE stays separate',()=>{
  const compact=objective({parcellation:'M09',graph:'continuous'},{coordination:'roles'});
  const aal=compact.terms.find(t=>t.id==='parcellation');
  assert.equal(aal.weight,0.2);assert.equal(aal.weightSymbol,'tissue');assert.equal(aal.adaptive,true);
  assert.match(compact.terms.find(t=>t.id==='segmentation').detail,/subject’s tissue reference/);
  const full=objective({}, {cycle:{method:'fastsurfer',config:{coordination:'taskforce'}}});
  const fullAal=full.terms.find(t=>t.id==='parcellation');
  assert.equal(fullAal.weight,1);assert.equal(fullAal.adaptive,false);
  assert.equal(full.terms.filter(t=>t.adaptive).length,4);
  assert.match(full.terms.find(t=>t.id==='segmentation').detail,/template tissue labels/);
});

test('only a matching compact build can change BCE to BCE plus Dice',()=>{
  const selection={...library.adapter_defaults,extraction:'M02'};
  const build={selection,plan:{extraction_loss:'bce_dice'},status:'ready'};
  const model=objective(selection,{build});
  assert.equal(model.terms[0].short,'BCE + Dice');assert.equal(model.status,'Prepared objective');
  assert.equal(objective({...selection,extraction:'M03'},{build}).terms[0].short,'BCE');
  assert.equal(objective(selection,{build,coordination:'roles'}).terms[0].short,'BCE');
  assert.equal(objective({}, {build:{...build,selection:library.defaults}}).terms[0].short,'BCE');
  assert.equal(objective(selection,{build:{...build,status:'failed'}}).status,'Planned objective');
});

test('dense regularization and auxiliary supervision reflect each execution profile',()=>{
  const compact=objective({registration:'M06',segmentation:'M08'});
  assert.deepEqual(Array.from(compact.terms[1].extras),['+ 0.01 smoothness']);
  assert.match(compact.terms[2].detail,/0.5 × CE/);
  assert.equal(objective({}, {cycle:{method:'voxelmorph'}}).terms[1].extras.length,0);
  assert.match(objective({}, {cycle:{method:'nnunet'}}).terms[2].detail,/normalized multi-scale CE/);
});

test('skipping and resetting removes optional losses and stale methods',()=>{
  const modified=objective({parcellation:'M10',denoising:'M14'});
  assert.equal(modified.terms.length,5);
  const restored=objective();
  assert.equal(restored.terms.length,4);
  assert.equal(restored.preprocessing.length,0);
  const message=objectiveContext.objectiveChange(modified,restored);
  assert.match(message,/denoising skipped/);
  assert.match(message,/loss terms/);
});

test('preprocessing uses downstream losses and TaskForce changes weights, not loss definitions',()=>{
  const original=objective();
  const preprocess=objective({bias:'M01'});
  assert.equal(preprocess.terms.length,4);
  assert.match(objectiveContext.objectiveChange(original,preprocess),/no separate preprocessing loss/);
  const policy=objective({}, {coordination:'taskforce'});
  assert.equal(policy.objectiveKey,original.objectiveKey);
  assert.equal(policy.terms.every(t=>t.adaptive),true);
  assert.match(objectiveContext.objectiveChange(original,policy),/actors adapt/);
  assert.equal(objectiveContext.objectiveChange(policy,policy),'');
});

test('every full research candidate maps to an available catalog choice without changing the draft',()=>{
  const methods=vm.runInContext('objectiveMethods',objectiveContext);
  for(const method of Object.keys(methods)) {
    const draft={...library.defaults};
    const model=objective(draft,{cycle:{method}});
    assert.equal(model.full,true);
    for(const [stage,tool] of Object.entries(model.choices))
      assert.ok(library.stages.find(s=>s.id===stage).options.some(o=>o.id===tool),method+': '+stage);
    assert.deepEqual(draft,library.defaults);
  }
});
