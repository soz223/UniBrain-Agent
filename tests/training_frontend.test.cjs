const {test}=require('node:test');
const assert=require('node:assert/strict');
const {readFileSync}=require('node:fs');
const vm=require('node:vm');

const context=vm.createContext({});
vm.runInContext(readFileSync('web/training.js','utf8'),context);

test('replay uses real steps and keeps missing policy telemetry missing',()=>{
  const rows=context.trainingFrames({history:[{step:9},{step:2},{step:9,gradient_policy:{reward:1}},{step:null}]});
  assert.equal(rows.length,2);assert.equal(rows[0].step,2);assert.equal(rows[0].gradient_policy,undefined);
  assert.equal(context.trainingSelected(rows,2,false).step,2);
  assert.equal(context.trainingSelected(rows,2,true).step,9);
});

test('zero gradient norm is undefined, not a zero or positive cosine',()=>{
  const policy={gradient_norms:{extraction:0,registration:2},cosine_similarity:[[0,0],[0,1]]};
  assert.equal(context.trainingCosine(policy,0,1),null);
  assert.equal(context.trainingCosine(policy,1,1),1);
  assert.equal(context.trainingNumber(null),'N/A');
});

test('history keeps a bounded moving window without reindexing step numbers',()=>{
  const rows=context.trainingFrames({history:Array.from({length:300},(_,step)=>({step}))});
  assert.equal(rows.length,128);assert.equal(rows[0].step,172);
});

test('core and legacy records retain their actual coefficient and replay semantics',()=>{
  const core=context.trainingPolicyDescription({profile:'paper_core',gradient_normalization:'l2'});
  assert.match(core.formula,/‖∇L/);assert.match(core.scope,/Every trainable/);assert.equal(core.action,'a');
  const legacy=context.trainingPolicyDescription({weights:{extraction:1}});
  assert.match(legacy.formula,/base coefficient/);assert.match(legacy.label,/LEGACY/);assert.equal(legacy.action,'w');
  assert.match(context.trainingPolicyDescription(null).scope,/await/);
  assert.match(context.trainingPolicyDescription({profile:'paper_core',action_bound:1}).method,/±1 before softmax/);
  assert.doesNotMatch(core.method,/bounds actor outputs/);
});

function assistantPage(hash) {
  const nodes=new Map(),listeners={};let resolveBackend;
  const node=id=>{if(!nodes.has(id))nodes.set(id,{id,hidden:false,innerHTML:'',dataset:{},classList:{toggle(){}},addEventListener(){},scrollIntoView(){},setAttribute(){}});return nodes.get(id);};
  const views=['pipeline','roles','settings','research','data'].map(name=>({...node(name+'View'),dataset:{workspace:name}}));
  const sandbox={location:{hash},history:{replaceState(_,__,hash){sandbox.location.hash=hash;}},
    document:{querySelectorAll:selector=>selector==='[data-workspace]'?views:[],querySelector:()=>node('query')},
    byId:node,window:{addEventListener:(name,fn)=>listeners[name]=fn,dispatchEvent(){}},CustomEvent:function(){},
    appConfig:{standalone:true,roles:[],research_prompt:'',catalog:{stages:[]}},backendReady:new Promise(resolve=>resolveBackend=resolve),
    escapeHtml:x=>x,tasks:[],titleCase:x=>x};
  vm.createContext(sandbox);vm.runInContext(readFileSync('web/assistant.js','utf8'),sandbox);
  return {sandbox,views,listeners,resolveBackend};
}

test('offline public startup opens Pipeline, including explicit workspace hashes',async()=>{
  for(const hash of ['','#workspace','#builderPanel']){
    const page=assistantPage(hash);
    assert.equal(page.views.find(view=>!view.hidden).dataset.workspace,'pipeline');
    page.resolveBackend(false);await new Promise(resolve=>setImmediate(resolve));
    assert.equal(page.sandbox.location.hash,'#workspace');
  }
});

test('backend readiness cannot overwrite user navigation, and hash changes are honored',async()=>{
  const page=assistantPage('#workspace');
  page.sandbox.location.hash='#roles';page.listeners.hashchange();
  page.resolveBackend(false);await new Promise(resolve=>setImmediate(resolve));
  assert.equal(page.sandbox.location.hash,'#roles');
  assert.equal(page.views.find(view=>!view.hidden).dataset.workspace,'roles');
  page.sandbox.location.hash='#settings';page.listeners.hashchange();
  assert.equal(page.views.find(view=>!view.hidden).dataset.workspace,'settings');
});
