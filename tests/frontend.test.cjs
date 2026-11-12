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

const objectiveContext=vm.createContext({});
vm.runInContext(readFileSync('web/objective.js','utf8'),objectiveContext);
test('formula renders backend terms and coefficients, including unfamiliar losses',()=>{
 const packet={status:'observed',terms:[{id:'novel',name:'New loss',weight:0.037},{id:'regularization',name:'Field penalty',weight:0.2}]};
 const formula=objectiveContext.objectiveFormula(packet);
 assert.equal(formula.length,2);assert.equal(formula[0].expression,'0.037');assert.equal(formula[1].name,'Field penalty');
});
test('declared coordinated subterms share the backend supplied weight group',()=>{
 const packet={status:'declared',coordination:'taskforce',terms:[{id:'penalty',name:'Penalty',weight:.001,weight_group:'registration',relative_weight:.01}]};
 assert.equal(objectiveContext.objectiveFormula(packet)[0].expression,'0.01 × w(registration, t)');
 packet.status='observed';assert.equal(objectiveContext.objectiveFormula(packet)[0].expression,'0.001');
});

test('loss colors follow their owning stage rather than loss order',()=>{
 const packet={stages:[{id:'denoising',loss_ids:[]},{id:'registration',loss_ids:['registration','regularization']}]};
 assert.equal(objectiveContext.objectiveTermColor(packet,'registration',0),objectiveContext.objectiveTermColor(packet,'regularization',4));
});
