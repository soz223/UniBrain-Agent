const {test}=require('node:test');
const assert=require('node:assert/strict');
const {readFileSync}=require('node:fs');
const vm=require('node:vm');
const script=readFileSync('web/connection.js','utf8');

function context(saved='',token='') {
  const calls=[];
  const sandbox={URL,Headers,AbortSignal,location:{protocol:'https:',origin:'https://soz223.github.io'},
    document:{getElementById:()=>({textContent:'{"standalone":true}'})},
    localStorage:{getItem:()=>saved},sessionStorage:{getItem:()=>token},
    fetch:async(url,options)=>{calls.push({url,options});return {ok:true,json:async()=>({app:'unibrain_agent',api_version:1})};}};
  vm.createContext(sandbox);vm.runInContext(script,sandbox);return {sandbox,calls};
}

test('a public frontend makes no backend request before configuration',async()=>{
  const {sandbox,calls}=context();assert.equal(await vm.runInContext('backendReady',sandbox),false);assert.equal(calls.length,0);
});
test('configured requests use the backend prefix and bearer token, without cookies or redirects',async()=>{
  const {sandbox,calls}=context('https://gpu.example/workspace','test-token');
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
