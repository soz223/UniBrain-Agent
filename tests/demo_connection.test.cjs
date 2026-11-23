const {test}=require('node:test');
const assert=require('node:assert/strict');
const {readFileSync}=require('node:fs');
const vm=require('node:vm');
const script=readFileSync('web/connection.js','utf8');

function demoContext(available=true) {
  const calls=[],storage=[],nodes={};
  for(const id of ['demoBadge','demoToggle'])nodes[id]={hidden:true,addEventListener:(type,fn)=>nodes[id][type]=fn,setAttribute:()=>{}};
  const sandbox={URL,Headers,AbortSignal,Response,location:{href:'https://soz223.github.io/UniBrain-Agent/?demo=1#roles',origin:'https://soz223.github.io',protocol:'https:',assign:url=>calls.push({navigate:url})},
    document:{getElementById:id=>id==='appConfig'?{textContent:'{"standalone":true}'}:nodes[id]},
    localStorage:{getItem:key=>{storage.push(key);return 'https://private.example';}},
    sessionStorage:{getItem:key=>{storage.push(key);return 'private-token';},removeItem:key=>storage.push(key)},
    fetch:()=>{throw new Error('External fetch forbidden');}};
  if(available)sandbox.DemoBackend={configure:config=>calls.push({configure:config}),fetch:async(path,options)=>{calls.push({path,options});return new Response(JSON.stringify({app:'unibrain_agent',api_version:2,simulated:true}));}};
  vm.createContext(sandbox);vm.runInContext(script,sandbox);
  return {sandbox,calls,storage,nodes};
}

test('demo initialization uses only the embedded backend and ignores saved credentials',async()=>{
  const {sandbox,calls,storage,nodes}=demoContext();
  assert.equal(await vm.runInContext('backendReady',sandbox),true);
  assert.deepEqual(storage,[]);
  assert.equal(vm.runInContext('backend.token',sandbox),'');
  assert.equal(nodes.demoBadge.hidden,false);
  assert.equal(nodes.demoToggle.textContent,'Exit demo');
  assert.equal(calls.find(item=>item.path)?.path,'/api/status');
});

test('demo intercepts explicit backend targets without forwarding their token',async()=>{
  const {sandbox,calls}=demoContext();await vm.runInContext('backendReady',sandbox);
  await vm.runInContext("backendFetch('/api/builds',{method:'POST',body:'{}'},{address:'https://private.example',token:'private-token'})",sandbox);
  const request=calls.at(-1);
  assert.equal(request.path,'/api/builds');assert.equal(request.options.method,'POST');
  assert.equal(JSON.stringify(request).includes('private-token'),false);
});

test('missing demo data fails closed instead of contacting a server',async()=>{
  const {sandbox,storage}=demoContext(false);
  assert.equal(await vm.runInContext('backendReady',sandbox),false);
  assert.deepEqual(storage,[]);
  await assert.rejects(vm.runInContext("backendFetch('/api/status')",sandbox),/Demo data unavailable/);
});

test('exiting demo changes only the URL and preserves the chosen workspace',async()=>{
  const {nodes,calls,storage}=demoContext();nodes.demoToggle.click();
  const destination=calls.find(item=>item.navigate).navigate;
  assert.equal(destination,'https://soz223.github.io/UniBrain-Agent/#roles');
  assert.deepEqual(storage,[]);
});
