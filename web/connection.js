const appConfig = JSON.parse(document.getElementById('appConfig').textContent);
const backendPreference = 'unibrainBackendV1';
const backendTokenKey = address => 'unibrainAccessV1:' + address;
let aiSession = '', aiSessionProvider = '';
function normalizeBackend(value) {
  const url = new URL(value);
  if (url.username || url.password || url.search || url.hash || (url.protocol !== 'https:' && !(url.protocol === 'http:' && ['127.0.0.1','localhost','[::1]'].includes(url.hostname)))) throw new Error('Use an HTTPS backend URL, or HTTP on localhost. Do not include credentials or query parameters.');
  return url.href.replace(/\/+$/, '');
}
let savedBackend = '';
try {savedBackend = localStorage.getItem(backendPreference) || '';} catch (_) {}
const backend = {address:savedBackend==='disconnected'?'':savedBackend||(!appConfig.standalone&&location.protocol!=='file:'?location.origin:''),token:'',connected:false,error:'',status:null};
try {
  if (backend.address) {backend.address=normalizeBackend(backend.address);backend.token=sessionStorage.getItem(backendTokenKey(backend.address)) || '';}
} catch (_) {backend.address='';}
if (appConfig.standalone) {
  const manualConnect = sessionStorage.getItem('unibrainManualConnect');
  sessionStorage.removeItem('unibrainManualConnect');
  if (manualConnect !== backend.address) {backend.address='';backend.token='';}
}

async function backendFetch(path, options = {}, target = backend) {
  if (!target.address) throw new Error('Connect your backend in AI settings to use this feature.');
  if (!path.startsWith('/api/')) throw new Error('Invalid backend route');
  const headers = new Headers(options.headers || {});
  if (target.token) headers.set('Authorization','Bearer '+target.token);
  if (aiSession && target.address === backend.address) headers.set('X-UniBrain-AI-Session', aiSession);
  const response = await fetch(target.address + path, {...options,headers,credentials:'omit',redirect:'error',signal:options.signal || AbortSignal.timeout(30000)});
  return response;
}

async function requestJson(path, data, target) {
  let response;
  try {response=await backendFetch(path,data===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)},target);}
  catch(error) {throw new Error(error.message.includes('Connect your backend') ? error.message : 'Cannot reach this backend. Check its address, HTTPS certificate and allowed frontend origin.');}
  const value=await response.json();
  if(!response.ok) throw new Error(value.error || 'Backend request failed');
  return value;
}

async function probeBackend(target) {
  const state=await requestJson('/api/status',undefined,target);
  if(state.app!=='unibrain_agent' || state.api_version!==2) throw new Error('This backend needs UniBrain Agent 0.9 or later. Update it and reconnect.');
  return state;
}

const backendReady=(async()=>{
  if(!backend.address)return false;
  try {backend.status=await probeBackend(backend);backend.connected=true;return true;}
  catch(error){backend.error=error.message;return false;}
})();
