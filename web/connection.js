const appConfig = JSON.parse(document.getElementById('appConfig').textContent);
const demoMode = new URL(location.href || location.origin).searchParams.get('demo') === '1';
const backendPreference = 'unibrainBackendV1';
const backendTokenKey = address => 'unibrainAccessV1:' + address;
let aiSession = demoMode ? 'simulated-demo-session' : '', aiSessionProvider = demoMode ? 'demo' : '';
function normalizeBackend(value) {
  const url = new URL(value);
  if (url.username || url.password || url.search || url.hash || (url.protocol !== 'https:' && !(url.protocol === 'http:' && ['127.0.0.1','localhost','[::1]'].includes(url.hostname)))) throw new Error('Use an HTTPS backend URL, or HTTP on localhost. Do not include credentials or query parameters.');
  return url.href.replace(/\/+$/, '');
}
let savedBackend = '';
try {if (!demoMode) savedBackend = localStorage.getItem(backendPreference) || '';} catch (_) {}
const backend = {address:demoMode?'https://demo.invalid':savedBackend==='disconnected'?'':savedBackend||(!appConfig.standalone&&location.protocol!=='file:'?location.origin:''),token:'',connected:false,error:'',status:null};
try {
  if (!demoMode && backend.address) {backend.address=normalizeBackend(backend.address);backend.token=sessionStorage.getItem(backendTokenKey(backend.address)) || '';}
} catch (_) {backend.address='';}
if (appConfig.standalone && !demoMode) {
  const manualConnect = sessionStorage.getItem('unibrainManualConnect');
  sessionStorage.removeItem('unibrainManualConnect');
  if (manualConnect !== backend.address) {backend.address='';backend.token='';}
}

if (demoMode && typeof DemoBackend !== 'undefined') DemoBackend.configure(appConfig);
const demoToggle = document.getElementById('demoToggle');
const demoBadge = document.getElementById('demoBadge');
if (demoBadge) demoBadge.hidden = !demoMode;
if (demoToggle?.addEventListener) {
  demoToggle.textContent = demoMode ? 'Exit demo' : 'Try demo';
  demoToggle.setAttribute('aria-pressed', String(demoMode));
  demoToggle.addEventListener('click', () => {
    const url = new URL(location.href);
    if (demoMode) url.searchParams.delete('demo'); else url.searchParams.set('demo', '1');
    url.searchParams.delete('experiment');
    if (!url.hash) url.hash = 'workspace';
    location.assign(url.href);
  });
}

async function backendFetch(path, options = {}, target = backend) {
  if (!path.startsWith('/api/')) throw new Error('Invalid backend route');
  if (demoMode) {
    if (typeof DemoBackend === 'undefined') throw new Error('Demo data unavailable. Reload this page.');
    return DemoBackend.fetch(path, options);
  }
  if (!target.address) throw new Error('Connect your backend in AI settings to use this feature.');
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
