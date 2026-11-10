let aiConfig=null, aiBusy=false, selectedProvider='codex';
const accountStates={};
const cliProvider=id=>['codex','claude'].includes(id);

function aiNotice(message,busy=false,error=false) {
  const node=byId('aiFeedback');node.hidden=!message;node.classList.toggle('assistant-error',error);
  node.innerHTML=(busy?'<span class="spinner"></span>':'')+'<span>'+escapeHtml(message)+'</span>';
}

function aiControls(busy) {
  aiBusy=busy;
  ['aiSave','aiTest','aiRefreshLogin','aiLogout'].forEach(id=>byId(id).disabled=busy||!backend.connected||!aiConfig);
  document.querySelectorAll('[data-provider]').forEach(node=>node.disabled=busy);
}

function paintProviderCards() {
  document.querySelectorAll('[data-provider]').forEach(node=>{
    const id=node.dataset.provider, item=aiConfig?.providers.find(item=>item.id===id), state=accountStates[id];
    const selected=id===selectedProvider;
    node.classList.toggle('selected',selected);node.setAttribute('aria-checked',String(selected));
    const status=node.querySelector('.provider-state');
    const verified=aiConfig?.test?.provider===id&&aiConfig.test.status==='complete';
    status.textContent=!backend.connected?'Connect backend':cliProvider(id)?state?.authenticated?'Signed in':item?.installed===false?'Not installed':state?.authenticated===false?'Signed out':state?.authenticated===null?'Unavailable':'Checking…':verified?'Connected':item?.credential!=='not supplied'?'Key configured':'Add API key';
    status.classList.toggle('online',!!state?.authenticated||verified);
  });
}

function paintAccount() {
  const id=selectedProvider,item=aiConfig?.providers.find(item=>item.id===id),state=accountStates[id];
  const cli=cliProvider(id),signedIn=!!state?.authenticated;
  let name='Connect a backend',plan='Your account is loaded from the selected server.',badge='Not connected',note='Configure the compute backend above to use AI services.';
  if(backend.connected&&item) {
    if(cli) {
      badge=signedIn?'Signed in':state?.authenticated===null?'Status unavailable':item.installed===false?'Not installed':state?'Signed out':'Checking account';
      name=signedIn?(state.email||state.name||'Connected CLI account'):item.installed===false?'Install '+item.name:'Connect '+item.name;
      plan=signedIn?[state.plan?titleCase(state.plan):null,state.method==='chatgpt'?'ChatGPT':state.method?titleCase(state.method):null].filter(Boolean).join(' · ')||'Official CLI session':'Sign in on the machine running this backend.';
      note=state?.message||'Reading account details from the official CLI…';
    } else {
      const ready=item.credential!=='not supplied';
      badge=ready?'Key configured':'Key needed';name=item.name;plan=ready?'Credential source: '+item.credential:'Bring your own API credentials';
      note='API keys do not expose an account email here. Use your provider dashboard to manage the key and billing.';
    }
  }
  byId('accountBadge').textContent=badge;byId('accountBadge').classList.toggle('connected',signedIn||(!cli&&item?.credential!=='not supplied'&&backend.connected));
  byId('accountName').textContent=name;byId('accountPlan').textContent=plan;byId('aiLoginStatus').textContent=note;
  byId('accountAvatar').textContent=signedIn?(name.slice(0,2).toUpperCase()):cli?(id==='codex'?'C':'A'):'API';
  byId('aiLoginHelp').hidden=!cli;byId('aiLoginHelp').textContent=signedIn?'Switch account ↗':'Sign in ↗';
  byId('aiLogout').hidden=!cli||!signedIn;byId('aiRefreshLogin').hidden=!cli;
  paintProviderCards();
}

function renderAIProfile() {
  const id=selectedProvider,cli=cliProvider(id),profile=aiConfig?.profiles[id]||{model:'',base_url:''},item=aiConfig?.providers.find(item=>item.id===id);
  byId('aiApiFields').hidden=cli;byId('aiModel').value=profile.model;byId('aiModel').required=!cli;
  byId('aiModel').placeholder=cli?'Default model from your CLI':'Model ID from your provider';
  byId('modelHint').textContent=cli?'Leave empty to follow your CLI default.':'Choose a model that supports structured JSON output.';
  byId('aiBaseUrl').value=profile.base_url;byId('aiBaseUrl').readOnly=id!=='compatible';byId('aiBaseUrl').required=!cli;
  byId('aiKey').value='';byId('aiClearKey').checked=false;
  byId('aiKeyStatus').textContent=item?'Credential: '+item.credential+'. Backend environment: '+item.env+'. Keys entered here stay in backend memory until restart.':'';
  paintAccount();aiControls(aiBusy);
}

async function refreshAccount(id=selectedProvider) {
  if(!backend.connected||!cliProvider(id))return;
  try {accountStates[id]=await requestJson('/api/ai/login?provider='+id);}
  catch(error){accountStates[id]={authenticated:null,message:error.message};}
  paintAccount();
}

function renderAI(config,select=true) {
  aiConfig=config;if(select)selectedProvider=config.provider;
  renderAIProfile();
}

function signInHelp() {
  const id=selectedProvider,item=aiConfig?.providers.find(item=>item.id===id);
  if(!item){aiNotice('Connect your backend first.');return;}
  const docs=id==='codex'?'https://learn.chatgpt.com/docs/auth?surface=cli':'https://code.claude.com/docs/en/authentication';
  showDetails('Connect '+item.name,'<p>On the machine at <strong>'+escapeHtml(backend.address)+'</strong>, run the official sign-in command and complete authentication yourself.</p><pre class="prompt-content">'+escapeHtml(item.command)+'</pre>'+(id==='codex'?'<p>For a headless GPU server:</p><pre class="prompt-content">codex login --device-auth</pre>':'')+'<p>To switch accounts, sign out of the current account first. Return here and refresh after signing in. This workspace shares the backend operating system user’s CLI login.</p><a target="_blank" rel="noopener" href="'+docs+'">Official sign-in guide ↗</a>','BACKEND ACCOUNT');
}

function confirmSignOut() {
  const id=selectedProvider,account=accountStates[id];
  showDetails('Sign out of '+(id==='codex'?'Codex':'Claude Code')+'?','<p>This signs <strong>'+escapeHtml(account?.email||'the connected account')+'</strong> out of the CLI on <strong>'+escapeHtml(backend.address)+'</strong>. Other CLI sessions using that operating system account may also require sign-in.</p><div class="ai-buttons"><button id="confirmSignOut" class="primary">Sign out of backend account</button><button id="cancelSignOut" class="text-button">Keep account</button></div>','ACCOUNT ACCESS');
  byId('cancelSignOut').onclick=()=>byId('detailDialog').close();
  byId('confirmSignOut').onclick=async()=>{
    byId('detailDialog').close();aiControls(true);aiNotice('Signing out…',true);
    try {accountStates[id]=await requestJson('/api/ai/logout',{provider:id,confirm:true});paintAccount();aiNotice(accountStates[id].authenticated?'A managed credential is still active on this backend.':'Signed out of the backend account.');}
    catch(error){aiNotice(error.message,false,true);}
    finally{aiControls(false);}
  };
}

async function pollAITest() {
  try {
    const state=await requestJson('/api/ai');
    if(state.test.status==='running'){setTimeout(pollAITest,1200);return;}
    aiConfig=state;aiControls(false);paintProviderCards();
    aiNotice(state.test.message+(state.test.seconds?' ('+state.test.seconds+'s)':''),false,state.test.status==='failed');
  } catch(error){aiControls(false);aiNotice(error.message,false,true);}
}

async function initializeAISettings() {
  document.querySelectorAll('[data-provider]').forEach(node=>{
    node.onclick=()=>{selectedProvider=node.dataset.provider;renderAIProfile();aiNotice('');if(cliProvider(selectedProvider))refreshAccount();};
    node.onkeydown=event=>{
      if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key))return;
      event.preventDefault();const cards=[...document.querySelectorAll('[data-provider]')],index=cards.indexOf(node),next=cards[(index+(['ArrowLeft','ArrowUp'].includes(event.key)?cards.length-1:1))%cards.length];next.focus();next.click();
    };
  });
  byId('aiRefreshLogin').onclick=()=>refreshAccount();byId('aiLoginHelp').onclick=signInHelp;byId('aiLogout').onclick=confirmSignOut;
  byId('editBackend').onclick=()=>{byId('backendForm').hidden=!byId('backendForm').hidden;};
  byId('backendUrl').value=backend.address;byId('backendToken').value='';
  byId('backendForm').onsubmit=async event=>{
    event.preventDefault();byId('connectBackend').disabled=true;byId('backendFeedback').textContent='Connecting to your backend…';
    try {
      const address=normalizeBackend(byId('backendUrl').value.trim());
      const token=byId('backendToken').value.trim()||(address===backend.address?backend.token:'');
      await probeBackend({address,token});
      localStorage.setItem(backendPreference,address);sessionStorage.setItem(backendTokenKey(address),token);
      byId('backendToken').value='';location.hash='settings';location.reload();
    } catch(error){byId('backendFeedback').textContent=error.message;byId('connectBackend').disabled=false;}
  };
  byId('disconnectBackend').onclick=()=>{
    if(backend.address)sessionStorage.removeItem(backendTokenKey(backend.address));
    localStorage.setItem(backendPreference,'disconnected');location.hash='settings';location.reload();
  };
  byId('deploymentHelp').onclick=()=>showDetails('Connect your GPU server','<p>Run the Python backend on your own machine. Set a random UNIBRAIN_ACCESS_TOKEN and allow this frontend origin:</p><pre class="prompt-content">'+escapeHtml(location.origin)+'</pre><p>Remote servers need HTTPS, usually through a reverse proxy. Your backend keeps datasets, checkpoints and AI credentials. The GitHub Pages frontend contains only the application and public method catalog.</p><p>This is one private workspace per backend. People sharing its access token share the workspace and its CLI account.</p><a target="_blank" rel="noopener" href="https://github.com/soz223/UniBrain-Agent/blob/main/docs/deployment.md">Setup commands and deployment guide ↗</a>','FRONTEND + COMPUTE BACKEND');
  byId('aiSettingsForm').onsubmit=async event=>{
    event.preventDefault();if(aiBusy||!aiConfig)return;aiControls(true);aiNotice('Saving connection…',true);
    try {
      const value=await requestJson('/api/ai/settings',{provider:selectedProvider,model:byId('aiModel').value,base_url:byId('aiBaseUrl').value,api_key:byId('aiKey').value,clear_key:byId('aiClearKey').checked});
      byId('aiKey').value='';renderAI(value);aiNotice('Saved. New agent requests use this connection.');
    } catch(error){byId('aiKey').value='';aiNotice(error.message,false,true);}
    finally{aiControls(false);}
  };
  byId('aiTest').onclick=async()=>{
    if(aiBusy||!aiConfig)return;
    const saved=aiConfig.profiles[aiConfig.provider];
    if(selectedProvider!==aiConfig.provider||byId('aiModel').value.trim()!==saved.model||byId('aiBaseUrl').value.replace(/\/+$/,'')!==saved.base_url||byId('aiKey').value||byId('aiClearKey').checked){aiNotice('Save your changes before testing.');return;}
    aiControls(true);aiNotice('Calling your model…',true);
    try{await requestJson('/api/ai/test',{});pollAITest();}catch(error){aiControls(false);aiNotice(error.message,false,true);}
  };
  const connected=await backendReady;
  byId('aiCurrent').textContent=connected?'Backend connected':'Backend offline';byId('aiCurrent').classList.toggle('connected',connected);
  byId('backendLabel').textContent=connected?(new URL(backend.address).hostname==='127.0.0.1'||new URL(backend.address).hostname==='localhost'?'Local workstation':'Remote compute server'):'Connect your server';
  byId('backendAddress').textContent=backend.address||'Local workstation or remote GPU server';
  byId('backendForm').hidden=connected;byId('backendFeedback').textContent=backend.error;byId('disconnectBackend').hidden=!backend.address;
  if(!connected){renderAIProfile();return;}
  try {
    const config=await requestJson('/api/ai');renderAI(config);
    Promise.allSettled(['codex','claude'].map(refreshAccount));
    if(config.test.status==='running'){aiControls(true);aiNotice('Connection test is running…',true);pollAITest();}
  } catch(error){aiNotice(error.message,false,true);}
}

initializeAISettings();
