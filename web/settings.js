let aiConfig = null, aiBusy = false;

function aiNotice(message, busy = false, error = false) {
  const node = byId('aiFeedback');
  node.hidden = !message;
  node.classList.toggle('assistant-error', error);
  node.innerHTML = (busy ? '<span class="spinner"></span>' : '') + '<span>' + escapeHtml(message) + '</span>';
}

function aiControls(busy) {
  aiBusy = busy;
  ['aiSave','aiTest','aiProvider','aiRefreshLogin'].forEach(id => byId(id).disabled = busy);
}

function renderAIProfile() {
  const provider = byId('aiProvider').value;
  const item = aiConfig.providers.find(item => item.id === provider);
  const profile = aiConfig.profiles[provider];
  const cli = ['codex','claude'].includes(provider);
  byId('aiCliDetails').hidden = !cli;
  byId('aiApiFields').hidden = cli;
  byId('aiModel').value = profile.model;
  byId('aiModel').required = !cli;
  byId('aiModel').placeholder = cli ? 'Leave empty to use the CLI default' : 'Enter a model ID from your provider';
  byId('aiBaseUrl').value = profile.base_url;
  byId('aiBaseUrl').readOnly = provider !== 'compatible';
  byId('aiBaseUrl').required = !cli;
  byId('aiKey').value = '';
  byId('aiClearKey').checked = false;
  byId('aiLoginTitle').textContent = item.name + ' account';
  byId('aiLoginStatus').textContent = item.installed ? 'CLI detected. Check login to inspect authentication.' : 'CLI is not installed on the backend machine.';
  byId('aiKeyStatus').textContent = 'Credential: ' + item.credential + '. Environment variable: ' + item.env + '.';
}

function renderAI(config) {
  aiConfig = config;
  byId('aiProvider').innerHTML = config.providers.map(item => '<option value="' + item.id + '">' + escapeHtml(item.name) + '</option>').join('');
  byId('aiProvider').value = config.provider;
  const selected = config.providers.find(item => item.id === config.provider);
  byId('aiCurrent').textContent = 'Active connection: ' + selected.name + ' · ' + (config.profiles[config.provider].model || 'CLI default model');
  renderAIProfile();
}

async function refreshAILogin() {
  byId('aiRefreshLogin').disabled = true;
  byId('aiLoginStatus').textContent = 'Checking backend login…';
  try { const value = await requestJson('/api/ai/login?provider=' + byId('aiProvider').value);byId('aiLoginStatus').textContent = value.message; }
  catch (error) { byId('aiLoginStatus').textContent = error.message; }
  finally { byId('aiRefreshLogin').disabled = aiBusy; }
}

async function pollAITest() {
  try {
    const state = await requestJson('/api/ai');
    if (state.test.status === 'running') { setTimeout(pollAITest, 1200);return; }
    aiControls(false);
    aiNotice(state.test.message + (state.test.seconds ? ' (' + state.test.seconds + 's)' : ''), false, state.test.status === 'failed');
  } catch(error) {aiControls(false);aiNotice(error.message,false,true);}
}

async function initializeAISettings() {
  byId('aiProvider').onchange = () => {renderAIProfile();aiNotice('Save this connection before testing or running an agent.');};
  byId('aiRefreshLogin').onclick = refreshAILogin;
  byId('aiLoginHelp').onclick = () => {
    const item = aiConfig.providers.find(item => item.id === byId('aiProvider').value);
    const docs = item.id === 'codex' ? 'https://learn.chatgpt.com/docs/auth?surface=cli' : 'https://code.claude.com/docs/en/authentication';
    showDetails('Connect ' + item.name, '<p>Run this command in a terminal on the machine hosting the Python backend. Complete the official sign-in yourself, then return here and select Check login.</p><pre class="prompt-content">' + item.command + '</pre>' + (item.id === 'codex' ? '<p>For a headless server, device sign-in is available through:</p><pre class="prompt-content">codex login --device-auth</pre>' : '') + '<p>Your CLI account is used by this private workspace. Changing its login can also affect other local CLI sessions.</p><a href="' + docs + '" target="_blank" rel="noopener">Official authentication guide ↗</a>', 'YOUR BACKEND ACCOUNT');
  };
  byId('aiSettingsForm').onsubmit = async event => {
    event.preventDefault();if (aiBusy) return;
    aiControls(true);aiNotice('Saving connection…',true);
    try {
      const state = await requestJson('/api/ai/settings',{provider:byId('aiProvider').value,model:byId('aiModel').value,
        base_url:byId('aiBaseUrl').value,api_key:byId('aiKey').value,clear_key:byId('aiClearKey').checked});
      byId('aiKey').value = '';renderAI(state);aiNotice('Saved. New research, role reviews and custom builds use this connection.');
    } catch(error) {byId('aiKey').value='';aiNotice(error.message,false,true);}
    finally {aiControls(false);}
  };
  byId('aiTest').onclick = async () => {
    if (aiBusy) return;
    if (byId('aiProvider').value !== aiConfig.provider || byId('aiModel').value.trim() !== aiConfig.profiles[aiConfig.provider].model || byId('aiBaseUrl').value.replace(/\/+$/,'') !== aiConfig.profiles[aiConfig.provider].base_url || byId('aiKey').value || byId('aiClearKey').checked) {aiNotice('Save your changes before testing the connection.');return;}
    aiControls(true);aiNotice('Calling the saved model connection…',true);
    try {await requestJson('/api/ai/test',{});pollAITest();}
    catch(error) {aiControls(false);aiNotice(error.message,false,true);}
  };
  if (location.protocol === 'file:') {aiNotice('Start the Python backend to configure an AI connection.');return;}
  try {
    const config=await requestJson('/api/ai');renderAI(config);
    if(config.test.status==='running') {aiControls(true);aiNotice('Connection test is running…',true);pollAITest();}
  } catch(error) {aiNotice(error.message,false,true);}
}

initializeAISettings();
