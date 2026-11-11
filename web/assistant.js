"use strict";

let assistantConfig = null;
let researchJob = null;
let rolesJob = null;
let assistantBusy = false;
let assistantPoll = null;
let assistantLibrary = null;
const examples = {
  registration: "Compare VoxelMorph and LapIRN for replacing the registration stage. Explain their mechanisms, gradient paths, compute cost and the smallest useful experiment.",
  extraction: "Why can a trainable skull stripping network still break end-to-end gradients? Use SynthStrip as an example and recommend the connection checks.",
  classification: "Design a small experiment to understand whether classification gradients reach the upstream sMRI modules. What should each training role inspect?"
};

function openWorkspace(view, target) {
  document.querySelectorAll("[data-workspace]").forEach(node => node.hidden = node.dataset.workspace !== view);
  document.querySelectorAll(".sidebar nav a").forEach(link => {
    const selected = link.dataset.workspaceLink === view || view === "pipeline" && link.getAttribute("href") === "#builderPanel";
    link.classList.toggle("active", selected);
    if (selected) link.setAttribute("aria-current", "page"); else link.removeAttribute("aria-current");
  });
  byId("snapshotLabel").hidden = view !== "pipeline";
  history.replaceState(null, "", view === "pipeline" ? "#workspace" : "#" + view);
  (target ? byId(target) : byId(view + "View"))?.scrollIntoView({ block: "start" });
}

function assistantStatus(mode, record) {
  const node = byId(mode === "research" ? "researchProgress" : "rolesProgress");
  const busy = record?.status === "running";
  node.hidden = !record || record.status === "complete";
  node.classList.toggle("assistant-error", record?.status === "failed");
  const message = record?.error || record?.events?.at(-1)?.message || "Starting the assistant…";
  node.innerHTML = (busy ? '<span class="spinner"></span>' : '<span>!</span>') + '<span>' + escapeHtml(message) + '</span>';
  byId("sendResearch").disabled = assistantBusy;
  byId("newResearch").disabled = assistantBusy;
  byId("runAllRoles").disabled = assistantBusy;
  document.querySelectorAll("[data-run-role]").forEach(button => button.disabled = assistantBusy);
}

function sourceCards(sources, ids, label = 'Sources') {
  const chosen = sources.filter(source => !ids || ids.includes(source.source_id));
  return chosen.length ? '<details class="source-disclosure"><summary>' + escapeHtml(label) + ' · ' + chosen.length + '</summary>' + chosen.map(source =>
    '<article class="source-card"><span class="source-tag">' + escapeHtml(source.source_id) + ' · ' + escapeHtml(source.origin) + '</span><strong>' + escapeHtml(source.publication) + '</strong><p>' + escapeHtml(source.verification) + '</p><div>' + sourceLink(source.paper, 'Paper ↗') + (source.code ? ' · ' + sourceLink(source.code, 'Code ↗') : '') + '</div></article>').join("") + '</details>' : "";
}

function responseBody(result, sources, actionable) {
  const cited = [...new Set([...result.source_ids, ...result.recommendations.map(item => item.source_id)])];
  let body = '<div class="answer-text">' + escapeHtml(result.answer) + '</div>';
  body += result.recommendations.map(item => {
    const source = sources.find(source => source.source_id === item.source_id);
    const stage = assistantLibrary.stages.find(stage => stage.id === item.stage);
    const option = stage?.options.find(option => option.id === item.tool_id);
    const usable = actionable && option && option.status !== "requires_inputs";
    return '<article class="recommendation"><span class="source-tag">' + escapeHtml(stage?.name || item.stage) + ' · ' + escapeHtml(item.source_id) + '</span><strong>' + escapeHtml(source?.publication || item.tool_id) + '</strong><p>' + escapeHtml(item.reason) + '</p><p class="verification-needed">Check next: ' + escapeHtml(item.verification_needed) + '</p>' + (usable ? '<button class="text-button" data-recommend-stage="' + item.stage + '" data-recommend-tool="' + item.tool_id + '">Use in pipeline ↗</button>' : '<small>' + (item.tool_id ? 'Requires pipeline validation' : 'Literature lead · code inspection required') + '</small>') + '</article>';
  }).join("");
  if (result.next_question) body += '<p class="followup-question">' + escapeHtml(result.next_question) + '</p>';
  return body + sourceCards(sources, cited) + sourceCards(sources.filter(source => source.origin === 'Crossref live metadata' && !cited.includes(source.source_id)),null,'Other retrieved papers');
}

function showResearch(record) {
  researchJob = record;
  const node = byId("researchConversation");
  if (!record) return;
  const history = record.history.map(turn => '<article class="chat-message user-message">' + escapeHtml(turn.question) + '</article><article class="chat-message assistant-message"><span class="source-tag">Researcher</span><div class="answer-text">' + escapeHtml(turn.answer) + '</div>' + sourceCards(turn.sources || [],turn.source_ids) + '</article>').join("");
  node.innerHTML = history + '<article class="chat-message user-message">' + escapeHtml(record.message) + '</article>' + (record.result ? '<article class="chat-message assistant-message"><span class="source-tag">Researcher · ' + (record.search?.status === "complete" ? 'catalog + live metadata' : 'project catalog') + '</span>' + responseBody(record.result, record.sources, true) + '</article>' : "");
  if (record.search?.status === "failed") node.insertAdjacentHTML("beforeend", '<p class="retrieval-warning">Online search was unavailable. This reply uses the project catalog only.</p>');
  node.querySelectorAll("[data-recommend-stage]").forEach(button => button.onclick = () => {
    if (buildBusy) { showDetails("Pipeline is busy", '<p>Wait for the running experiment before changing its tools.</p>'); return; }
    openWorkspace("pipeline");
    selectTool(button.dataset.recommendStage, button.dataset.recommendTool);
  });
  assistantStatus("research", record);
}

function renderRoleCards(record) {
  if (!assistantConfig) return;
  const decisions = (record?.events || []).filter(event => event.kind === "decision");
  const running = record?.status === "running" ? record.events.filter(event => event.kind === "agent").at(-1)?.role : null;
  byId("roleCards").innerHTML = assistantConfig.roles.map(role => {
    const done = decisions.some(event => event.role === role.id);
    const active = running === role.id && !done;
    return '<article class="role-card ' + (active ? 'role-active' : '') + '"><div class="role-card-heading"><strong>' + escapeHtml(role.name) + '</strong><span>' + (active ? 'Working…' : done ? 'Reviewed' : 'Ready') + '</span></div><p>' + escapeHtml(role.purpose) + '</p><div><button class="text-button" data-prompt-role="' + role.id + '">View prompt</button><button class="text-button" data-run-role="' + role.id + '"' + (assistantBusy ? ' disabled' : '') + '>Run role ↗</button></div></article>';
  }).join("");
  document.querySelectorAll("[data-prompt-role]").forEach(button => button.onclick = () => {
    const role = assistantConfig.roles.find(role => role.id === button.dataset.promptRole);
    showDetails(role.name + ' prompt', '<pre class="prompt-content">' + escapeHtml(role.prompt) + '</pre><p class="disclosure-note">Versioned in the repository prompts directory. Each role receives the same measured context.</p>', 'TRAINING ROLE');
  });
  document.querySelectorAll("[data-run-role]").forEach(button => button.onclick = () => submitAssistant("roles", button.dataset.runRole));
}

function showRoles(record) {
  rolesJob = record;
  renderRoleCards(record);
  assistantStatus("roles", record);
  const context = record?.context || assistantConfig?.context;
  const measured = !!context?.diagnostic?.task_norms;
  byId("roleEvidenceTitle").textContent = context?.run_id ? 'Run ' + context.run_id : 'No completed run yet';
  byId("roleEvidenceDescription").textContent = measured ? 'Measured task norms, loss paths and shared gradient cosines are available.' : 'Per-task gradient measurements are unavailable. Roles can identify the next checks; weight balancing is blocked.';
  if (!record) return;
  const reviews = record.result?.roles || record.events.filter(event => event.kind === "decision" && event.role !== "coordinator").map(event => ({role:event.role,...event.decision}));
  let content = reviews.map(review => '<details class="disclosure role-response"><summary>' + escapeHtml(titleCase(review.role)) + ' perspective</summary><div><p>' + escapeHtml(review.summary) + '</p><ul>' + review.evidence.map(item => '<li>' + escapeHtml(item) + '</li>').join("") + '</ul><strong>' + escapeHtml(review.recommendation) + '</strong><p class="disclosure-note">' + escapeHtml(review.uncertainty) + '</p></div></details>').join("");
  if (record.result) {
    const result = record.result;
    if (record.role === 'all' || record.role === 'coordinator') content += '<section class="panel coordinator-decision"><span class="source-tag">Coordinator · advisory only</span><h2>' + escapeHtml(titleCase(result.action)) + '</h2><p>' + escapeHtml(result.rationale) + '</p>' + table(['Task','Current weight','Proposed weight'], tasks.map(task => [titleCase(task),format(result.weights_before[task]),format(result.weights_proposed[task])])) + '<small>No weights were changed by this review.</small></section>';
    else content += '<p class="disclosure-note">Single role review completed. Run all roles to include the coordinator.</p>';
  }
  byId("roleReview").innerHTML = content;
}

async function pollAssistant(id, mode) {
  clearTimeout(assistantPoll);
  try {
    const record = await requestJson('/api/assistant/jobs/' + id);
    assistantBusy = record.status === 'running';
    mode === 'research' ? showResearch(record) : showRoles(record);
    assistantStatus(mode, record);
    if (assistantBusy) assistantPoll = setTimeout(() => pollAssistant(id, mode), 1000);
  } catch (error) {
    assistantBusy = false;
    assistantStatus(mode, {status:'failed',error:error.message});
  }
}

async function submitAssistant(mode, role = 'all') {
  if (!aiSession) {openWorkspace("settings");aiNotice("Connect your own AI account before starting an agent.");return;}
  if (assistantBusy) return;
  if (!backend.connected) {openWorkspace("settings");return;}
  const message = mode === 'research' ? byId('researchInput').value.trim() : 'Review the available task evidence and recommend the next training action.';
  if (!message) return;
  assistantBusy = true;
  const pending = {mode,status:'running',message,events:[],history:researchJob?.history || []};
  if (mode === 'research') assistantStatus(mode,pending);
  else showRoles(pending);
  try {
    const result = await requestJson('/api/assistant', {mode,message,role,stage:byId('researchStage').value,
      online:byId('onlineResearch').checked,selection:draft || assistantLibrary.defaults,
      previous:mode === 'research' ? (researchJob?.status === 'complete' ? researchJob.id : researchJob?.previous) : null});
    if (mode === 'research') byId('researchInput').value = '';
    pollAssistant(result.id,mode);
  } catch (error) { assistantBusy = false; assistantStatus(mode,{status:'failed',error:error.message}); }
}

function updateTrainingTeam(value) {
  const roles = (value.agents || []).filter(item => tasks.includes(item.role) || item.role === 'coordinator');
  if (!roles.length) return;
  byId('trainingTeamLive').hidden = false;
  byId('trainingTeamLive').innerHTML = '<h2>Roles in the training run</h2><p class="disclosure-note">These calls belong to the run monitor, separate from advisory reviews.</p>' + roles.map(role => '<details class="disclosure"><summary>' + escapeHtml(titleCase(role.role)) + '<span>' + escapeHtml(role.status) + '</span></summary><p class="answer-text">' + escapeHtml(role.decision?.rationale || role.decision?.summary || 'Reading measured gradients…') + '</p></details>').join('');
}

async function initializeAssistant() {
  document.querySelectorAll('.sidebar nav a, .sidebar .brand').forEach(link => link.addEventListener('click', event => {
    event.preventDefault();
    openWorkspace(link.dataset.workspaceLink || 'pipeline', link.dataset.workspaceLink ? null : link.getAttribute('href').slice(1));
  }));
  byId('resetPipeline').addEventListener('click', () => openWorkspace('pipeline'));
  byId('openTrainingRoles').onclick = () => {openWorkspace('roles');document.querySelector('.prompt-review').open=true;};
  byId('researchForm').onsubmit = event => { event.preventDefault(); submitAssistant('research'); };
  byId('researchInput').onkeydown = event => { if (event.key === 'Enter' && !event.shiftKey) {event.preventDefault();submitAssistant('research');} };
  byId('newResearch').onclick = () => { researchJob=null;byId('researchConversation').innerHTML='<div class="conversation-empty">Start a new research conversation.</div>';byId('researchProgress').hidden=true;byId('researchInput').focus(); };
  document.querySelectorAll('[data-example]').forEach(button => button.onclick = () => {byId('researchStage').value=button.dataset.example;byId('researchInput').value=examples[button.dataset.example];byId('researchInput').focus();});
  byId('runAllRoles').onclick = () => submitAssistant('roles');
  byId('researchPrompt').onclick = () => showDetails('Researcher prompt','<pre class="prompt-content">'+escapeHtml(assistantConfig?.research_prompt || 'Loading…')+'</pre>');
  byId('inspectRoleEvidence').onclick = () => showDetails('Measured context','<pre class="prompt-content">'+escapeHtml(JSON.stringify(rolesJob?.context || assistantConfig?.context,null,2))+'</pre>','AGGREGATE MEASUREMENTS');
  window.addEventListener('trainingtelemetry', event => updateTrainingTeam(event.detail));
  const connected = await backendReady;
  openWorkspace(['#data','#research','#roles','#settings'].includes(location.hash) ? location.hash.slice(1) : !connected && appConfig.standalone ? 'settings' : 'pipeline');
  if (!connected) {
    assistantConfig = {roles:appConfig.roles,research_prompt:appConfig.research_prompt,context:null};
    assistantLibrary = appConfig.catalog;showRoles(null);
    byId('researchStage').innerHTML += assistantLibrary.stages.map(stage => '<option value="'+stage.id+'">'+escapeHtml(stage.name)+'</option>').join('');
    return;
  }
  try {
    [assistantConfig,assistantLibrary] = await Promise.all([requestJson('/api/assistant'),requestJson('/api/catalog')]);
    byId('researchStage').innerHTML += assistantLibrary.stages.map(stage => '<option value="'+stage.id+'">'+escapeHtml(stage.name)+'</option>').join('');
    if (assistantConfig.latest.research) {
      byId('researchStage').value=assistantConfig.latest.research.stage;
      byId('onlineResearch').checked=assistantConfig.latest.research.online;
      showResearch(assistantConfig.latest.research);
    }
    showRoles(assistantConfig.latest.roles || null);
    const pending=Object.values(assistantConfig.latest).find(job=>job.status==='running');
    if (pending) {assistantBusy=true;pollAssistant(pending.id,pending.mode);}
    if (liveRun) updateTrainingTeam(liveRun);
  } catch(error) {assistantStatus('research',{status:'failed',error:error.message});}
}

initializeAssistant();
