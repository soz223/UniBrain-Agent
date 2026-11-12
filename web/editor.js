"use strict";

let library = null;
let draft = null;
let activeBuild = null;
let buildBusy = false;
let buildPoll = null;
let activityStarted = null;
const mainStages = ["denoising", "bias", "extraction", "normalization", "registration", "parcellation", "graph", "classification"];
const stageNames = { extraction: "Extraction", normalization: "Normalization", parcellation: "Parcellation", graph: "Network generation" };
const draftKey = "unibrainDraftV3:" + backend.address;
const buildKey = "unibrainBuildV3:" + backend.address;

function isOriginal() {
  return library && draft && library.stages.every(stage => draft[stage.id] === library.defaults[stage.id]);
}

function selectedCoordination() {
  return byId("teamTraining").checked ? (isOriginal() ? "taskforce" : "roles") : "single";
}

function chosenOption(stage, selection = draft) {
  return stage.options.find(option => option.id === selection[stage.id]);
}

function toolName(option) {
  return option.method || { none: "Skip", soft_atlas: "Soft atlas", continuous: "Cosine graph" }[option.id] || option.label.split(" (")[0];
}

function renderChoices() {
  byId("profileLabel").textContent = isOriginal() ? '⚙ Original full UniBrain' : '○ Legacy compact pipeline';
  byId("budgetLabel").textContent = isOriginal() ? "Epochs" : "Steps";
  byId("prepareBuild").textContent = isOriginal() ? "Prepare UniBrain" : "Prepare compact prototype";
  byId("resetPipeline").disabled = buildBusy;
  if (byId("teamTraining")) {
    byId("teamTraining").disabled = buildBusy;
    byId("teamTrainingLabel").textContent = isOriginal() ? "Learned gradient policy" : "Use training roles";
    byId("teamTrainingNote").textContent = isOriginal() ? "TaskForce-style actors learn from losses and task gradients at every update. The original model architecture is retained." : "Four task specialists and a coordinator review measured gradients before update 2.";
  }
  const button = (stage, index) => {
    const option = chosenOption(stage);
    const optional = stage.options.some(item => item.id === "none");
    const skipped = option.id === "none";
    return '<button class="stage-choice ' + (skipped ? "skipped" : "enabled") + '" data-choice-stage="' + stage.id +
      '" aria-label="' + escapeHtml(stage.name + ": " + toolName(option) + ". Change tool") + '"' + (buildBusy ? " disabled" : "") +
      '><span class="stage-top"><span class="stage-number">' + (index === null ? "↳" : String(index + 1).padStart(2, "0")) +
      '</span><strong>' + escapeHtml(stageNames[stage.id] || stage.name) + '</strong>' + (optional ? '<small>optional</small>' : "") +
      '</span><span class="stage-tool"><span>' + (skipped ? "○ " : "✓ ") + escapeHtml(toolName(option)) + (option.id === "unibrain" ? '<small class="method-origin">(UniBrain original)</small>' : "") + '</span><span>⌄</span></span></button>';
  };
  byId("stageChoices").innerHTML = mainStages.map((id, index) => button(library.stages.find(stage => stage.id === id), index)).join("");
  byId("branchChoices").innerHTML = ["segmentation", "surface"].map(id => button(library.stages.find(stage => stage.id === id), null)).join("");
  document.querySelectorAll("[data-choice-stage]").forEach(node => node.onclick = () => chooseTool(node.dataset.choiceStage));
}

function chooseTool(stageId) {
  if (buildBusy) return;
  const stage = library.stages.find(item => item.id === stageId);
  showDetails(stage.name, '<input id="toolSearch" type="search" placeholder="Search tools…" aria-label="Search tools" autocomplete="off">' +
    '<div id="toolOptions" class="tool-options"></div><div id="toolEvidence"></div>', "SELECT ONE TOOL");
  byId("detailDialog").classList.add("tool-dialog");
  const renderOptions = () => {
    const query = byId("toolSearch").value.trim().toLowerCase();
    const options = stage.options.filter(option => (option.label + " " + toolName(option)).toLowerCase().includes(query));
    byId("toolOptions").innerHTML = options.length ? options.map(option => {
      const selected = draft[stageId] === option.id;
      const blocked = option.status === "requires_inputs";
      return '<button class="tool-option ' + (selected ? "selected" : "") + '" data-tool="' + option.id +
        '" aria-pressed="' + selected + '"' + (blocked ? ' disabled title="Requires a mesh and physical image coordinates"' : "") +
        '><span><strong>' + escapeHtml(toolName(option)) + '</strong><small>' + escapeHtml(blocked ? "Unavailable · mesh inputs required" :
          option.id === "none" ? "Bypass this optional step" : option.label) +
        '</small></span><span class="tool-mark">' + (selected ? "✓ Selected" : blocked ? "Unavailable" : "Select") + '</span></button>';
    }).join("") : '<p class="no-tools">No matching tools.</p>';
    byId("toolOptions").querySelectorAll("[data-tool]").forEach(node => node.onclick = () => selectTool(stageId, node.dataset.tool));
  };
  const option = chosenOption(stage);
  byId("toolEvidence").innerHTML = '<details><summary>Current tool: adaptation & sources</summary><p>' + escapeHtml(option.adaptation) + '</p>' +
    (option.research ? '<p>' + escapeHtml(option.research.interface) + '</p><p>' + escapeHtml(option.research.limitations) + '</p>' +
      sourceLink(option.research.paper, "Paper") + ' · ' + sourceLink(option.research.code, "Source") : "") + '</details>';
  byId("toolSearch").oninput = renderOptions;
  byId("toolSearch").onkeydown = event => {
    if (event.key === "ArrowDown") { event.preventDefault(); byId("toolOptions").querySelector("button:not(:disabled)")?.focus(); }
    if (event.key === "Enter") { event.preventDefault(); byId("toolOptions").querySelector("button:not(:disabled)")?.click(); }
  };
  renderOptions();
  byId("toolSearch").focus();
}

function selectTool(stageId, choice) {
  if (draft[stageId] === choice) { byId("detailDialog").close(); return; }
  const wasOriginal = isOriginal();
  draft[stageId] = choice;
  let note = "Selection updated. Apply changes to generate and validate the pipeline.";
  if (choice === "M03") {
    draft.extraction = draft.registration = "M03";
    draft.normalization = "none";
    note = "ERNet links extraction and registration. Normalization is skipped to preserve that connection.";
  } else if (stageId === "extraction" && draft.registration === "M03") draft.registration = "unibrain";
  else if (stageId === "registration" && draft.extraction === "M03") draft.extraction = "unibrain";
  if (wasOriginal && !isOriginal()) {
    if (draft.parcellation === "unibrain") draft.parcellation = "soft_atlas";
    if (draft.graph === "unibrain") draft.graph = "continuous";
    byId("buildSteps").value = "5";
    note = "Custom adapter profile. Soft atlas transfer and continuous graph connections are enabled for gradient checks. Review the generated changes before running.";
  }
  activeBuild = null;
  sessionStorage.removeItem(buildKey);
  sessionStorage.setItem(draftKey, JSON.stringify(draft));
  byId("detailDialog").close();
  renderChoices();
  byId("activityPanel").hidden = true;
  setBuildState("dirty", "Unapplied changes");
  byId("buildMessage").textContent = note;
  byId("confirmBuild").hidden = true;
  byId("prepareBuild").hidden = false;
  byId("viewBuild").disabled = true;
}

function showActivity(record) {
  const original = record.profile === "original" || isOriginal();
  const phases = original ? ["contracts", "audit", "planning", "ready"] : ["contracts", "planning", "training", "audit", "ready"];
  const labels = original ? ["Source files", "CUDA & data", "Original settings", "Ready"] : ["Check tools", "Agent plan", "Train × 2", "Audit", "Ready"];
  const latestPhase = record.events?.filter(event => event.phase).at(-1)?.phase || "contracts";
  const phase = record.status === "ready" ? "ready" : record.progress?.phase || latestPhase;
  const position = phases.indexOf(phase);
  byId("activityPanel").hidden = false;
  byId("activityPanel").classList.toggle("is-working", buildBusy);
  byId("activityPanel").classList.toggle("has-failed", record.status === "failed");
  byId("activitySteps").innerHTML = labels.map((label, i) => '<li class="' +
    (i < position || record.status === "ready" ? "done" : i === position ? record.status === "failed" ? "failed" : "current" : "") +
    '"><span>' + (i < position || record.status === "ready" ? "✓" : i + 1) + '</span>' + label + '</li>').join("");
  const history = [...(record.events || []), ...(record.progress?.events || [])].sort((a, b) => (a.at || "").localeCompare(b.at || ""));
  byId("activityLog").innerHTML = history.map(event => '<div><time>' + (event.at ?
    Math.max(0, Math.round((Date.parse(event.at) - Date.parse(record.created)) / 1000)) + "s" : "") + '</time>' + escapeHtml(event.message) + '</div>').join("");
  byId("activityIcon").className = buildBusy ? "spinner" : "activity-result";
  byId("activityIcon").textContent = buildBusy ? "" : record.status === "ready" ? "✓" : "!";
  activityStarted = buildBusy ? Date.parse(record.created) : null;
  const end = history.at(-1)?.at;
  byId("buildElapsed").textContent = (buildBusy ? Math.floor((Date.now() - Date.parse(record.created)) / 1000) :
    end ? Math.max(0, Math.round((Date.parse(end) - Date.parse(record.created)) / 1000)) : 0) + "s";
}

function showBuild(record) {
  activeBuild = record;
  buildBusy = ["queued", "preparing", "validating"].includes(record.status);
  const matches = library.stages.every(stage => record.selection[stage.id] === draft[stage.id]) &&
    (record.coordination || "single") === selectedCoordination();
  const titles = { queued: "Starting checks", preparing: "Agent is building your pipeline", validating: "Validating the training graph",
    ready: "Ready to run", failed: "Build needs attention" };
  if (record.profile === "original") titles.preparing = "Checking original UniBrain";
  setBuildState(record.status === "ready" && !matches ? "dirty" : record.status === "ready" ? "ready" : record.status === "failed" ? "failed" : "working", record.status === "ready" && !matches ? "Changes need preparation" : titles[record.status] || record.status);
  byId("buildMessage").textContent = record.status === "ready" && !matches ? "Prepare the selected stages and training policy before running." : record.error || (record.status === "ready" ? record.profile === "original" ? "Original sources and CUDA runtime checked. Training verification runs after confirmation." : "Gradient, parameter update and spatial checks passed on one sample." :
    record.progress?.message || record.events?.at(-1)?.message || "Preparing selected stages");
  byId("prepareBuild").disabled = buildBusy;
  byId("prepareBuild").hidden = record.status === "ready" && matches;
  byId("confirmBuild").hidden = record.status !== "ready" || !matches;
  byId("confirmBuild").disabled = buildBusy || record.status !== "ready" || !matches;
  byId("viewBuild").hidden = false;
  byId("viewBuild").disabled = !record.code;
  showActivity(record);
  if (record.status === "ready" && matches && report?.build?.id === record.id && report.status === "complete") {
    byId("activityPanel").hidden = true;
    setBuildState("ready", "Ready for another run");
    byId("buildMessage").textContent = "Last run completed. Adjust the tools or run this pipeline again.";
  }
  renderChoices();
}


function buildError(error) {
  buildBusy = false;
  activityStarted = null;
  setBuildState("failed", "Build needs attention");
  byId("buildMessage").textContent = error.message;

  byId("activityPanel").classList.remove("is-working");
  byId("activityIcon").className = "activity-result";
  byId("activityIcon").textContent = "!";
  byId("prepareBuild").disabled = false;
  byId("prepareBuild").hidden = false;
  byId("confirmBuild").disabled = true;
  renderChoices();
}

async function pollBuild(identifier) {
  clearTimeout(buildPoll);
  try {
    showBuild(await requestJson("/api/builds/" + identifier));
    if (buildBusy) buildPoll = setTimeout(() => pollBuild(identifier), 1000);
  } catch (error) { buildError(error); }
}

async function prepareSelection() {
  if (buildBusy) return;
  if (!backend.connected) {openWorkspace("settings");return;}
  if (!isOriginal() && !aiSession) {openWorkspace("settings");aiNotice("Connect your AI account before generating a custom pipeline.");return;}
  showBuild({ status: "queued", selection: { ...draft }, created: new Date().toISOString(), events: [] });
  try {
    const value = await requestJson("/api/prepare", { selection: draft, backend: "configured", coordination: selectedCoordination() });
    sessionStorage.setItem(buildKey, value.build_id);
    sessionStorage.removeItem(draftKey);
    await pollBuild(value.build_id);
  } catch (error) { buildError(error); }
}

async function reviewBuild() {
  if (!activeBuild?.code) return;
  showDetails('Review Generated Code','<p><span class="spinner"></span> Loading generated code and registered implementations…</p>','BUILD '+activeBuild.id);
  try {
    const review=await requestJson('/api/builds/'+activeBuild.id+'/code');
    const files=[...review.files,{name:'Changes.diff',code:review.diff,kind:'Composition changes',sha256:''}];
    byId('dialogBody').innerHTML='<p>'+escapeHtml(review.scope)+'</p><label class="code-file-label">File<select id="reviewFile" aria-label="Code file">'+files.map((file,i)=>'<option value="'+i+'">'+escapeHtml(file.name)+'</option>').join('')+'</select></label><p id="codeKind" class="muted"></p><pre id="reviewCode" class="code-preview"></pre><button id="downloadCode" class="text-button">Download this file ↧</button>';
    const paint=()=>{const file=files[Number(byId('reviewFile').value)];byId('reviewCode').textContent=file.code;byId('codeKind').textContent=file.kind+(file.sha256?' · SHA256 '+file.sha256.slice(0,12):'');};
    byId('reviewFile').onchange=paint;paint();
    byId('downloadCode').onclick=()=>{const file=files[Number(byId('reviewFile').value)],url=URL.createObjectURL(new Blob([file.code],{type:'text/plain'})),link=document.createElement('a');link.href=url;link.download=file.name.split('/').at(-1);link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
  } catch(error) {byId('dialogBody').innerHTML='<p>'+escapeHtml(error.message)+'</p>';}
}

async function runBuild() {
  if (!activeBuild || activeBuild.status !== "ready" || buildBusy) return;
  if (!byId("buildSteps").reportValidity()) return;
  if (activeBuild.profile!=="original" && !aiSession) {openWorkspace("settings");aiNotice("Connect your AI account before running a custom pipeline.");return;}
  buildBusy = true;
  byId("confirmBuild").disabled = true;
  byId("prepareBuild").disabled = true;
  byId("buildSteps").disabled = true;
  byId("confirmBuild").textContent = "Running…";
  setBuildState("working", "Running, please wait");
  byId("buildMessage").textContent = "Your selected pipeline is starting. Follow its progress below.";
  startLiveRun(Number(byId("buildSteps").value));
  renderChoices();
  try {
    await requestJson("/api/builds/" + activeBuild.id + "/run", { steps: Number(byId("buildSteps").value), backend: activeBuild.backend,
      coordination: selectedCoordination() });
    poll();
  } catch (error) { setFailure(error); }
}

function setBuildState(state, label) {
  const node=byId("buildStatus");
  node.className="build-state "+state;
  node.innerHTML=(state==='working'?'<span class="spinner"></span>': '<span aria-hidden="true">'+({dirty:'●',ready:'✓',failed:'!',idle:'○'}[state]||'○')+'</span>')+' '+escapeHtml(label);
}

async function initializeEditor() {
  byId("teamTraining").onchange = () => {
    if (activeBuild) showBuild(activeBuild);
    else renderChoices();
  };
  window.addEventListener('datasetchanged',()=>{
    activeBuild=null;sessionStorage.removeItem(buildKey);
    byId('confirmBuild').hidden=true;byId('viewBuild').disabled=true;byId('prepareBuild').hidden=false;
    setBuildState('dirty','Dataset changed');
    byId('buildMessage').textContent='Prepare the pipeline again to validate it with the attached dataset.';
  });
  byId("resetPipeline").onclick = () => {
    if (buildBusy) return;
    draft = { ...library.defaults }; activeBuild = null;
    byId("teamTraining").checked = false;
    sessionStorage.removeItem(draftKey); sessionStorage.removeItem(buildKey);
    byId("buildSteps").value = "2";
    setBuildState("idle", "Ready to configure");
    byId("buildMessage").textContent = "Full original architecture, atlas transfer, cosine graph and GCN. Optional stages are skipped.";
    byId("prepareBuild").hidden = false; byId("confirmBuild").hidden = true;
    byId("viewBuild").disabled = true; byId("activityPanel").hidden = true;
    renderChoices();
  };
  byId("runtimeDetails").onclick = async () => {
    try {
      const runtime = await requestJson("/api/runtime");
      showDetails("Runtime", table(["Setting", "Value"], Object.entries(runtime).map(([key, value]) => [titleCase(key), escapeHtml(value)])) + '<p>Use UNIBRAIN_CONFIG or environment variables to move this workspace to another machine. Remote access uses an SSH port forward. Original training uses one visible CUDA GPU.</p>', "PORTABLE CONFIGURATION");
    } catch (error) { showDetails("Runtime unavailable", '<p>' + escapeHtml(error.message) + '</p>'); }
  };
  byId("prepareBuild").onclick = prepareSelection;
  byId("viewBuild").onclick = reviewBuild;
  byId("confirmBuild").onclick = runBuild;
  window.addEventListener("experimentfailed", () => {
    byId("buildSteps").disabled = false;
    byId("confirmBuild").textContent = "Confirm & run";
    buildError(new Error("See the run error below. Apply changes to retry."));
  });
  setInterval(() => {
    if (activityStarted) byId("buildElapsed").textContent = Math.max(0, Math.floor((Date.now() - activityStarted) / 1000)) + "s";
  }, 1000);
  if (!(await backendReady)) {
    library = appConfig.catalog;draft = {...library.defaults};renderChoices();
    byId("buildMessage").textContent = "Browse stages, then connect your backend in AI settings to build and train.";
    byId("prepareBuild").disabled = true;
    return;
  }
  try {
    library = await requestJson("/api/catalog");
    draft = { ...library.defaults };
    const saved = sessionStorage.getItem(draftKey);
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        if (library.stages.every(stage => stage.options.some(option => option.id === parsed[stage.id]))) draft = parsed;
      } catch (_) { sessionStorage.removeItem(draftKey); }
    }
    renderChoices();
    if (report) renderMethods();
    const status = await requestJson("/api/status");
    const previous = status.running ? status.build_id : sessionStorage.getItem(buildKey);
    if (previous) {
      try {
        const record = await requestJson("/api/builds/" + previous);
        draft = { ...record.selection };
        byId("teamTraining").checked = record.coordination === "taskforce" || record.coordination === "roles";
        showBuild(record);
        if (buildBusy) pollBuild(previous);
      } catch (_) { sessionStorage.removeItem(buildKey); }
    }
    if (status.running && status.job === "run") {
      buildBusy = true;
      byId("results").hidden = true;
      byId("metricsPanel").hidden = true;
      byId("gradientPanel").hidden = true;
      byId("confirmBuild").disabled = true;
      byId("prepareBuild").disabled = true;
      byId("buildSteps").disabled = true;
      byId("confirmBuild").textContent = "Running…";
      setBuildState("working", "Running, please wait");
      byId("buildMessage").textContent = "Your selected pipeline is running. Follow its progress below.";
      byId("buildSteps").value = String(status.live?.epochs || (isOriginal() ? 2 : status.live?.steps || 5));
      byId("activityPanel").hidden = true;
      renderChoices();
      poll();
    } else if (status.job === "run" && status.error) {
      if (status.live) renderLive(status.live);
      setFailure(status.error);
    }
  } catch (error) {
    byId("buildMessage").textContent = error.message;
    byId("prepareBuild").disabled = true;
  }
}

initializeEditor();
