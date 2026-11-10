"use strict";

let library = null;
let draft = null;
let activeBuild = null;
let buildBusy = false;
let buildPoll = null;
let activityStarted = null;
const mainStages = ["denoising", "bias", "extraction", "normalization", "registration", "parcellation", "graph", "classification"];
const stageNames = { extraction: "Extraction", normalization: "Normalization", parcellation: "Parcellation", graph: "Network generation" };
const draftKey = "unibrainDraftV2";
const buildKey = "unibrainBuildV2";

function isOriginal() {
  return library && draft && library.stages.every(stage => draft[stage.id] === library.defaults[stage.id]);
}

function chosenOption(stage, selection = draft) {
  return stage.options.find(option => option.id === selection[stage.id]);
}

function toolName(option) {
  return { none: "Skip", unibrain: "UniBrain", soft_atlas: "Soft atlas", continuous: "Cosine graph" }[option.id] || option.label.split(" (")[0];
}

function renderChoices() {
  byId("profileLabel").innerHTML = isOriginal() ? 'Original UniBrain<br><small>Full architecture</small>' : 'Custom pipeline<br><small>Compact adapters</small>';
  byId("budgetLabel").textContent = isOriginal() ? "Epochs" : "Steps";
  byId("prepareBuild").textContent = isOriginal() ? "Prepare UniBrain" : "Apply changes";
  byId("resetPipeline").disabled = buildBusy;
  if (byId("teamTraining")) {
    byId("teamTraining").disabled = isOriginal() || buildBusy;
    if (isOriginal()) byId("teamTraining").checked = false;
    byId("teamTrainingNote").textContent = isOriginal() ? "Original reproduction keeps its loss weights. Role reviews remain available." : "Four task specialists and a coordinator review measured gradients before update 2.";
  }
  const button = (stage, index) => {
    const option = chosenOption(stage);
    const optional = stage.options.some(item => item.id === "none");
    const skipped = option.id === "none";
    return '<button class="stage-choice ' + (skipped ? "skipped" : "enabled") + '" data-choice-stage="' + stage.id +
      '" aria-label="' + escapeHtml(stage.name + ": " + toolName(option) + ". Change tool") + '"' + (buildBusy ? " disabled" : "") +
      '><span class="stage-top"><span class="stage-number">' + (index === null ? "↳" : String(index + 1).padStart(2, "0")) +
      '</span><strong>' + escapeHtml(stageNames[stage.id] || stage.name) + '</strong>' + (optional ? '<small>optional</small>' : "") +
      '</span><span class="stage-tool"><span>' + (skipped ? "○ " : "✓ ") + escapeHtml(toolName(option)) + '</span><span>⌄</span></span></button>';
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
  byId("buildStatus").textContent = "Unapplied changes";
  byId("buildStatus").className = "";
  byId("buildMessage").textContent = note;
  byId("confirmBuild").hidden = true;
  byId("prepareBuild").hidden = false;
  byId("viewBuild").hidden = true;
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
  const matches = library.stages.every(stage => record.selection[stage.id] === draft[stage.id]);
  const titles = { queued: "Starting checks", preparing: "Agent is building your pipeline", validating: "Validating the training graph",
    ready: "Ready to run", failed: "Build needs attention" };
  if (record.profile === "original") titles.preparing = "Checking original UniBrain";
  byId("buildStatus").textContent = titles[record.status] || record.status;
  byId("buildStatus").className = record.status === "ready" ? "build-ready" : record.status === "failed" ? "build-failed" : "";
  byId("buildMessage").textContent = record.error || (record.status === "ready" ? record.profile === "original" ? "Original sources and CUDA runtime checked. Training verification runs after confirmation." : "Gradient, parameter update and spatial checks passed on one sample." :
    record.progress?.message || record.events?.at(-1)?.message || "Preparing selected stages");
  byId("prepareBuild").disabled = buildBusy;
  byId("prepareBuild").hidden = record.status === "ready" && matches;
  byId("confirmBuild").hidden = record.status !== "ready" || !matches;
  byId("confirmBuild").disabled = buildBusy || record.status !== "ready" || !matches;
  byId("viewBuild").hidden = !record.code;
  byId("viewBuild").disabled = !record.code;
  showActivity(record);
  if (record.status === "ready" && report?.build?.id === record.id && report.status === "complete") {
    byId("activityPanel").hidden = true;
    byId("buildStatus").textContent = "Ready for another run";
    byId("buildMessage").textContent = "Last run completed. Adjust the tools or run this pipeline again.";
  }
  renderChoices();
}

async function requestJson(url, data) {
  const response = await fetch(url, data === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || "Request failed");
  return value;
}

function buildError(error) {
  buildBusy = false;
  activityStarted = null;
  byId("buildStatus").textContent = "Build needs attention";
  byId("buildMessage").textContent = error.message;
  byId("buildStatus").className = "build-failed";
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
  showBuild({ status: "queued", selection: { ...draft }, created: new Date().toISOString(), events: [] });
  try {
    const value = await requestJson("/api/prepare", { selection: draft, backend: "configured" });
    sessionStorage.setItem(buildKey, value.build_id);
    sessionStorage.removeItem(draftKey);
    await pollBuild(value.build_id);
  } catch (error) { buildError(error); }
}

function reviewBuild() {
  if (!activeBuild) return;
  const validation = activeBuild.validation;
  let body = '<p>' + escapeHtml(activeBuild.plan?.rationale || "") + '</p><h3>Generated pipeline.py</h3><pre class="code-preview">' +
    escapeHtml(activeBuild.code) + '</pre><h3>Changes from baseline</h3><pre class="code-preview">' + escapeHtml(activeBuild.diff) + '</pre>';
  if (validation?.updates) body += '<h3>Measured parameter updates</h3>' + table(["Module", "Parameter tensors", "Updated tensors"],
    Object.entries(validation.updates).map(([name, info]) => [escapeHtml(name), info.parameter_tensors, info.updated_tensors]));
  body += '<p class="muted">' + escapeHtml(validation?.scope || activeBuild.error || "Validation in progress") + '</p>';
  showDetails("Review the build", body, "BUILD " + activeBuild.id);
}

async function runBuild() {
  if (!activeBuild || activeBuild.status !== "ready" || buildBusy) return;
  buildBusy = true;
  byId("confirmBuild").disabled = true;
  byId("prepareBuild").disabled = true;
  byId("buildSteps").disabled = true;
  byId("confirmBuild").textContent = "Running…";
  byId("buildStatus").textContent = "Running, please wait";
  byId("buildMessage").textContent = "Your selected pipeline is starting. Follow its progress below.";
  startLiveRun(Number(byId("buildSteps").value));
  renderChoices();
  try {
    await requestJson("/api/builds/" + activeBuild.id + "/run", { steps: Number(byId("buildSteps").value), backend: activeBuild.backend,
      coordination: byId("teamTraining").checked ? "roles" : "single" });
    poll();
  } catch (error) { setFailure(error); }
}

async function initializeEditor() {
  byId("resetPipeline").onclick = () => {
    if (buildBusy) return;
    draft = { ...library.defaults }; activeBuild = null;
    sessionStorage.removeItem(draftKey); sessionStorage.removeItem(buildKey);
    byId("buildSteps").value = "2";
    byId("buildStatus").textContent = "Original UniBrain";
    byId("buildMessage").textContent = "Full original architecture, atlas transfer, cosine graph and GCN. Optional stages are skipped.";
    byId("prepareBuild").hidden = false; byId("confirmBuild").hidden = true;
    byId("viewBuild").hidden = true; byId("activityPanel").hidden = true;
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
  if (location.protocol === "file:") {
    byId("buildMessage").textContent = "Open the local dashboard to edit and run a pipeline.";
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
    const previous = status.running ? status.build_id : null;
    if (previous) {
      try {
        const record = await requestJson("/api/builds/" + previous);
        draft = { ...record.selection };
        showBuild(record);
        if (buildBusy) pollBuild(previous);
      } catch (_) { sessionStorage.removeItem(buildKey); }
    }
    if (status.running && status.job === "run") {
      buildBusy = true;
      byId("confirmBuild").disabled = true;
      byId("prepareBuild").disabled = true;
      byId("buildSteps").disabled = true;
      byId("confirmBuild").textContent = "Running…";
      byId("buildStatus").textContent = "Running, please wait";
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
