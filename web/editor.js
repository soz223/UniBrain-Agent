"use strict";

let library = null;
let draft = null;
let activeBuild = null;
let buildBusy = false;
let buildPoll = null;
let activityStarted = null;
const mainStages = ["denoising", "bias", "extraction", "normalization", "registration", "parcellation", "graph", "classification"];
const stageNames = { extraction: "Extraction", normalization: "Normalization", parcellation: "Parcellation", graph: "Network generation" };

function chosenOption(stage, selection = draft) {
  return stage.options.find(option => option.id === selection[stage.id]);
}

function toolName(option) {
  return { none: "Skip", unibrain: "UniBrain", soft_atlas: "Soft atlas", continuous: "Cosine graph" }[option.id] || option.label.split(" (")[0];
}

function renderChoices() {
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
          option.id === "none" ? "Bypass this optional step" : option.research ? option.label : "Current compact adapter") +
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
  draft[stageId] = choice;
  let note = "Selection updated. Apply changes to generate and validate the pipeline.";
  if (choice === "M03") {
    draft.extraction = draft.registration = "M03";
    draft.normalization = "none";
    note = "ERNet links extraction and registration. Normalization is skipped to preserve that connection.";
  } else if (stageId === "extraction" && draft.registration === "M03") draft.registration = "unibrain";
  else if (stageId === "registration" && draft.extraction === "M03") draft.extraction = "unibrain";
  activeBuild = null;
  sessionStorage.removeItem("unibrainBuild");
  sessionStorage.setItem("unibrainDraft", JSON.stringify(draft));
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
  const phases = ["contracts", "planning", "training", "audit", "ready"];
  const labels = ["Check tools", "Agent plan", "Train × 2", "Audit", "Ready"];
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
  byId("buildStatus").textContent = titles[record.status] || record.status;
  byId("buildStatus").className = record.status === "ready" ? "build-ready" : record.status === "failed" ? "build-failed" : "";
  byId("buildMessage").textContent = record.error || (record.status === "ready" ? "Gradient, parameter update and spatial checks passed on one sample." :
    record.progress?.message || record.events?.at(-1)?.message || "Preparing selected stages");
  byId("prepareBuild").disabled = buildBusy;
  byId("prepareBuild").hidden = record.status === "ready" && matches;
  byId("confirmBuild").hidden = record.status !== "ready" || !matches;
  byId("confirmBuild").disabled = buildBusy || record.status !== "ready" || !matches;
  byId("viewBuild").hidden = !record.code;
  byId("viewBuild").disabled = !record.code;
  showActivity(record);
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
    const value = await requestJson("/api/prepare", { selection: draft, backend: "codex" });
    sessionStorage.setItem("unibrainBuild", value.build_id);
    sessionStorage.removeItem("unibrainDraft");
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
  renderChoices();
  try {
    await requestJson("/api/builds/" + activeBuild.id + "/run", { steps: Number(byId("buildSteps").value), backend: activeBuild.backend });
    byId("buildStatus").textContent = "Running the pipeline";
    byId("buildMessage").textContent = "Training baseline and candidate. Results below show the last completed experiment.";
    byId("progressTitle").textContent = "Experiment running";
    byId("error").textContent = "";
    byId("runSpinner").hidden = false;
    poll();
  } catch (error) { buildError(error); }
}

async function initializeEditor() {
  byId("prepareBuild").onclick = prepareSelection;
  byId("viewBuild").onclick = reviewBuild;
  byId("confirmBuild").onclick = runBuild;
  window.addEventListener("experimentfailed", () => buildError(new Error("See the experiment error above. Apply changes to retry.")));
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
    draft = { ...library.defaults, ...(report?.build?.selection || {}) };
    const saved = sessionStorage.getItem("unibrainDraft");
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        if (library.stages.every(stage => stage.options.some(option => option.id === parsed[stage.id]))) draft = parsed;
      } catch (_) { sessionStorage.removeItem("unibrainDraft"); }
    }
    renderChoices();
    if (report) renderMethods();
    const status = await requestJson("/api/status");
    const previous = status.running ? status.build_id : saved ? null : sessionStorage.getItem("unibrainBuild");
    if (previous) {
      try {
        const record = await requestJson("/api/builds/" + previous);
        draft = { ...record.selection };
        showBuild(record);
        if (buildBusy) pollBuild(previous);
      } catch (_) { sessionStorage.removeItem("unibrainBuild"); }
    }
    if (status.running && status.job === "run") {
      buildBusy = true;
      byId("confirmBuild").disabled = true;
      byId("prepareBuild").disabled = true;
      renderChoices();
      poll();
    }
  } catch (error) {
    byId("buildMessage").textContent = error.message;
    byId("prepareBuild").disabled = true;
  }
}

initializeEditor();
