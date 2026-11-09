"use strict";

let library = null;
let draft = null;
let activeBuild = null;
let buildBusy = false;
let buildPoll = null;

function chosenOption(stage, selection = draft) {
  return stage.options.find(option => option.id === selection[stage.id]);
}

function renderChoices() {
  byId("stageChoices").innerHTML = library.stages.map((stage, index) => {
    const option = chosenOption(stage);
    return '<article class="choice-card ' + (option.id !== library.defaults[stage.id] ? "chosen" : "") + '"><div class="choice-heading"><span>' +
      String(index + 1).padStart(2, "0") + '</span><span>' + escapeHtml(stage.route) + '</span></div><h3>' + escapeHtml(stage.name) +
      '</h3><p>' + escapeHtml(option.label) + '</p><button class="text-button choose-tool" data-choice-stage="' + stage.id +
      '" aria-label="Choose tool for ' + escapeHtml(stage.name) + '">Choose tool <span>↗</span></button></article>';
  }).join("");
  byId("stageChoices").querySelectorAll("button").forEach(button => {
    button.onclick = () => chooseTool(button.dataset.choiceStage);
  });
}

function toolEvidence(stage, choice) {
  const option = stage.options.find(item => item.id === choice);
  const research = option.research;
  byId("toolEvidence").innerHTML = '<p class="tool-adaptation">' + escapeHtml(option.adaptation) + '</p>' +
    (research ? '<h3>Interface</h3><p>' + escapeHtml(research.interface) + '</p><h3>Prior isolated checks</h3><p>' +
      escapeHtml(research.prior_check) + '</p><h3>Transfer limits</h3><p>' + escapeHtml(research.limitations) + '</p><div class="tool-sources">' +
      sourceLink(research.paper, "Paper") + sourceLink(research.code, "Pinned source") + '</div>' : "") +
    '<p class="muted">The selected combination must pass a new runtime check. Prior isolated checks do not certify this pipeline.</p>';
}

function chooseTool(stageId) {
  const stage = library.stages.find(item => item.id === stageId);
  const options = stage.options.map(option => '<option value="' + option.id + '"' + (draft[stageId] === option.id ? " selected" : "") + '>' +
    escapeHtml(option.label + (option.status === "requires_inputs" ? " · needs mesh inputs" : "")) + '</option>').join("");
  showDetails(stage.name, '<label class="tool-select-label" for="toolChoice">Tool for this stage</label><select id="toolChoice">' + options +
    '</select><div id="toolEvidence"></div><div class="tool-action"><button id="useTool" class="primary"' + (buildBusy ? " disabled" : "") +
    '>Use tool and validate</button><span>' + (buildBusy ? "Wait for the current build to finish." : "Generates code and tests two optimizer steps.") + '</span></div>', "STAGE TOOL SELECTION");
  byId("toolChoice").onchange = event => toolEvidence(stage, event.target.value);
  toolEvidence(stage, draft[stageId]);
  byId("useTool").onclick = () => {
    const choice = byId("toolChoice").value;
    draft[stageId] = choice;
    if (choice === "M03") {
      draft.extraction = "M03";
      draft.registration = "M03";
    } else if (stageId === "extraction" && draft.registration === "M03") {
      draft.registration = "unibrain";
    } else if (stageId === "registration" && draft.extraction === "M03") {
      draft.extraction = "unibrain";
    }
    activeBuild = null;
    byId("detailDialog").close();
    renderChoices();
    prepareSelection();
  };
}

function showBuild(record) {
  activeBuild = record;
  buildBusy = ["queued", "preparing", "validating"].includes(record.status);
  const matches = JSON.stringify(record.selection) === JSON.stringify(draft);
  const titles = { queued: "Build queued", preparing: "Agent is preparing the changes", validating: "Checking the training graph",
    ready: "Training checks passed", failed: "This composition cannot run yet" };
  byId("buildStatus").textContent = titles[record.status] || record.status;
  byId("buildMessage").textContent = record.error || record.events?.at(-1)?.message || "Preparing selected stages";
  byId("buildStatus").className = record.status === "ready" ? "build-ready" : record.status === "failed" ? "build-failed" : "";
  byId("prepareBuild").disabled = buildBusy;
  byId("viewBuild").disabled = !record.code;
  byId("confirmBuild").disabled = buildBusy || record.status !== "ready" || !matches;
  const validation = record.validation;
  byId("buildChecks").innerHTML = validation?.passed ? ['Finite outputs', 'Input gradients', 'Parameter gradients', 'Optimizer updates', 'Spatial checks']
    .map(label => '<span>✓ ' + label + '</span>').join("") : "";
  if (validation?.passed) byId("buildChecks").innerHTML += '<small>One sample · two updates · random weights</small>';
}

async function requestJson(url, data) {
  const response = await fetch(url, data === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || "Request failed");
  return value;
}

async function pollBuild(identifier) {
  clearTimeout(buildPoll);
  try {
    const value = await requestJson("/api/builds/" + identifier);
    showBuild(value);
    if (buildBusy) buildPoll = setTimeout(() => pollBuild(identifier), 1500);
  } catch (error) {
    buildBusy = false;
    byId("buildMessage").textContent = error.message;
    byId("prepareBuild").disabled = false;
  }
}

async function prepareSelection() {
  buildBusy = true;
  byId("prepareBuild").disabled = true;
  byId("confirmBuild").disabled = true;
  byId("viewBuild").disabled = true;
  byId("buildStatus").textContent = "Preparing the selected tools";
  byId("buildMessage").textContent = "The current experiment remains visible below.";
  byId("buildChecks").innerHTML = "";
  try {
    const value = await requestJson("/api/prepare", { selection: draft, backend: byId("backend").value });
    sessionStorage.setItem("unibrainBuild", value.build_id);
    await pollBuild(value.build_id);
  } catch (error) {
    buildBusy = false;
    byId("buildStatus").textContent = "Preparation failed";
    byId("buildMessage").textContent = error.message;
    byId("prepareBuild").disabled = false;
  }
}

function reviewBuild() {
  if (!activeBuild) return;
  const validation = activeBuild.validation;
  let body = '<p>' + escapeHtml(activeBuild.plan?.rationale || "") + '</p><h3>Generated pipeline.py</h3><pre class="code-preview">' +
    escapeHtml(activeBuild.code) + '</pre><h3>Changes from baseline</h3><pre class="code-preview">' + escapeHtml(activeBuild.diff) + '</pre>';
  if (validation?.updates) body += '<h3>Measured parameter updates</h3>' + table(["Module", "Parameter tensors", "Updated tensors"],
    Object.entries(validation.updates).map(([name, info]) => [escapeHtml(name), info.parameter_tensors, info.updated_tensors]));
  if (validation?.model) body += '<h3>Applied adaptations</h3><ul>' + validation.model.adaptations.map(text => '<li>' + escapeHtml(text) + '</li>').join("") + '</ul>';
  body += '<p class="muted">' + escapeHtml(validation?.scope || activeBuild.error || "Validation in progress") + '</p>';
  showDetails("Review the build", body, "BUILD " + activeBuild.id);
}

async function runBuild() {
  if (!activeBuild || activeBuild.status !== "ready") return;
  byId("confirmBuild").disabled = true;
  byId("prepareBuild").disabled = true;
  buildBusy = true;
  try {
    await requestJson("/api/builds/" + activeBuild.id + "/run", { steps: Number(byId("buildSteps").value), backend: activeBuild.backend });
    byId("buildStatus").textContent = "Running the generated pipeline";
    byId("buildMessage").textContent = "Paired baseline and candidate training is running. Results below belong to the previous experiment.";
    byId("progressTitle").textContent = "Experiment running";
    byId("error").textContent = "";
    document.querySelector(".spinner").hidden = false;
    poll();
  } catch (error) {
    buildBusy = false;
    byId("buildStatus").textContent = "Run could not start";
    byId("buildMessage").textContent = error.message;
    byId("prepareBuild").disabled = false;
  }
}

async function initializeEditor() {
  byId("run").onclick = () => byId("builderPanel").scrollIntoView({ behavior: "smooth", block: "start" });
  byId("prepareBuild").onclick = prepareSelection;
  byId("viewBuild").onclick = reviewBuild;
  byId("confirmBuild").onclick = runBuild;
  window.addEventListener("experimentfailed", () => {
    buildBusy = false;
    byId("prepareBuild").disabled = false;
    byId("confirmBuild").disabled = true;
  });
  if (location.protocol === "file:") {
    byId("buildMessage").textContent = "Open the local dashboard to edit and run a pipeline.";
    byId("prepareBuild").disabled = true;
    return;
  }
  try {
    library = await requestJson("/api/catalog");
    draft = { ...library.defaults, ...(report?.build?.selection || {}) };
    byId("catalogCount").textContent = library.method_count + " RESEARCH METHODS";
    renderChoices();
    if (report) renderMethods();
    const status = await requestJson("/api/status");
    const previous = status.build_id || sessionStorage.getItem("unibrainBuild");
    if (previous) {
      try {
        const record = await requestJson("/api/builds/" + previous);
        draft = { ...record.selection };
        renderChoices();
        showBuild(record);
        if (buildBusy) pollBuild(previous);
      } catch (_) { sessionStorage.removeItem("unibrainBuild"); }
    }
    if (status.running && status.job === "run") {
      buildBusy = true;
      byId("confirmBuild").disabled = true;
      byId("prepareBuild").disabled = true;
      poll();
    }
  } catch (error) {
    byId("buildMessage").textContent = error.message;
    byId("prepareBuild").disabled = true;
  }
}

initializeEditor();
