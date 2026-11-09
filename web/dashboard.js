"use strict";

const report = JSON.parse(document.getElementById("reportData").textContent);
const byId = id => document.getElementById(id);
const escapeHtml = value => String(value ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const format = (value, digits = 3) => Number.isFinite(value) ? value.toFixed(digits) : "Not measured";
const scientific = value => Number.isFinite(value) ? value.toExponential(1) : "Not measured";
const titleCase = value => value.charAt(0).toUpperCase() + value.slice(1).replaceAll("_", " ");
const modules = { denoise_net: "Denoising", bias_net: "Bias correction", ext_net: "Extraction", norm_net: "Normalization", reg_net: "Registration", seg_net: "Segmentation", par_net: "Parcellation head", feature_net: "ROI encoder", graph_gen: "Graph generation", gnn_pred: "Classifier" };
const tasks = ["extraction", "registration", "segmentation", "classification"];
const signalNames = { active: "Receives a learning signal", zero: "Connected · zero gradient", disconnected: "No path from this loss", inactive: "No trainable parameters", nonfinite: "Nonfinite gradient" };
let imageStage = "input";
let volumeRequest = 0;
let volumeAbort = null;
let pollTimer = null;
let liveRun = null;
let runBusy = false;
let journalSignature = "";
let reloadPending = false;
const sliceCache = new Map();

function paintIcons() {}

function table(headers, rows) {
  return '<table class="data-table"><thead><tr>' + headers.map(text => '<th scope="col">' + escapeHtml(text) + '</th>').join("") +
    '</tr></thead><tbody>' + rows.map(row => '<tr>' + row.map(value => '<td>' + value + '</td>').join("") + '</tr>').join("") + '</tbody></table>';
}

function showDetails(title, body, eyebrow = "EXPERIMENT") {
  byId("detailDialog").classList.remove("tool-dialog");
  byId("dialogTitle").textContent = title;
  byId("dialogEyebrow").textContent = eyebrow;
  byId("dialogBody").innerHTML = body;
  if (!byId("detailDialog").open) byId("detailDialog").showModal();
}

function sourceLink(url, label) {
  return /^https:\/\//.test(url || "") ? '<a href="' + escapeHtml(url) + '" target="_blank" rel="noreferrer">' + escapeHtml(label) + '</a>' : "";
}

function duration(seconds) {
  const n = Math.max(0, Math.floor(seconds || 0));
  return n < 60 ? n + "s" : Math.floor(n / 60) + "m " + n % 60 + "s";
}

function startLiveRun(steps) {
  runBusy = true;
  byId("results").hidden = true;
  byId("empty").hidden = true;
  byId("activityPanel").hidden = true;
  byId("runError").hidden = true;
  renderLive({ phase: "starting", message: "Starting your pipeline", steps, history: [], modules: [], agents: [], events: [], elapsed_seconds: 0, resources: {} });
  byId("runPanel").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderLive(value) {
  liveRun = value;
  const finished = value.phase === "complete";
  const failed = value.phase === "failed";
  runBusy = !finished && !failed;
  byId("runPanel").hidden = false;
  byId("runPanel").classList.toggle("has-failed", failed);
  byId("runIndicator").className = runBusy ? "spinner" : "run-mark";
  byId("runIndicator").textContent = runBusy ? "" : failed ? "!" : "✓";
  const titles = { starting: "Running, please wait", preparing: "Running, please wait", training: "Model is running", coordination: "Agent is coordinating training", audit: "Checking learning signals", validation: "Validating the model", saving: "Saving your experiment", complete: "Run complete", failed: "Run needs attention" };
  byId("runTitle").textContent = titles[value.phase] || "Running, please wait";
  byId("runMessage").textContent = value.message || "Waiting for execution feedback";
  byId("runElapsed").textContent = duration(value.elapsed_seconds);
  byId("snapshotLabel").textContent = runBusy ? "Live experiment" : failed ? "Run interrupted" : "Saved experiment";
  const phases = ["preparing", "training", "audit", "validation", "saving"];
  const position = finished ? phases.length : value.phase === "coordination" ? 1 : Math.max(0, phases.indexOf(value.phase));
  byId("runPhases").innerHTML = ["Load pipeline", "Train", "Check gradients", "Validate", "Save"].map((label, i) => '<li class="' + (i < position ? "done" : i === position && !failed ? "current" : "") + '">' + (i < position ? "✓ " : "") + label + '</li>').join("");
  const stats = value.resources || {};
  const history = value.history || [];
  byId("stepCount").textContent = history.length + " / " + value.steps + " updates";
  byId("runtimeStats").innerHTML = [["Device", stats.device || "Pending", stats.threads ? stats.threads + " threads" : ""], ["Worker CPU", Number.isFinite(stats.cpu_percent) ? format(stats.cpu_percent, 1) + "%" : "…", "of total CPU"], ["Worker memory", Number.isFinite(stats.rss_mb) ? format(stats.rss_mb / 1024, 2) : "…", "GB resident"], ["Updates", history.length + " / " + value.steps, finished ? "completed" : "optimizer steps"]].map(([label, number, unit]) => '<div class="runtime-stat"><span>' + label + '</span><strong>' + number + '</strong><br><small>' + unit + '</small></div>').join("");
  const current = value.modules?.find(item => item.module === value.current_module);
  const agent = runBusy && value.agents?.find(item => item.status === "running");
  let activity = current ? current.label + " is running" : agent ? titleCase(agent.role) + " agent is thinking" : value.phase === "training" ? "Computing gradients and updating weights" : value.phase === "validation" ? "Validation " + value.validation?.completed + " / " + value.validation?.total : finished ? "Checkpoint and outputs saved" : failed ? "Execution stopped" : "Waiting for the next stage";
  byId("currentActivity").innerHTML = escapeHtml(activity) + '<small>' + (finished ? "Resources: last observed worker sample." : "Live worker metrics · refreshed every second.") + '</small>';
  renderLoss();
  renderCalls(value);
  renderJournal(value.events || []);
}

function renderLoss() {
  const key = byId("lossKind").value;
  const history = liveRun?.history || [];
  if (!history.length) {
    byId("lossChart").innerHTML = '<div class="chart-empty"><span>Waiting for the first update</span><small>The curve appears as optimizer steps finish.</small></div>';
    byId("lossSummary").textContent = "No completed updates yet";
    return;
  }
  const values = history.map(step => key === "weighted_loss" ? step.weighted_loss : step.losses[key]);
  const low = Math.min(...values), high = Math.max(...values), pad = Math.max((high - low) * .18, Math.abs(high) * .02, .01);
  const minimum = low - pad, maximum = high + pad;
  const x = i => 48 + i * 478 / Math.max(1, (liveRun.steps || history.length) - 1);
  const y = n => 155 - (n - minimum) * 128 / (maximum - minimum);
  const grid = Array.from({ length: 4 }, (_, i) => {
    const n = minimum + (maximum - minimum) * i / 3;
    return '<path d="M48 ' + y(n) + 'H532" stroke="#37413b" stroke-width=".7"/><text x="40" y="' + (y(n) + 3) + '" text-anchor="end" fill="#95a69b">' + n.toFixed(2) + '</text>';
  }).join("");
  const line = '<polyline points="' + values.map((v, i) => x(i) + ',' + y(v)).join(' ') + '" fill="none" stroke="#a6d8c7" stroke-width="2"/>';
  const points = values.map((v, i) => '<circle cx="' + x(i) + '" cy="' + y(v) + '" r="3.5" fill="#a6d8c7"><title>Step ' + history[i].step + ': ' + format(v, 5) + '</title></circle>').join("");
  byId("lossChart").innerHTML = '<svg viewBox="0 0 550 185" role="img" aria-label="Selected model training loss by optimizer step">' + grid + line + points + '<text x="48" y="179" fill="#89988f">STEP 1</text><text x="528" y="179" text-anchor="end" fill="#89988f">STEP ' + liveRun.steps + '</text></svg>';
  byId("lossSummary").textContent = "Latest " + format(values.at(-1), 4) + (key === "weighted_loss" ? " · includes current loss weights" : "");
}

function renderCalls(value) {
  const calls = value.modules || [];
  const agentCount = (value.agents || []).length;
  byId("callSummary").textContent = calls.length + " modules · " + agentCount + (agentCount === 1 ? " agent" : " agents");
  byId("modelCalls").innerHTML = calls.length ? table(["Module", "Forward calls", "Total time", "State"], calls.map(item => [escapeHtml(item.label), item.calls, format(item.seconds, 2) + "s", item.status === "started" ? runBusy ? "Running" : "Interrupted" : "Completed"])) : '<p class="disclosure-note">Model calls will appear as the pipeline starts.</p>';
  byId("agentCalls").innerHTML = (value.agents || []).map(item => {
    const usage = item.usage || {};
    const tokens = usage.input_tokens != null || usage.output_tokens != null ? (usage.input_tokens || 0).toLocaleString() + " input · " + (usage.output_tokens || 0).toLocaleString() + " output tokens" : "Usage not reported";
    return '<div class="agent-call"><strong>' + escapeHtml(titleCase(item.role)) + ' agent</strong><span>' + (item.status === "running" ? runBusy ? "Thinking…" : "Interrupted" : duration(item.seconds) + " · " + tokens) + '</span></div>';
  }).join("");
}

function renderJournal(events) {
  const visible = events.filter(event => event.kind !== "module" && event.branch !== "baseline" && event.role !== "review" && !/^baseline:/i.test(event.message));
  byId("eventCount").textContent = visible.length + (visible.length === 1 ? " entry" : " entries");
  const signature = JSON.stringify(visible);
  if (signature === journalSignature) return;
  journalSignature = signature;
  const opened = new Set([...byId("events").querySelectorAll("details[open]")].map(node => node.dataset.entry));
  byId("events").innerHTML = visible.map((event, index) => {
    const id = String(index), label = event.message.replace(/^candidate: /, "Training: ");
    const data = Object.fromEntries(Object.entries(event).filter(([key]) => !["message", "seconds", "kind", "subject_id", "config"].includes(key)));
    let detail;
    if (event.kind === "training") detail = table(["Loss", "Value", "Weight"], Object.entries(event.losses || {}).map(([task, loss]) => [titleCase(task), format(loss, 5), format(event.weights?.[task])])) + '<p>Weighted total: ' + format(event.weighted_loss, 5) + '</p>';
    else if (event.decision) detail = '<p>' + escapeHtml(event.decision.rationale) + '</p><p>Action: ' + escapeHtml(titleCase(event.decision.action || "recorded")) + ' · ' + duration(event.duration) + '</p>';
    else detail = '<pre>' + escapeHtml(Object.keys(data).length ? JSON.stringify(data, null, 2) : "Event recorded by the pipeline runner.") + '</pre>';
    return '<details class="journal-entry" data-entry="' + id + '"' + (opened.has(id) ? " open" : "") + '><summary><time>' + duration(event.seconds) + '</time><span>' + escapeHtml(label) + '</span></summary>' + detail + '</details>';
  }).join("");
}

function setFailure(error) {
  const message = error.message || String(error);
  renderLive({ ...liveRun, phase: "failed", message });
  byId("runError").hidden = false;
  byId("runError").textContent = message;
  window.dispatchEvent(new CustomEvent("experimentfailed"));
}

async function poll() {
  clearTimeout(pollTimer);
  try {
    const response = await fetch("/api/status");
    if (!response.ok) throw new Error("Execution service unavailable");
    const state = await response.json();
    if (state.job !== "run") return;
    if (state.live) renderLive(state.live);
    if (state.error) { setFailure(state.error); return; }
    if (state.running) {
      byId("results").hidden = true;
      byId("empty").hidden = true;
      pollTimer = setTimeout(poll, 1000);
    } else if (state.completed && state.run_dir !== report?.run_dir && !reloadPending) {
      reloadPending = true;
      setTimeout(() => location.reload(), 700);
    } else if (state.completed) {
      byId("results").hidden = !report?.candidate;
    }
  } catch (_) {
    byId("runMessage").textContent = "Connection interrupted. Reconnecting to the running experiment…";
    pollTimer = setTimeout(poll, 2000);
  }
}

async function renderVolume() {
  const request = ++volumeRequest;
  volumeAbort?.abort();
  const preview = report.previews.candidate;
  const maximum = Math.min(...(preview.shape || [96, 96, 96])) - 1;
  const index = Math.min(maximum, Math.max(0, Number(byId("sliceIndex").value)));
  byId("sliceIndex").max = maximum;
  byId("sliceIndex").value = index;
  byId("sliceValue").textContent = index + " / " + maximum;
  byId("previousSlice").disabled = index === 0;
  byId("nextSlice").disabled = index === maximum;
  const position = preview.slice_indices.indexOf(index);
  const key = imageStage + ":" + index;
  let planes = position >= 0 ? preview.planes[imageStage].map(axis => axis[position]) : sliceCache.get(key);
  if (!planes) {
    byId("volumeImages").setAttribute("aria-busy", "true");
    byId("volumeStatus").textContent = "Loading slice " + index + "…";
    try {
      if (location.protocol === "file:") throw new Error("Open the local workspace to view this slice.");
      volumeAbort = new AbortController();
      const query = new URLSearchParams({ run: preview.run_id || report.run_dir.split(/[\\/]/).at(-1), branch: "candidate", stage: imageStage, index });
      const response = await fetch("/api/volume?" + query, { signal: volumeAbort.signal });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "Slice unavailable");
      if (request !== volumeRequest) return;
      planes = result.planes;
      sliceCache.set(key, planes);
      if (sliceCache.size > 120) sliceCache.delete(sliceCache.keys().next().value);
    } catch (error) {
      if (request !== volumeRequest || error.name === "AbortError") return;
      byId("volumeImages").innerHTML = "";
      byId("volumeImages").setAttribute("aria-busy", "false");
      byId("volumeStatus").textContent = error.message;
      return;
    }
  }
  if (request !== volumeRequest) return;
  const crosshair = byId("crosshair").checked ? " crosshair" : "";
  byId("volumeImages").innerHTML = ["I", "J", "K"].map((plane, axis) => '<figure class="volume-plane' + crosshair + '" style="--cross-x:' + (index / maximum * 100) + '%;--cross-y:' + ((maximum - index) / maximum * 100) + '%"><img src="' + planes[axis] + '" alt="' + escapeHtml(titleCase(imageStage)) + ', array ' + plane + ' plane, index ' + index + '"><figcaption>' + plane + ' PLANE</figcaption><span class="plane-index">' + plane + ' = ' + index + '</span></figure>').join("");
  byId("volumeImages").setAttribute("aria-busy", "false");
  byId("volumeStatus").textContent = "All " + (maximum + 1) + " slices · one slice per arrow key";
  byId("sampleId").textContent = "After training";
}

function renderGradients() {
  const task = byId("gradientTask").value;
  const matrix = report.candidate.gradients.matrix;
  const connected = Object.values(matrix[task]).filter(cell => cell.state === "active").length;
  byId("gradientExplanation").textContent = titleCase(task) + " loss sends a nonzero learning signal to " + connected + " modules in this measurement. Select a module to inspect it.";
  byId("learningPaths").innerHTML = Object.entries(modules).filter(([key]) => matrix[task][key]?.parameter_tensors > 0).map(([key, name]) => {
    const state = matrix[task][key]?.state || "inactive";
    return '<button class="learning-chip ' + state + '" data-module="' + key + '"><strong>' + (state === "active" ? "✓ " : "") + name + '</strong><span>' + signalNames[state] + '</span></button>';
  }).join("");
  byId("learningPaths").querySelectorAll("button").forEach(button => button.onclick = () => {
    const key = button.dataset.module, cell = matrix[task][key] || {};
    showDetails(titleCase(task) + " → " + modules[key], '<p>' + escapeHtml(signalNames[cell.state || "inactive"]) + '.</p>' + table(["Measurement", "Value"], [["Trainable parameter tensors", cell.parameter_tensors || 0], ["Connected tensors", cell.connected_tensors || 0], ["Tensors with nonzero gradients", cell.nonzero_tensors || 0], ["Gradient L2 norm", scientific(cell.norm)], ["Parameter values changed during training", report.candidate.updated_parameters[key] || 0]]) + '<p>Updates accumulate contributions from all losses. No path from one loss does not mean this module never trains.</p>', "MEASURED LEARNING SIGNAL");
  });
  byId("gradientMatrix").innerHTML = table(["Module", ...tasks.map(titleCase)], Object.entries(modules).map(([key, name]) => [name, ...tasks.map(t => {
    const cell = matrix[t][key];
    return !cell || cell.state === "inactive" ? "No parameters" : cell.state === "active" ? scientific(cell.norm) : cell.state === "disconnected" ? "No path" : cell.state === "zero" ? "0 · connected" : "Nonfinite";
  })]));
}

function coordinationDetails() {
  const control = report.candidate.coordination;
  const rows = tasks.map(task => [titleCase(task), scientific(control.diagnostic.task_norms[task]), format(control.weights_before[task]), format(control.weights_after[task])]);
  const shared = control.diagnostic.shared_gradient_cosines.map(item => [item.tasks.map(titleCase).join(" / "), item.shared_parameter_tensors, format(item.cosine)]);
  showDetails("Training coordination", '<p>' + escapeHtml(control.rationale) + '</p>' + table(["Task", "Gradient norm", "Weight before", "Weight after"], rows) + '<h3>Gradients on shared parameters</h3>' + table(["Tasks", "Shared tensors", "Cosine"], shared) + '<p>One agent decision before optimizer update 2. Available actions: retain the loss weights or apply bounded gradient norm balancing. TaskForce multiagent learning is not implemented.</p>', "STEP " + control.step);
}

function renderMethods() {
  if (!report?.candidate || !library) return;
  const selection = report.build?.selection || report.candidate.model.config.stages || library.defaults;
  byId("methodCards").innerHTML = library.stages.map(stage => {
    const option = stage.options.find(item => item.id === selection[stage.id]);
    if (!option || option.id === "none") return "";
    return '<div class="method-row"><span>' + escapeHtml(stage.name) + '</span><strong>' + escapeHtml(option.label) + '</strong>' + (option.research ? sourceLink(option.research.paper, "Paper") + sourceLink(option.research.code, "Code") : '') + '</div>';
  }).join("");
}

function savedTelemetry() {
  if (report.live) return report.live;
  const model = report.candidate;
  const events = report.events || [];
  const observed = new Map();
  if (report.mode === "single") events.filter(event => event.kind === "module").forEach(event => {
    const item = observed.get(event.module) || { module: event.module, label: event.label, calls: 0, seconds: 0 };
    if (event.action === "completed") { item.calls++; item.seconds += event.duration; }
    item.status = event.action;
    observed.set(event.module, item);
  });
  return { phase: "complete", message: "Saved model outputs and training records", steps: model.steps, history: model.history, elapsed_seconds: report.mode === "single" ? report.elapsed_seconds : model.training_seconds, events, modules: [...observed.values()], agents: (report.agent_calls || []).filter(call => call.role === "coordination").map(call => ({ ...call, status: "completed" })), resources: { device: (report.environment?.device || "cpu").toUpperCase(), threads: report.environment?.threads } };
}

function renderResults() {
  const model = report.candidate, validation = model.validation;
  byId("results").hidden = false;
  byId("runDate").textContent = new Date(report.created).toLocaleString("en-US", { dateStyle: "medium", timeStyle: "short" });
  renderLive(savedTelemetry());
  renderVolume();
  renderGradients();
  const control = model.coordination;
  byId("trainingControl").innerHTML = control ? '<div><strong>Agent training decision</strong><span>Step ' + control.step + ' · ' + (control.action === "keep_weights" ? "Kept loss weights" : "Balanced gradient magnitudes") + '</span></div><button id="coordinationDetails" class="text-button">Inspect decision ↗</button>' : '<span>No coordination decision recorded.</span>';
  if (control) byId("coordinationDetails").onclick = coordinationDetails;
  byId("validationMetrics").innerHTML = [["Extraction Dice", validation.extraction_dice], ["Tissue Dice", validation.tissue_dice], ["Classification accuracy", validation.accuracy], ["Reference loss", validation.reference_loss]].map(([label, value]) => '<div class="metric"><span>' + label + '</span><strong>' + format(value) + '</strong></div>').join("");
  byId("dataCaveat").textContent = validation.count + " validation samples. Smoke test only: the training sample contains one class. These metrics do not establish diagnostic performance.";
  const splits = report.data.splits;
  byId("modelFacts").innerHTML = '<p class="facts">Compact research adapters · ' + model.steps + ' optimizer updates · seed ' + report.seed + '<br>' + splits.train.count + ' training samples · ' + splits.val.count + ' validation samples<br>Original UniBrain source: ' + escapeHtml(report.upstream_commit.slice(0, 12)) + '<br>Checkpoint: ' + escapeHtml(model.checkpoint) + '<br>Run: ' + escapeHtml(report.run_dir.split(/[\\/]/).at(-1)) + '</p><p class="disclosure-note">This workspace runs compact stage adapters. The verified full original UniBrain workflow is available through scripts/run_upstream.py.</p>';
  byId("geometry").innerHTML = table(["Check", "Measured result"], Object.entries(model.spatial).map(([key, value]) => [escapeHtml(titleCase(key)), typeof value === "number" ? format(value, 5) : value === true ? "Passed" : key === "physical_coordinates_verified" ? "Not checked: physical affine unavailable" : "Failed"]));
  byId("footerMeta").textContent = "UniBrain " + report.upstream_commit.slice(0, 8) + " · seed " + report.seed;
}

function initialize() {
  byId("closeDialog").onclick = () => byId("detailDialog").close();
  byId("detailDialog").addEventListener("click", event => {
    if (event.target !== byId("detailDialog")) return;
    const r = event.target.getBoundingClientRect();
    if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) event.target.close();
  });
  byId("lossKind").onchange = renderLoss;
  byId("gradientTask").onchange = renderGradients;
  byId("sliceIndex").oninput = renderVolume;
  byId("crosshair").onchange = renderVolume;
  byId("previousSlice").onclick = () => { byId("sliceIndex").stepDown(); renderVolume(); };
  byId("nextSlice").onclick = () => { byId("sliceIndex").stepUp(); renderVolume(); };
  document.querySelectorAll("[data-stage]").forEach(button => button.onclick = () => {
    imageStage = button.dataset.stage;
    document.querySelectorAll("[data-stage]").forEach(node => { node.classList.toggle("selected", node === button); node.setAttribute("aria-pressed", node === button); });
    renderVolume();
  });
  byId("download").onclick = () => {
    const { previews, ...data } = report || {};
    const payload = liveRun?.phase !== "complete" ? { status: liveRun?.phase, live: liveRun } : { ...data, live: liveRun };
    const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }));
    const anchor = document.createElement("a");
    anchor.href = url; anchor.download = "unibrain_experiment.json"; anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
  };
  if (report?.candidate) renderResults();
  else byId("empty").hidden = false;
}

initialize();
