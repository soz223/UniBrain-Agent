"use strict";

let report = JSON.parse(document.getElementById("reportData").textContent);
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
  byId("metricsPanel").hidden = true;
  byId("gradientPanel").hidden = true;
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
  byId("snapshotLabel").textContent = runBusy ? "◉ Training in progress" : failed ? "! Run interrupted" : "✓ Last run completed";
  const phases = ["preparing", "training", "audit", "validation", "saving"];
  const position = finished ? phases.length : value.phase === "coordination" ? 1 : Math.max(0, phases.indexOf(value.phase));
  byId("runPhases").innerHTML = ["Load pipeline", "Train", "Check gradients", "Validate", "Save"].map((label, i) => '<li class="' + (i < position ? "done" : i === position && !failed ? "current" : "") + '">' + (i < position ? "✓ " : "") + label + '</li>').join("");
  const stats = value.resources || {};
  const gpuNote = Number.isFinite(stats.gpu_allocated_mb) ? " · GPU allocated " + format(stats.gpu_allocated_mb / 1024, 2) + " GB" : "";
  const history = value.history || [];
  const completed = value.completed_steps ?? history.at(-1)?.step ?? 0;
  byId("stepCount").textContent = completed + " / " + value.steps + " updates";
  byId("runtimeStats").innerHTML = [["Device", stats.device || "Pending", stats.threads ? stats.threads + " threads" : ""], ["Worker CPU", Number.isFinite(stats.cpu_percent) ? format(stats.cpu_percent, 1) + "%" : "…", "of total CPU"], ["Worker memory", Number.isFinite(stats.rss_mb) ? format(stats.rss_mb / 1024, 2) : "…", "GB resident"], ["Updates", completed + " / " + value.steps, finished ? "completed" : "optimizer steps"]].map(([label, number, unit]) => '<div class="runtime-stat"><span>' + label + '</span><strong>' + number + '</strong><br><small>' + unit + '</small></div>').join("");
  const current = value.modules?.find(item => item.module === value.current_module);
  const agent = runBusy && value.agents?.find(item => item.status === "running");
  let activity = current ? (modules[current.label] || current.label) + " is running" : agent ? titleCase(agent.role) + " agent is thinking" : value.phase === "training" ? "Computing gradients and updating weights" : value.phase === "validation" ? "Validation " + value.validation?.completed + " / " + value.validation?.total : finished ? "Checkpoint and outputs saved" : failed ? "Execution stopped" : "Waiting for the next stage";
  byId("currentActivity").innerHTML = escapeHtml(activity) + '<small>' + (finished ? "Resources: last observed worker sample." : "Last reported worker metrics · status polled every second.") + escapeHtml(gpuNote) + '</small>';
  const policy = [...history].reverse().find(step => step.gradient_policy)?.gradient_policy;
  byId("policyLive").hidden = !policy;
  if (policy) byId("policyLive").innerHTML = '<strong>Gradient policy · ' + policy.policy_updates + ' learned updates</strong>' +
    tasks.map(task => '<div><span>' + titleCase(task) + '</span><b>' + format(policy.weights[task], 3) + '</b></div>').join('');
  renderLoss();
  renderCalls(value);
  renderJournal(value.events || []);
  window.dispatchEvent(new CustomEvent("trainingtelemetry", {detail:value}));
}

function renderLoss() {
  const key = byId("lossKind").value;
  const history = (liveRun?.history || []).filter(step => Number.isFinite(key === "weighted_loss" ? step.weighted_loss : step.losses?.[key]));
  if (!history.length) {
    const finished = ["complete", "failed"].includes(liveRun?.phase);
    byId("lossChart").innerHTML = finished ? '<div class="chart-empty compact"><span>Loss was not recorded in this run</span></div>' : '<div class="chart-empty"><span>Waiting for a loss measurement</span><small>The curve appears as recorded optimizer steps arrive.</small></div>';
    byId("lossSummary").textContent = finished ? (liveRun?.history?.length || 0) + " optimizer updates recorded" : "Waiting for training telemetry";
    return;
  }
  const values = history.map(step => key === "weighted_loss" ? step.weighted_loss : step.losses[key]);
  const low = Math.min(...values), high = Math.max(...values), pad = Math.max((high - low) * .18, Math.abs(high) * .02, .01);
  const minimum = low - pad, maximum = high + pad;
  const firstStep = history[0].step, lastStep = history.at(-1).step;
  const x = i => 48 + (history[i].step - firstStep) * 478 / Math.max(1, lastStep - firstStep);
  const y = n => 155 - (n - minimum) * 128 / (maximum - minimum);
  const grid = Array.from({ length: 4 }, (_, i) => {
    const n = minimum + (maximum - minimum) * i / 3;
    return '<path d="M48 ' + y(n) + 'H532" stroke="#37413b" stroke-width=".7"/><text x="40" y="' + (y(n) + 3) + '" text-anchor="end" fill="#95a69b">' + n.toFixed(2) + '</text>';
  }).join("");
  const line = '<polyline points="' + values.map((v, i) => x(i) + ',' + y(v)).join(' ') + '" fill="none" stroke="#a6d8c7" stroke-width="2"/>';
  const points = values.map((v, i) => '<circle cx="' + x(i) + '" cy="' + y(v) + '" r="3.5" fill="#a6d8c7"><title>Step ' + history[i].step + ': ' + format(v, 5) + '</title></circle>').join("");
  byId("lossChart").innerHTML = '<svg viewBox="0 0 550 185" role="img" aria-label="Selected model training loss by optimizer step">' + grid + line + points + '<text x="48" y="179" fill="#89988f">STEP ' + firstStep + '</text><text x="528" y="179" text-anchor="end" fill="#89988f">STEP ' + lastStep + '</text></svg>';
  byId("lossSummary").textContent = "Latest " + format(values.at(-1), 4) + (key === "weighted_loss" ? " · includes current loss weights" : "");
}

function renderCalls(value) {
  const calls = value.modules || [];
  const agentCount = (value.agents || []).length;
  byId("callSummary").textContent = calls.length + " modules · " + agentCount + (agentCount === 1 ? " agent" : " agents");
  byId("modelCalls").innerHTML = calls.length ? table(["Module", "Forward calls", "Total time", "State"], calls.map(item => [escapeHtml(modules[item.label] ? "UniBrain " + modules[item.label] : item.label), item.calls, format(item.seconds, 2) + "s", item.status === "started" ? runBusy ? "Running" : "Interrupted" : "Completed"])) : '<p class="disclosure-note">Model calls will appear as the pipeline starts.</p>';
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
    const response = await backendFetch("/api/status");
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
      await loadLatestReport();
      reloadPending = false;
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
      const response = await backendFetch("/api/volume?" + query, { signal: volumeAbort.signal });
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
  if (report.profile === "original") {
    const policy = report.candidate.coordination?.mode === "taskforce";
    byId("gradientTask").hidden = true;
    byId("gradientDisclosure").textContent = policy ? "Policy gradient measurements" : "Original gradient boundaries";
    byId("gradientMeasurementNote").textContent = policy ? "The task Gram matrix uses bounded parameter probes. Module updates are sampled during Adam steps." : "Module updates are sampled during Adam steps. Loss by module connectivity is measured separately for custom adapter runs.";
    const updates = report.native.updates;
    byId("gradientExplanation").textContent = policy ? "The original model received policy-weighted gradients. These modules had observed parameter updates; the sampled Gram matrix is available below." : "Original training preserved. These modules had observed parameter updates; this is not a loss by module connectivity audit.";
    byId("learningPaths").innerHTML = Object.entries(modules).filter(([key]) => updates.some(step => step.modules_with_observed_updates.includes(key))).map(([key, label]) => '<div class="learning-chip active"><strong>' + label + '</strong><span>Updated in ' + updates.filter(step => step.modules_with_observed_updates.includes(key)).length + ' optimizer steps</span></div>').join("");
    const gram = report.candidate.coordination?.latest?.gram;
    byId("gradientMatrix").innerHTML = (policy && gram ? table(["Task", ...tasks.map(titleCase)], tasks.map((task, i) => [titleCase(task), ...tasks.map((_, j) => scientific(gram[i]?.[j]))])) : "") + '<p class="disclosure-note">The original graph retains nearest label sampling and discrete edge augmentation. Classification gradients to upstream image modules are not asserted. The unused predicted parcellation head is preserved.</p>';
    return;
  }
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
  if (control.mode === "taskforce") {
    const steps = report.candidate.history.filter(step => step.gradient_policy);
    const latest = control.latest || steps.at(-1)?.gradient_policy;
    const rows = steps.map(step => [step.step, ...tasks.map(task => format(step.gradient_policy.weights[task], 3)),
      step.gradient_policy.reward === null ? "Pending" : format(step.gradient_policy.reward, 4), step.gradient_policy.policy_updates]);
    const gram = latest?.gram || [];
    showDetails("Learned gradient policy", '<p>Four actors choose loss weights from task losses and a gradient Gram matrix measured on ' +
      (report.native.gradient_probes?.parameter_values || 0).toLocaleString() + ' sampled trainable parameter values across ' +
      escapeHtml((report.native.gradient_probes?.modules || []).map(titleCase).join(', ')) + '. Centralized critics update the actors from observed training rewards.</p>' +
      table(["Step", ...tasks.map(titleCase), "Reward", "Policy updates"], rows) +
      '<h3>Latest task gradient Gram matrix</h3>' + table(["Task", ...tasks.map(titleCase)], tasks.map((task, i) => [titleCase(task), ...tasks.map((_, j) => scientific(gram[i]?.[j]))])) +
      '<p>Weights change the backward pass of the original UniBrain model. Validation and inference use its trained checkpoint. The policy requires labeled losses and does not run during unlabeled inference.</p>' +
      '<p>Policy checkpoint: ' + escapeHtml(control.checkpoint || "Unavailable") + '</p>', "TRAINING POLICY");
    return;
  }
  const rows = tasks.map(task => [titleCase(task), scientific(control.diagnostic.task_norms[task]), format(control.weights_before[task]), format(control.weights_after[task])]);
  const shared = control.diagnostic.shared_gradient_cosines.map(item => [item.tasks.map(titleCase).join(" / "), item.shared_parameter_tensors, format(item.cosine)]);
  const reviews = (control.roles || []).map(role => '<h3>' + escapeHtml(titleCase(role.role)) + '</h3><p>' + escapeHtml(role.summary) + '</p><p>' + escapeHtml(role.recommendation) + '</p>').join('');
  showDetails("Training coordination", '<p>' + escapeHtml(control.rationale) + '</p>' + table(["Task", "Gradient norm", "Weight before", "Weight after"], rows) + reviews + '<h3>Gradients on shared parameters</h3>' + table(["Tasks", "Shared tensors", "Cosine"], shared) + '<p>Decision before optimizer update 2. Actions: retain loss weights or apply bounded gradient norm balancing. Roles use prompts; a TaskForce reinforcement learning policy is not implemented.</p>', "STEP " + control.step);
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
  if (report.live) return { ...report.live, history: report.candidate.history, steps: report.candidate.steps };
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
  byId("empty").hidden = true;
  byId("results").hidden = false;
  byId("gradientPanel").hidden = false;
  byId("metricsPanel").hidden = false;
  byId("metricsPanel").open = true;
  byId("runDate").textContent = new Date(report.created).toLocaleString("en-US", { dateStyle: "medium", timeStyle: "short" });
  renderLive(savedTelemetry());
  renderVolume();
  renderGradients();
  const control = model.coordination;
  byId("trainingControl").innerHTML = control?.mode === "taskforce" ?
    '<div><strong>Task-wise gradient policy</strong><span>' + control.policy_updates + ' actor-critic updates · policy-trained UniBrain checkpoint used for inference</span></div><button id="coordinationDetails" class="text-button">Inspect policy ↗</button>' :
    control ? '<div><strong>Agent training decision</strong><span>Step ' + control.step + ' · ' + (control.action === "keep_weights" ? "Kept loss weights" : "Balanced gradient magnitudes") + '</span></div><button id="coordinationDetails" class="text-button">Inspect decision ↗</button>' : '<span>Original loss weights and Adam optimizer preserved.</span>';
  if (control) byId("coordinationDetails").onclick = coordinationDetails;
  const finalLoss = model.history.at(-1)?.weighted_loss;
  byId("validationMetrics").innerHTML = [["Extraction Dice", validation.extraction_dice], ["Tissue Dice", validation.tissue_dice], ["Classification accuracy", validation.accuracy], report.profile === "original" ? Number.isFinite(finalLoss) ? ["Final training loss", finalLoss] : ["Optimizer updates", model.steps] : ["Reference loss", validation.reference_loss]].map(([label, value]) => '<div class="metric"><span>' + label + '</span><strong>' + (label === "Optimizer updates" ? value : format(value)) + '</strong></div>').join("");
  byId("dataCaveat").textContent = validation.count + " validation samples. These execution checks do not establish diagnostic performance.";
  const splits = report.data.splits;
  if (report.profile === "original") {
    byId("modelFacts").innerHTML = '<p class="facts">Original UniBrain · ' + report.native.parameters.toLocaleString() + ' parameters · ' + report.epochs + ' epochs<br>' + splits.train.count + ' training samples · ' + splits.val.count + ' validation samples<br>Full widths · five registration stages · ' + (control?.mode === "taskforce" ? "learned task gradient weighting" : "original loss weights") + ' · Adam<br>Source files unchanged: ' + escapeHtml(report.upstream_commit) + '</p><p class="disclosure-note">Execution and checkpoint verification do not establish the paper accuracy. Dataset reads use memory mapping. ' + (control?.mode === "taskforce" ? "Training gradients are coordinated by the selected policy; validation follows the original model path." : "Original trainer behavior, including mode handling, is preserved.") + '</p>';
    byId("geometry").innerHTML = table(["Original workflow check", "Result"], Object.entries(report.native.audit.checks).map(([key, value]) => [titleCase(key), value ? "Passed" : "Failed"]));
    byId("checksTitle").textContent = "Execution checks";
    byId("footerMeta").textContent = "Original UniBrain " + report.upstream_commit.slice(0, 8);
    return;
  }
  byId("modelFacts").innerHTML = '<p class="facts">Compact research adapters · ' + model.steps + ' optimizer updates · seed ' + report.seed + '<br>' + splits.train.count + ' training samples · ' + splits.val.count + ' validation samples<br>Original UniBrain source: ' + escapeHtml(report.upstream_commit.slice(0, 12)) + '<br>Checkpoint: ' + escapeHtml(model.checkpoint) + '<br>Run: ' + escapeHtml(report.run_dir.split(/[\\/]/).at(-1)) + '</p><p class="disclosure-note">This saved experiment used compact stage adapters. Select New UniBrain run to restore the full original workflow.</p>';
  byId("geometry").innerHTML = table(["Check", "Measured result"], Object.entries(model.spatial).map(([key, value]) => [escapeHtml(titleCase(key)), typeof value === "number" ? format(value, 5) : value === true ? "Passed" : key === "physical_coordinates_verified" ? "Not checked: physical affine unavailable" : "Failed"]));
  byId("footerMeta").textContent = "UniBrain " + report.upstream_commit.slice(0, 8) + " · seed " + report.seed;
}

async function loadLatestReport() {
  const data = await requestJson("/api/report");
  report = data.report;
  sliceCache.clear();
  if (report?.candidate) renderResults();
}

async function initialize() {
  document.querySelectorAll('.sidebar nav a').forEach(link => link.onclick = () => {
    const target = document.querySelector(link.getAttribute('href'));
    if (target?.tagName === 'DETAILS') target.open = true;
  });
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
  if (await backendReady) {
    try {await loadLatestReport();} catch(error) {byId("empty").textContent = error.message;}
  }
}

initialize();
