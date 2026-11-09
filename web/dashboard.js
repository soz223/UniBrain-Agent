"use strict";

const report = JSON.parse(document.getElementById("reportData").textContent);
const byId = id => document.getElementById(id);
const escapeHtml = value => String(value ?? "").replace(/[&<>"']/g, character =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[character]);
const format = (value, digits = 3) => Number.isFinite(value) ? value.toFixed(digits) : "Not defined";
const scientific = value => Number.isFinite(value) ? value.toExponential(1) : "Not defined";
const titleCase = value => value.charAt(0).toUpperCase() + value.slice(1).replaceAll("_", " ");
const icons = {
  grid: '<rect x="3" y="3" width="6" height="6" rx="1"/><rect x="15" y="3" width="6" height="6" rx="1"/><rect x="3" y="15" width="6" height="6" rx="1"/><rect x="15" y="15" width="6" height="6" rx="1"/>',
  flow: '<rect x="2" y="8" width="5" height="8" rx="1"/><rect x="17" y="8" width="5" height="8" rx="1"/><path d="M7 12h10m-4-3 3 3-3 3"/>',
  pulse: '<path d="M2 12h4l3-7 5 14 3-7h5"/>',
  chart: '<path d="M3 3v18h18M6 15l5-4 4 2 6-7"/>',
  nodes: '<circle cx="5" cy="5" r="3"/><circle cx="19" cy="12" r="3"/><circle cx="5" cy="19" r="3"/><path d="m8 6 8 5m-8 7 8-5M5 8v8"/>',
  book: '<path d="M12 5C9 3 5 3 2 4v15c3-1 7-1 10 1 3-2 7-2 10-1V4c-3-1-7-1-10 1Zm0 0v15"/>',
  cube: '<path d="m12 2 9 5v10l-9 5-9-5V7l9-5Zm-9 5 9 5 9-5m-9 5v10"/>',
  play: '<path d="m7 4 13 8-13 8Z"/>',
  branch: '<path d="M5 3v11c0 4 3 5 7 5h7M5 9h14m-4-4 4 4-4 4"/>',
  download: '<path d="M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 6v6l4 2"/>'
};
const modules = {
  denoise_net: ["DNS", "Denoising"], bias_net: ["BIAS", "Bias correction"], norm_net: ["NORM", "Normalization"],
  ext_net: ["EXT", "Extraction"], reg_net: ["REG", "Registration"],
  seg_net: ["SEG", "Segmentation"], par_net: ["PAR", "Parcellation head"],
  feature_net: ["ROI", "ROI encoder"], graph_gen: ["GRAPH", "Graph generation"],
  gnn_pred: ["CLS", "Classifier"]
};
const tasks = { extraction: "EXT", registration: "REG", segmentation: "SEG", classification: "CLS" };
let branch = "candidate";
let imageStage = "input";
let selectedCell = { task: "classification", module: "reg_net" };
let graphCell = null;
let pollTimer = null;

function paintIcons(root = document) {
  root.querySelectorAll("[data-icon]").forEach(node => {
    node.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true">' + (icons[node.dataset.icon] || icons.grid) + "</svg>";
  });
}

function table(headers, rows, selectedColumn = -1) {
  const header = headers.map((value, index) => '<th scope="col"' +
    (index === selectedColumn ? ' class="selected-column"' : "") + ">" + escapeHtml(value) + "</th>").join("");
  const body = rows.map(row => "<tr>" + row.map((value, index) =>
    "<td" + (index === selectedColumn ? ' class="selected-column"' : "") + ">" + value + "</td>"
  ).join("") + "</tr>").join("");
  return '<table class="data-table"><thead><tr>' + header + "</tr></thead><tbody>" + body + "</tbody></table>";
}

function showDetails(title, body, eyebrow = "EXPERIMENT DETAILS") {
  byId("dialogTitle").textContent = title;
  byId("dialogEyebrow").textContent = eyebrow;
  byId("dialogBody").innerHTML = body;
  byId("detailDialog").showModal();
}

function sourceLink(url, label) {
  if (!/^https:\/\//.test(url || "")) return "";
  return '<a href="' + escapeHtml(url) + '" target="_blank" rel="noreferrer">' + escapeHtml(label) + "</a>";
}

function renderStats() {
  const linked = Object.values(report[branch].gradients.matrix.classification).filter(cell => cell.state === "active").length;
  const stats = [
    ["Sample data", report.data.splits.train.count + " / " + report.data.splits.val.count, "train / val", "Test set untouched", "cube", ""],
    ["Training updates", report.baseline.steps + report.candidate.steps, "steps", "Matched budget · 2 configurations", "chart", ""],
    ["Classification paths", String(linked).padStart(2, "0"), "modules", "Nonzero parameter gradients · " + branch, "pulse", "cyan"],
    ["Compute ratio", format(report.gate.cost_ratio, 2) + "×", "", "Candidate / baseline · limit " + report.gate.thresholds.max_cost_ratio.toFixed(1) + "×", "clock", "amber"]
  ];
  byId("stats").innerHTML = stats.map(item =>
    '<article class="stat"><div class="stat-title">' + item[0] + '<span data-icon="' + item[4] +
    '"></span></div><div class="stat-value ' + item[5] + '">' + escapeHtml(item[1]) +
    "<small>" + item[2] + '</small></div><div class="stat-caption">' + escapeHtml(item[3]) + "</div></article>"
  ).join("");
  paintIcons(byId("stats"));
}

function renderPipeline() {
  const config = report[branch].model.config;
  const selected = config.stages;
  const stages = [
    ["ext_net", "Extraction", selected?.extraction === "M03" ? "ERNet · soft mask" : config.extraction === "synthstrip" ? "SynthStrip · soft mask" : "UniBrain U-Net", selected?.extraction === "M03" || config.extraction === "synthstrip"],
    ["reg_net", "Registration", selected?.registration === "M06" ? "Affine + VoxelMorph" : selected?.registration === "M07" ? "Affine + LapIRN" : "Affine · " + config.registration_steps + " stage", selected?.registration && selected.registration !== "unibrain"],
    ["atlas", "Parcellation", selected?.parcellation === "M09" ? "FastSurfer · AAL head" : selected?.parcellation === "M10" ? "HighResNet · AAL head" : config.soft_atlas ? "Soft atlas probabilities" : "Nearest labels", config.soft_atlas],
    ["feature_net", "ROI features", "Original ROI encoder", false],
    ["graph_gen", "Brain graph", selected?.graph === "M11" ? "FBNetGen pair MLP" : config.continuous_edges ? "Continuous weights" : "Discrete perturbation", config.continuous_edges],
    ["gnn_pred", "Classification", selected?.classification === "M12" ? "BrainNetCNN" : selected?.classification === "M13" ? "Brain Network Transformer" : "Original GCN", selected?.classification && selected.classification !== "unibrain"]
  ];
  byId("pipelineLabel").textContent = branch.toUpperCase() + " CONFIGURATION";
  byId("pipeline").innerHTML = stages.map((stage, index) =>
    (index ? '<span class="pipeline-arrow" aria-hidden="true">→</span>' : "") +
    '<button class="pipeline-stage ' + (stage[3] ? "changed" : "") + '" data-module="' + stage[0] +
    '" aria-label="Inspect ' + stage[1] + '"><span class="stage-kind"><span>' +
    String(index + 1).padStart(2, "0") + "</span><span>" + (stage[3] ? "ADAPTED" : "UNIBRAIN") +
    "</span></span><strong>" + stage[1] + "</strong><small>" + stage[2] + "</small></button>"
  ).join("");
  byId("pipeline").querySelectorAll("button").forEach(button => {
    button.onclick = () => moduleDetails(button.dataset.module);
  });
}

function renderVolume() {
  const preview = report.previews[branch];
  const position = Number(byId("sliceIndex").value);
  const index = preview.slice_indices[position];
  const crosshair = byId("crosshair").checked ? " crosshair" : "";
  byId("volumeImages").innerHTML = ["I", "J", "K"].map((plane, axis) =>
    '<figure class="volume-plane' + crosshair + '" style="--cross-x:' + (index / 95 * 100) +
    '%;--cross-y:' + ((95 - index) / 95 * 100) + '%"><img src="' + preview.planes[imageStage][axis][position] +
    '" alt="' + escapeHtml(titleCase(imageStage)) + ", array " + plane + " plane, index " + index +
    '"><figcaption>' + plane + ' PLANE</figcaption><span class="plane-index">' + plane + " = " + index +
    "</span></figure>"
  ).join("");
  byId("sliceValue").textContent = index + " / 95";
  byId("sampleId").textContent = "POST-TRAIN AUDIT";
}

function gradientDetail() {
  const cell = report[branch].gradients.matrix[selectedCell.task][selectedCell.module] || {state: "inactive", parameter_tensors: 0, connected_tensors: 0, nonzero_tensors: 0, norm: 0};
  const name = modules[selectedCell.module][1];
  const detail = selectedCell.module === "graph_gen" && !cell.parameter_tensors
    ? "This module has no parameters. Its output gradient is inspected in the Brain graph panel."
    : selectedCell.module === "par_net" && !cell.parameter_tensors
      ? "The unused predicted-parcellation head is explicitly frozen in this atlas-based configuration."
      : cell.connected_tensors + "/" + cell.parameter_tensors + " parameter tensors connected · " +
        cell.nonzero_tensors + " nonzero · L2 norm " + scientific(cell.norm);
  byId("gradientDetail").innerHTML = "<strong>" + titleCase(selectedCell.task) + " → " + name +
    " · " + cell.state + "</strong>" + escapeHtml(detail);
}

function renderGradients() {
  const matrix = report[branch].gradients.matrix;
  const headers = Object.entries(modules).map(([name, item]) =>
    '<th scope="col" title="' + item[1] + '">' + item[0] + "</th>").join("");
  const rows = Object.entries(tasks).map(([task, short]) => {
    const cells = Object.keys(modules).map(name => {
      const cell = matrix[task][name] || {state: "inactive", norm: 0};
      const focused = selectedCell.task === task && selectedCell.module === name;
      const label = task + " to " + modules[name][1] + ": " + cell.state;
      const value = cell.state === "active" ? cell.norm.toExponential(0) : cell.state === "zero" ? "0" : "·";
      return '<td><button class="gradient-cell ' + cell.state + (focused ? " focused" : "") +
        '" data-task="' + task + '" data-module="' + name + '" aria-label="' + label +
        '" title="' + label + " · norm " + scientific(cell.norm) + '">' + value + "</button></td>";
    }).join("");
    return '<tr><th scope="row" title="' + titleCase(task) + '">' + short + "</th>" + cells + "</tr>";
  }).join("");
  byId("gradientMatrix").innerHTML = '<table class="gradient-matrix"><thead><tr><th scope="col">LOSS</th>' +
    headers + "</tr></thead><tbody>" + rows + "</tbody></table>";
  byId("gradientMatrix").querySelectorAll("button").forEach(button => {
    button.onclick = () => {
      selectedCell = { task: button.dataset.task, module: button.dataset.module };
      byId("gradientMatrix").querySelectorAll(".focused").forEach(cell => cell.classList.remove("focused"));
      button.classList.add("focused");
      gradientDetail();
    };
  });
  gradientDetail();
}

function renderLoss() {
  const key = byId("lossKind").value;
  const value = step => key === "weighted_loss" ? step.weighted_loss : step.losses[key];
  const series = [report.baseline.history, report.candidate.history];
  const values = series.flatMap(history => history.map(value));
  const low = Math.min(...values);
  const high = Math.max(...values);
  const padding = Math.max((high - low) * 0.15, 0.001);
  const minimum = low - padding;
  const maximum = high + padding;
  const count = Math.max(...series.map(history => history.length));
  const x = step => 40 + step * 330 / Math.max(1, count - 1);
  const y = amount => 150 - (amount - minimum) * 120 / (maximum - minimum);
  const grid = Array.from({ length: 4 }, (_, index) => {
    const level = minimum + (maximum - minimum) * index / 3;
    return '<path d="M40 ' + y(level) + "H376" + '" stroke="#293540" stroke-dasharray="3 4"/>' +
      '<text x="32" y="' + (y(level) + 3) + '" text-anchor="end">' + level.toFixed(2) + "</text>";
  }).join("");
  const lines = series.map((history, index) => {
    const color = index ? "#62c7ce" : "#839bbd";
    const points = history.map((step, i) => x(i) + "," + y(value(step))).join(" ");
    return '<polyline points="' + points + '" fill="none" stroke="' + color + '" stroke-width="2"/>' +
      history.map((step, i) => '<circle cx="' + x(i) + '" cy="' + y(value(step)) + '" r="3" fill="' + color +
        '" stroke="#14191f" stroke-width="1.4"><title>' + (index ? "Candidate" : "Baseline") +
        " · step " + step.step + " · " + format(value(step), 5) + "</title></circle>").join("");
  }).join("");
  byId("lossChart").innerHTML = '<svg viewBox="0 0 400 178" role="img" aria-label="' +
    escapeHtml(byId("lossKind").selectedOptions[0].textContent) + ' loss by training step">' + grid + lines +
    '<text x="40" y="171">STEP 1</text><text x="370" y="171" text-anchor="end">STEP ' + count + "</text></svg>";
  byId("lossSummary").textContent = format(value(report[branch].history.at(-1)), 3) + " FINAL";
}

function renderAgents() {
  const names = { research: "Research agent", composition: "Composition agent", coordination: "Training coordinator", review: "Experiment reviewer" };
  const summary = {
    research: report.plan.extraction === "synthstrip" ? "Selected SynthStrip + connection adapters" : "Selected UniBrain configuration",
    coordination: report.candidate.coordination?.action === "keep_weights" ? "Kept current loss weights" : "Applied bounded gradient balancing",
    review: report.review.recommendation === "revert" ? "Recommended baseline restoration" : titleCase(report.review.recommendation) + " recommendation"
  };
  byId("agentBackend").textContent = report.backend === "codex" ? "CODEX LLM" : "RULES DEMO";
  byId("agentFeed").innerHTML = report.agent_calls.map((call, index) =>
    '<div class="agent-item"><span class="agent-symbol">' + ["R", "C", "V"][index] +
    '</span><div><div class="agent-role">' + escapeHtml(names[call.role] || call.role) +
    '</div><div class="agent-action">' + escapeHtml(summary[call.role] || call.decision.rationale) +
    '</div></div><time>' + format(call.seconds, 1) + " s</time></div>"
  ).join("");
}

function renderGraph() {
  const canvas = byId("graphCanvas");
  const context = canvas.getContext("2d");
  const values = report.previews[branch].graph_values;
  const size = values.length;
  const cellSize = canvas.width / size;
  values.forEach((row, i) => row.forEach((weight, j) => {
    const strength = Math.min(1, Math.abs(weight));
    const start = [13, 26, 35];
    const end = weight >= 0 ? [91, 193, 204] : [196, 104, 111];
    context.fillStyle = "rgb(" + start.map((amount, k) => Math.round(amount + (end[k] - amount) * strength)).join(",") + ")";
    context.fillRect(j * cellSize, i * cellSize, cellSize + 0.1, cellSize + 0.1);
  }));
  if (graphCell) {
    context.strokeStyle = "#e0f7f9";
    context.lineWidth = 1;
    context.strokeRect(graphCell[1] * cellSize, graphCell[0] * cellSize, cellSize, cellSize);
    const label = index => report.data.atlas_label_values?.[index + 1] ?? index;
    byId("graphReadout").textContent = label(graphCell[0]) + " ↔ " + label(graphCell[1]) +
      " · " + format(values[graphCell[0]][graphCell[1]], 4);
  } else {
    byId("graphReadout").textContent = size + " × " + size + " · hover to inspect edge weight";
  }
  const edge = report[branch].gradients.classification_to_graph_weights;
  byId("graphGradient").textContent = "Classification → edge weights: " +
    (edge.connected ? "connected · norm " + scientific(edge.norm) : "disconnected");
  canvas.tabIndex = 0;
}

function renderComparison() {
  const baseline = report.baseline;
  const candidate = report.candidate;
  const cell = value => '<span class="mono">' + value + "</span>";
  const measurements = [
    ["Reference loss ↓", format(baseline.validation.reference_loss), format(candidate.validation.reference_loss)],
    ["Brain mask Dice ↑", format(baseline.validation.extraction_dice), format(candidate.validation.extraction_dice)],
    ["Tissue Dice ↑", format(baseline.validation.tissue_dice), format(candidate.validation.tissue_dice)],
    ['Diagnosis accuracy <span class="value-note">· smoke test</span>', format(baseline.validation.accuracy), format(candidate.validation.accuracy)],
    ["Training compute ↓", format(baseline.training_seconds, 2) + " s", format(candidate.training_seconds, 2) + " s"]
  ];
  byId("comparison").innerHTML = table(["Validation measure", "Baseline", "Candidate"],
    measurements.map(row => [row[0], cell(row[1]), cell(row[2])]), branch === "baseline" ? 1 : 2);
  const singleClass = Object.keys(report.data.splits.train.class_counts).length < 2;
  byId("dataCaveat").textContent = singleClass
    ? "All " + report.data.splits.train.count + " training samples share one diagnosis label. Functional checks only; no classification-performance claim."
    : "Small-sample functional validation only. This experiment does not establish clinical performance.";
}

function renderGate() {
  byId("gateBadge").textContent = report.gate.decision.toUpperCase();
  byId("gateBadge").className = "badge " + (report.gate.decision === "keep" ? "good" : "bad");
  const gates = [
    ["Reference loss ratio", report.gate.loss_ratio, report.gate.thresholds.max_loss_ratio],
    ["Training compute ratio", report.gate.cost_ratio, report.gate.thresholds.max_cost_ratio]
  ];
  byId("gateMeters").innerHTML = gates.map(([name, value, limit]) => {
    const maximum = Math.max(value, limit) * 1.2;
    return '<div class="meter ' + (value > limit ? "failed" : "") +
      '"><div class="meter-heading"><span>' + name + "</span><strong>" + format(value, 2) +
      '×</strong></div><div class="meter-track"><div class="meter-fill" style="width:' +
      (value / maximum * 100) + '%"></div><span class="meter-limit" style="left:' +
      (limit / maximum * 100) + '%"></span></div><div class="meter-caption"><span>0×</span><span>LIMIT ' +
      format(limit, 2) + "×</span></div></div>";
  }).join("");
  const control = report[branch].coordination;
  const weights = control?.weights_after || report[branch].history.at(-1).weights;
  byId("controlAction").textContent = control ? control.action.replaceAll("_", " ").toUpperCase() : "FIXED BASELINE";
  byId("weights").innerHTML = Object.entries(tasks).map(([task, short]) =>
    '<div class="weight"><label>' + short + "</label><strong>" + format(weights[task], 2) + "</strong></div>"
  ).join("");
}

function renderMethods() {
  if (typeof library !== "undefined" && library && report[branch].model.config.stages) {
    const choices = report[branch].model.config.stages;
    byId("methodCards").innerHTML = library.stages.filter(stage => choices[stage.id]?.startsWith("M")).map(stage => {
      const option = stage.options.find(item => item.id === choices[stage.id]);
      return '<article class="method-card"><header><h3>' + escapeHtml(option.label.split(" (")[0]) + '</h3>' +
        sourceLink(option.research.code, "↗") + '</header><p>' + escapeHtml(option.label) + '</p><div class="method-stage">' +
        escapeHtml(stage.name) + '</div></article>';
    }).join("");
    return;
  }
  const labels = { unibrain: "UniBrain", synthstrip: "SynthStrip", soft_atlas: "Soft atlas transfer", continuous_edges: "Continuous edges" };
  byId("methodCards").innerHTML = report.research.cards.map(card =>
    '<article class="method-card"><header><h3>' + escapeHtml(labels[card.id] || card.id) +
    "</h3>" + sourceLink(card.code, "↗") + "</header><p>" + escapeHtml(card.publication) +
    '</p><div class="method-stage">' + escapeHtml(card.stage) + "</div></article>"
  ).join("");
}

function renderGeometry() {
  const checks = report[branch].spatial;
  byId("geometryBadge").textContent = checks.passed ? "PASSED" : "CHECK REQUIRED";
  byId("geometryBadge").className = "badge " + (checks.passed ? "good" : "bad");
  const rows = [
    ["Finite outputs", checks.finite ? "Verified" : "Failed"],
    ["Array shape", checks.shape_preserved ? "Preserved" : "Changed"],
    ["Affine determinant", format(checks.affine_determinant, 4)],
    ["Outside sampling grid", format(checks.out_of_bounds_fraction * 100, 2) + "%"],
    ["Probability sum error", scientific(checks.atlas_probability_sum_error)]
  ];
  if (checks.folding_fraction !== undefined) rows.push(["Grid folding", format(checks.folding_fraction * 100, 3) + "%"],
    ["Inverse cycle error", scientific(checks.inverse_cycle_max_error)]);
  byId("geometry").innerHTML = rows.map(row =>
    '<div class="geometry-row"><span>' + row[0] + "</span><span>" + row[1] + "</span></div>"
  ).join("");
}

function renderEvents() {
  const filter = byId("eventFilter").value;
  const events = report.events.filter(event => filter === "all" || event.kind === filter).toReversed();
  byId("events").innerHTML = events.map(event =>
    '<div class="event" data-kind="' + escapeHtml(event.kind) + '"><time>' + format(event.seconds, 1) +
    ' s</time><span class="event-kind">' + escapeHtml(event.kind) + "</span><span>" + escapeHtml(event.message) + "</span></div>"
  ).join("");
}

function renderBranch() {
  document.querySelectorAll("[data-branch]").forEach(button => {
    const selected = button.dataset.branch === branch;
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
  renderStats();
  renderPipeline();
  renderVolume();
  renderGradients();
  renderLoss();
  renderGraph();
  renderComparison();
  renderGate();
  renderGeometry();
  renderMethods();
}

function moduleDetails(name) {
  const config = report[branch].model.config;
  if (name === "atlas") {
    if (config.stages && config.stages.parcellation !== "soft_atlas") {
      moduleDetails("par_net");
      return;
    }
    const card = report.research.cards.find(item => item.id === "soft_atlas");
    showDetails("Atlas transfer", "<p>" + (config.soft_atlas
      ? "The fixed atlas is converted to 117 one-hot channels before trilinear sampling in the inverse affine grid."
      : "Nearest-neighbor atlas labels are warped, then converted to one-hot ROI channels.") +
      "</p><p>The template label vocabulary defines the channel order. Border coverage and probability mass are checked separately.</p>" +
      sourceLink(card.code, "Original source"), branch.toUpperCase() + " / CONNECTION");
    return;
  }
  const rows = Object.entries(report[branch].gradients.matrix).map(([task, row]) =>
    [titleCase(task), escapeHtml(row[name]?.state || "inactive"), scientific(row[name]?.norm)]);
  const parameters = report[branch].model.parameters[name] || 0;
  showDetails(modules[name][1], "<p>" + parameters.toLocaleString() + " parameters in this compact configuration.</p>" +
    table(["Loss", "Gradient state", "L2 norm"], rows) +
    "<p>Parameter updates observed: " + (report[branch].updated_parameters[name]?.toLocaleString() || "Not an active trainable module") + ".</p>",
    branch.toUpperCase() + " / MODULE");
}

function decisionDetails() {
  const sections = report.agent_calls.map(call => {
    const decision = call.decision;
    const usage = call.usage;
    return '<section class="detail-section"><h3>' + titleCase(call.role) + " · " + escapeHtml(call.backend) +
      " · " + format(call.seconds, 1) + " s</h3><p>" + escapeHtml(decision.rationale) + "</p>" +
      (decision.limitations ? "<ul>" + decision.limitations.map(item => "<li>" + escapeHtml(item) + "</li>").join("") + "</ul>" : "") +
      (decision.next_experiment ? "<p><strong>Next experiment:</strong> " + escapeHtml(decision.next_experiment) + "</p>" : "") +
      (usage ? '<p class="mono tiny">Input tokens ' + usage.input_tokens.toLocaleString() + " · Output tokens " +
        usage.output_tokens.toLocaleString() + "</p>" : "") + "</section>";
  }).join("");
  showDetails("Agent decision record", sections, "RECORDED DECISIONS / NO SIMULATED ACTIVITY");
}

function dataDetails() {
  const data = report.data;
  const rows = [
    ["Data format", escapeHtml(data.format)], ["Array shape", data.shape.join(" × ")],
    ["Train / validation", data.splits.train.count + " / " + data.splits.val.count],
    ["Training labels", escapeHtml(JSON.stringify(data.splits.train.class_counts))],
    ["Validation labels", escapeHtml(JSON.stringify(data.splits.val.class_counts))],
    ["Atlas channels", data.atlas_label_values.length], ["Test data used", String(data.test_used)],
    ["Seed", report.seed], ["Runtime", escapeHtml(report.environment.torch) + " · " + escapeHtml(report.environment.device)],
    ["UniBrain commit", '<span class="mono">' + escapeHtml(report.upstream_commit) + "</span>"],
    ["Selected checkpoint", escapeHtml(report.selection.checkpoint) + " · " + escapeHtml(report.selection.branch)]
  ];
  showDetails("Data & provenance", table(["Field", "Recorded value"], rows) +
    "<h3>Scope</h3><ul>" + data.warnings.map(text => "<li>" + escapeHtml(text) + "</li>").join("") +
    "</ul><h3>Recorded adaptations</h3><ul>" + report[branch].model.adaptations.map(text => "<li>" + escapeHtml(text) + "</li>").join("") +
    "</ul><h3>Artifact directory</h3><pre>" + escapeHtml(report.run_dir) + "</pre>");
}

function catalogDetails() {
  if (typeof library !== "undefined" && library) {
    const rows = library.stages.flatMap(stage => stage.options.filter(option => option.research).map(option =>
      [escapeHtml(stage.name), escapeHtml(option.label), sourceLink(option.research.code, "Pinned code"),
        option.status === "requires_inputs" ? "Needs mesh inputs" : "Validate the selected combination"]));
    showDetails("Stage research catalog", '<p>Imported from ' + escapeHtml(library.workbook) + '</p>' +
      table(["Stage", "Method", "Source", "Integration"], rows), "15 RESEARCH METHODS");
    return;
  }
  const available = report.research.cards.map(card =>
    '<section class="detail-section"><h3>' + escapeHtml(card.id) + "</h3><p>" + escapeHtml(card.adaptation) +
    "</p><p>" + escapeHtml(card.interface) + "</p><ul>" +
    card.conditions.map(text => "<li>" + escapeHtml(text) + "</li>").join("") +
    "</ul>" + sourceLink(card.code, "Pinned implementation") +
    (card.paper ? " · " + sourceLink(card.paper, "Paper") : "") + "</section>"
  ).join("");
  const pending = table(["Candidate", "Publication", "Status"], report.research.other_candidates.map(item =>
    [escapeHtml(item.name), escapeHtml(item.publication), "Adapter pending"]));
  showDetails("Methods & evidence", "<p>The research agent reviewed the existing catalog and pinned source definitions. Open-ended web discovery is not implemented.</p>" +
    available + "<h3>Literature candidates awaiting integration</h3>" + pending, "RESEARCH CATALOG");
}

function gradientHelp() {
  const gradient = report[branch].gradients;
  showDetails("Reading the gradient audit", "<ul>" + gradient.notes.map(text => "<li>" + escapeHtml(text) + "</li>").join("") +
    "</ul><h3>Shared-parameter gradient cosines</h3>" + table(["Loss pair", "Shared tensors", "Cosine"],
      gradient.shared_gradient_cosines.map(item => [escapeHtml(item.tasks.join(" / ")), item.shared_parameter_tensors, format(item.cosine)])) +
    "<p>These observations describe the measured sample after this short run. A nonzero gradient does not establish a useful training signal.</p>",
    branch.toUpperCase() + " / GRADIENT DIAGNOSTICS");
}

function reviewDetails() {
  showDetails("Acceptance & rollback", "<p>" + escapeHtml(report.review.rationale) + "</p><h3>Hard gate</h3><ul>" +
    report.gate.reasons.map(text => "<li>" + escapeHtml(text) + "</li>").join("") +
    "</ul><p>" + escapeHtml(report.gate.scope) + "</p><h3>Selected artifact</h3><p>" +
    escapeHtml(report.selection.reason) + "</p><pre>" + escapeHtml(report.selection.checkpoint) +
    (report.selection.sha256 ? "\nSHA256 " + escapeHtml(report.selection.sha256) : "") + "</pre>");
}

function connectionDetails() {
  if (report[branch].model.config.stages) {
    showDetails("Pipeline connections", '<p>Image adapters feed extraction and registration. Registration produces forward and inverse grids. Soft region probabilities return to the native array grid before ROI feature extraction.</p>' +
      '<p>Tissue segmentation is an auxiliary branch with ground truth supervision. A learned parcellation head receives AAL supervision. Dense registration receives a velocity smoothness term.</p><ul>' +
      report[branch].model.adaptations.map(text => '<li>' + escapeHtml(text) + '</li>').join("") + '</ul>', "CURRENT COMPOSITION");
    return;
  }
  showDetails("Pipeline connections", "<p>Extraction feeds affine registration. The inverse affine maps template tissue labels and atlas regions into the input image grid.</p>" +
    "<p>The tissue segmentation U-Net reads the input directly and receives its own segmentation loss. It is a parallel branch.</p>" +
    "<p>The ROI encoder receives the original input multiplied by the transferred atlas probabilities. ROI features feed both graph generation and the GCN classifier.</p>" +
    "<p>The horizontal pipeline summarizes these dependencies. It does not imply that every module receives every downstream loss.</p>" +
    "<p>The optional predicted-parcellation head is frozen and skipped in this atlas-based configuration.</p>",
    "DATA DEPENDENCIES / " + branch.toUpperCase());
}

function setFailure(error) {
  byId("progress").hidden = false;
  byId("progressTitle").textContent = "Experiment could not complete";
  byId("error").textContent = error.message || String(error);
  byId("run").disabled = false;
  document.querySelector(".spinner").hidden = true;
  window.dispatchEvent(new Event("experimentfailed"));
}

async function poll() {
  clearTimeout(pollTimer);
  try {
    const response = await fetch("/api/status");
    if (!response.ok) throw new Error("The local service is unavailable.");
    const state = await response.json();
    if (state.running) {
      byId("progress").hidden = false;
      byId("run").disabled = true;
      byId("progressText").textContent = state.message;
      byId("snapshotLabel").textContent = "Experiment running";
      pollTimer = setTimeout(poll, 1500);
    } else if (state.error && state.job === "run") {
      setFailure(new Error(state.error));
    } else if (state.completed && state.job === "run") {
      location.reload();
    } else {
      byId("run").disabled = false;
    }
  } catch (error) {
    setFailure(error);
  }
}

async function startExperiment() {
  if (location.protocol === "file:") {
    showDetails("Open the local workspace", "<p>This is a saved experiment report. To run a new experiment, open " +
      '<a href="http://127.0.0.1:8768">the local dashboard</a> or launch <strong>Launch UniBrain Agent.cmd</strong> from the workspace folder.</p>');
    return;
  }
  byId("progress").hidden = false;
  byId("progressTitle").textContent = "Experiment running";
  byId("progressText").textContent = "Starting the measured comparison. The panels below show the last completed run.";
  byId("error").textContent = "";
  document.querySelector(".spinner").hidden = false;
  byId("run").disabled = true;
  try {
    const response = await fetch("/api/run", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ backend: byId("backend").value, steps: 5 })
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "The experiment could not start.");
    poll();
  } catch (error) {
    setFailure(error);
  }
}

function bindControls() {
  document.querySelectorAll("[data-branch]").forEach(button => {
    button.onclick = () => { branch = button.dataset.branch; graphCell = null; renderBranch(); };
  });
  document.querySelectorAll("[data-stage]").forEach(button => {
    button.onclick = () => {
      imageStage = button.dataset.stage;
      document.querySelectorAll("[data-stage]").forEach(item => {
        item.classList.toggle("selected", item === button);
        item.setAttribute("aria-pressed", String(item === button));
      });
      renderVolume();
    };
  });
  byId("sliceIndex").oninput = renderVolume;
  byId("crosshair").onchange = renderVolume;
  byId("lossKind").onchange = renderLoss;
  byId("eventFilter").onchange = renderEvents;
  byId("allDecisions").onclick = decisionDetails;
  byId("dataDetails").onclick = dataDetails;
  byId("catalogDetails").onclick = catalogDetails;
  byId("gradientHelp").onclick = gradientHelp;
  byId("reviewDetails").onclick = reviewDetails;
  byId("pipelineDetails").onclick = connectionDetails;
  byId("graphCanvas").onpointermove = event => {
    const rect = event.currentTarget.getBoundingClientRect();
    const size = report.previews[branch].graph_values.length;
    graphCell = [Math.min(size - 1, Math.max(0, Math.floor((event.clientY - rect.top) / rect.height * size))),
      Math.min(size - 1, Math.max(0, Math.floor((event.clientX - rect.left) / rect.width * size)))];
    renderGraph();
  };
  byId("graphCanvas").onkeydown = event => {
    const offset = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] }[event.key];
    if (!offset) return;
    event.preventDefault();
    const size = report.previews[branch].graph_values.length;
    graphCell = (graphCell || [0, 0]).map((value, axis) => Math.max(0, Math.min(size - 1, value + offset[axis])));
    renderGraph();
  };
  byId("download").onclick = () => {
    const { previews, ...data } = report;
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "unibrain_" + report.created.replaceAll(":", "") + ".json";
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
  };
}

function initialize() {
  paintIcons();
  byId("run").onclick = startExperiment;
  byId("closeDialog").onclick = () => byId("detailDialog").close();
  byId("detailDialog").addEventListener("click", event => {
    if (event.target === byId("detailDialog")) {
      const rect = event.target.getBoundingClientRect();
      if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) event.target.close();
    }
  });
  document.querySelectorAll(".nav-link[href]").forEach(link => {
    link.onclick = () => {
      document.querySelectorAll(".nav-link.active").forEach(item => item.classList.remove("active"));
      link.classList.add("active");
    };
  });
  if (!report) {
    byId("results").hidden = true;
    byId("empty").hidden = false;
    byId("dataDetails").disabled = true;
  } else {
    byId("backend").value = report.backend;
    byId("runId").textContent = "RUN " + report.run_dir.split(/[\\/]/).at(-1);
    byId("runDate").textContent = new Date(report.created).toLocaleString("en-US", { dateStyle: "medium", timeStyle: "short" });
    byId("statusPill").textContent = report.selection.action === "revert" ? "BASELINE RESTORED" : "CANDIDATE RETAINED";
    byId("statusPill").className = "badge " + (report.selection.action === "revert" ? "" : "good");
    byId("footerMeta").textContent = "UniBrain " + report.upstream_commit.slice(0, 8) +
      " · seed " + report.seed + " · " + format(report.elapsed_seconds, 1) + " s total";
    bindControls();
    renderBranch();
    renderAgents();
    renderMethods();
    renderEvents();
  }
  if (location.protocol !== "file:") {
    fetch("/api/status").then(response => response.json()).then(state => {
      if (state.running && state.job === "run") poll();
    }).catch(() => { byId("snapshotLabel").textContent = "Saved report · service offline"; });
  }
}

initialize();
