"use strict";

const objectiveMethods = {
  synthstrip: {extraction: "M02"}, voxelmorph: {registration: "M06"}, lapirn: {registration: "M07"},
  nnunet: {segmentation: "M08"}, redwgan: {denoising: "M14"}, deepn4: {bias: "M01"},
  ernet: {extraction: "M03", registration: "M03"}, deepnormalize: {normalization: "M04"},
  harmonizing_flows: {normalization: "M05"}, fastsurfer: {parcellation: "M09"},
  nobrainer: {parcellation: "M10"}, fbnetgen: {graph: "M11"}, brainnetcnn: {classification: "M12"},
  bnt: {classification: "M13"}
};

// Mirrors loss_terms, original_losses and the full research training command.
function describeObjective({selection, defaults, stages, build = null, coordination = "single", cycle = null}) {
  const full = Boolean(cycle?.method);
  const choices = full ? {...defaults, ...objectiveMethods[cycle.method]} : {...selection};
  const original = !full && stages.every(stage => choices[stage.id] === defaults[stage.id]);
  const compact = !original && !full;
  const matching = !full && build && stages.every(stage => build.selection?.[stage.id] === choices[stage.id]) &&
    (build.coordination || "single") === coordination;
  const plan = matching ? build.plan : null;
  const policy = full ? (cycle.config?.coordination || "none") : coordination;
  const adaptive = ["taskforce", "roles"].includes(policy);
  const names = Object.fromEntries(stages.map(stage => {
    const option = stage.options.find(item => item.id === choices[stage.id]);
    const name = option?.method || {none: "Skip", soft_atlas: "Soft atlas", continuous: "Cosine graph"}[option?.id] || option?.label.split(" (")[0] || "Unknown tool";
    return [stage.id, name];
  }));
  const bceDice = compact && plan?.extraction_loss === "bce_dice";
  const learnedAtlas = ["M09", "M10"].includes(choices.parcellation);
  const dense = ["M06", "M07"].includes(choices.registration);
  const auxiliary = choices.segmentation === "M08";
  const terms = [
    {id: "extraction", symbol: "brain", title: "Brain extraction", loss: bceDice ? "BCE + soft Dice loss" : "Binary cross entropy", short: bceDice ? "BCE + Dice" : "BCE", weight: 1, color: "#a6d8c7",
      detail: "Predicted soft brain mask ↔ each subject’s reference brain mask." + (bceDice ? " The Agent added soft Dice loss to BCE." : " BCE supervises brain versus nonbrain voxels.")},
    {id: "registration", symbol: "align", title: "Image registration", loss: "Global cross correlation", short: "GCC", weight: 0.1, color: "#a9c7ed",
      detail: "Warped brain image ↔ fixed MRI template. GCC is the image similarity loss." + (compact && dense ? " This adapter adds 0.01 × deformation smoothness inside this term." : "")},
    {id: "segmentation", symbol: "tissue", title: "Tissue segmentation", loss: "Cross entropy", short: "CE", weight: 1, color: "#d2b9e6",
      detail: "Predicted tissue logits ↔ " + (compact ? "each subject’s tissue reference mask." : "template tissue labels warped into subject space.") +
        (auxiliary ? (full ? " Deep supervision uses normalized multi-scale CE." : " Auxiliary heads add 0.5 × CE each.") : "") +
        (compact && learnedAtlas ? " The AAL term shares this task’s weight." : "") + " The tissue head has its own supervision; it does not feed the classifier."},
    {id: "classification", symbol: "class", title: "Diagnosis classification", loss: "Cross entropy", short: "CE", weight: 1, color: "#e2c08c",
      detail: "Subject diagnosis logits ↔ each subject’s class label. Classification loss trains the classifier and upstream paths that remain differentiable."}
  ];
  if (learnedAtlas) terms.splice(3, 0, {id: "parcellation", symbol: "AAL", title: "AAL parcellation", loss: "Cross entropy", short: "CE", weight: full ? 1 : 0.2, color: "#dfafb9",
    detail: "Predicted parcel logits ↔ each subject’s AAL reference labels." + (full ?
      " This additional full-model loss has a fixed weight of 1, outside the four TaskForce task weights." :
      " In this adapter, 0.2 × AAL CE is grouped inside the tissue task and shares its task weight.")});
  for (const term of terms) {
    term.method = names[term.id];
    term.adaptive = adaptive && (term.id !== "parcellation" || compact);
    term.weightSymbol = term.id === "parcellation" && compact ? "tissue" : term.symbol;
    term.adaptiveScale = term.id === "parcellation" && compact ? "0.2 " : "";
    term.extras = term.id === "segmentation" ? [auxiliary && "auxiliary CE"].filter(Boolean) :
      term.id === "registration" && compact && dense ? ["+ 0.01 smoothness"] : [];
  }
  let boundary;
  if (original) boundary = "Original atlas transfer and graph sparsification contain discrete operations. Joint optimization does not imply every loss reaches every module.";
  else if (full) boundary = "Full research training requests connected paths. Measured connectivity is reported in the run checks; this animation is illustrative.";
  else boundary = (choices.parcellation === "unibrain" || choices.graph === "unibrain" ? "A discrete atlas or graph boundary remains. " : "Differentiable atlas probabilities and continuous graphs permit gradient paths. ") + "Verify the selected pipeline with build checks.";
  const additions = stages.filter(stage => choices[stage.id] !== defaults[stage.id]);
  const description = additions.length ? additions.map(stage => stage.name + ": " + names[stage.id]).join("; ") + "." : "Original UniBrain: four supervised losses combined in one training objective.";
  const weighting = policy === "taskforce" ? "TaskForce actors adapt the four task weights during training." : policy === "roles" ?
    "Training roles can revise task weights after measured gradient checks." : "Default weights: brain 1, alignment 0.1, tissue 1, classification 1.";
  return {choices, names, terms, policy, adaptive, original, compact, full, boundary, weighting, description,
    profile: full ? "Full research run · " + cycle.method : original ? "Original full UniBrain" : "Custom adapter prototype",
    status: full ? "Run objective" : plan ? (build.status === "ready" ? "Prepared objective" : "Planned objective") : compact ? "Draft objective · Agent may revise loss" : "Default objective",
    objectiveKey: JSON.stringify(terms.map(term => [term.id, term.short, term.extras])),
    preprocessing: ["denoising", "bias", "normalization"].filter(id => choices[id] !== "none")};
}

function objectiveChange(previous, current) {
  if (!previous) return current.description + " " + current.weighting;
  const changed = Object.keys(current.choices).filter(id => current.choices[id] !== previous.choices[id]);
  const messages = [];
  if (changed.length) {
    messages.push("Updated: " + changed.map(id => current.names[id] === "Skip" ? id + " skipped" : current.names[id]).join(", ") + ".");
    messages.push(current.objectiveKey === previous.objectiveKey ? "The loss types are unchanged." : "The loss terms now match these modules.");
    if (previous.compact !== current.compact)
      messages.push(current.compact ? "The compact adapter uses subject tissue references." : "Tissue supervision uses warped template labels.");
  } else if (current.objectiveKey !== previous.objectiveKey) messages.push("The generated plan updated the loss definition below.");
  if (current.policy !== previous.policy) messages.push(current.weighting);
  if (changed.some(id => ["denoising", "bias", "normalization"].includes(id)) && current.preprocessing.length)
    messages.push("Preprocessing learns through downstream losses; no separate preprocessing loss is added.");
  return messages.join(" ");
}

function objectiveArt(kind) {
  const brain = '<path d="M29 5C17 2 8 12 9 23C2 33 10 47 21 48C27 55 38 53 42 46C54 46 60 34 54 24C58 12 46 3 37 6C34 3 31 3 29 5Z" fill="currentColor" fill-opacity=".12" stroke="currentColor" stroke-width="1.3"/><path d="M32 8V47M18 14L25 21L16 29L24 37L20 43M44 13L38 21L48 29L40 36L44 42" fill="none" stroke="currentColor" opacity=".5"/>';
  let drawing;
  if (kind === "registration") drawing = '<g transform="translate(9 0) rotate(-9 32 29)" opacity=".35">' + brain + '</g><g transform="translate(46 0)">' + brain + '</g><path class="objective-pulse" d="M17 20H100M15 32H103M20 44H100M31 8V49M47 6V52M63 8V50M79 9V49" fill="none" stroke="currentColor" stroke-width=".6"/>';
  else if (kind === "classification") drawing = '<g transform="translate(0 1) scale(.9)">' + brain + '</g><path d="M57 29H69M65 25L69 29L65 33" fill="none" stroke="currentColor"/><rect x="78" y="27" width="10" height="22" rx="2" fill="currentColor" opacity=".35"/><rect class="objective-pulse" x="94" y="11" width="10" height="38" rx="2" fill="currentColor"/><path d="M74 50H110" stroke="currentColor" opacity=".4"/>';
  else drawing = '<g transform="translate(25 0)">' + brain + (kind === "segmentation" || kind === "parcellation" ? '<path d="M16 20L25 13L30 23L23 33Z" fill="#a9c7ed" opacity=".7"/><path d="M36 14L47 21L40 34L35 28Z" fill="#d2b9e6" opacity=".8"/><path d="M25 34L33 29L40 40L31 47Z" fill="#e2c08c" opacity=".7"/>' : '<path d="M22 26L29 33L43 20" fill="none" stroke="currentColor" stroke-width="2.5"/>') + '</g><path class="objective-scan" d="M22 28H85" stroke="currentColor" stroke-width="1"/>';
  return '<svg class="objective-art" viewBox="0 0 115 58" aria-hidden="true">' + drawing + '</svg>';
}

let objectiveScope = "selection";
let objectiveSelected = "extraction";
let objectivePaused = typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
const objectivePrevious = {};
const objectiveSummaries = {};
let objectiveSignature = "";

function renderObjective() {
  if (!library || !draft) return;
  const cycle = typeof fullCycle !== "undefined" ? fullCycle : null;
  if (!cycle?.method) objectiveScope = "selection";
  const model = describeObjective({selection: draft, defaults: library.defaults, stages: library.stages,
    build: activeBuild, coordination: selectedCoordination(), cycle: objectiveScope === "research" ? cycle : null});
  const selector = byId("objectiveScope");
  selector.hidden = !cycle?.method;
  selector.value = objectiveScope;
  selector.onchange = () => {objectiveScope = selector.value; renderObjective();};
  const signature = JSON.stringify([model, objectiveScope, objectiveSelected]);
  if (signature !== objectiveSignature) {
    const message = objectiveChange(objectivePrevious[objectiveScope], model);
    if (message) objectiveSummaries[objectiveScope] = message;
    objectivePrevious[objectiveScope] = model;
    objectiveSignature = signature;
    byId("objectiveProfile").textContent = model.profile + " · " + model.status;
    const symbol = term => '<i>L</i><sub>' + term.symbol + '</sub>';
    byId("objectiveFormula").innerHTML = '<span><i>L</i><sub>joint</sub> =</span>' + model.terms.map((term, i) =>
      (i ? '<span class="objective-plus">+</span>' : "") + '<span class="objective-part" style="--loss-color:' + term.color + '"><span class="objective-weight">' +
      (term.adaptive ? term.adaptiveScale + '<i>w</i><sub>' + term.weightSymbol + '</sub>(t)' : term.weight) + '</span>' + symbol(term) + '</span>').join("");
    byId("objectiveFormula").setAttribute("aria-label", "Joint loss equals " + model.terms.map(term => (term.adaptive ? term.adaptiveScale + "adaptive " + term.weightSymbol + " weight" : term.weight) + " times " + term.title + " loss").join(" plus "));
    const selected = model.terms.find(term => term.id === objectiveSelected) || model.terms[0];
    objectiveSelected = selected.id;
    byId("objectiveTerms").classList.toggle("objective-terms-many", model.terms.length > 4);
    byId("objectiveTerms").innerHTML = model.terms.map(term => '<button class="objective-term" data-objective-term="' + term.id +
      '" aria-pressed="' + (term.id === selected.id) + '" aria-label="Inspect ' + term.title + ' loss" style="--loss-color:' + term.color + '">' +
      objectiveArt(term.id) + '<strong>' + term.title + '</strong><span class="objective-method">' + escapeHtml(term.method) +
      '</span><span class="objective-loss">' + term.short + (term.extras.length ? ' · ' + term.extras.join(" · ") : "") + '</span></button>').join("");
    byId("objectiveDetail").innerHTML = '<strong>' + selected.title + ' · ' + selected.loss + '.</strong> ' + escapeHtml(selected.detail);
    byId("objectiveBoundary").textContent = model.boundary;
    byId("objectiveChanges").textContent = objectiveSummaries[objectiveScope];
    byId("objectiveTerms").querySelectorAll("[data-objective-term]").forEach(node => node.onclick = () => {
      objectiveSelected = node.dataset.objectiveTerm;renderObjective();
      byId("objectiveTerms").querySelector('[data-objective-term="' + objectiveSelected + '"]').focus({preventScroll: true});
    });
  }
  byId("objectivePanel").classList.toggle("objective-paused", objectivePaused);
  byId("objectiveMotion").textContent = objectivePaused ? "▷ Play motion" : "Ⅱ Pause motion";
  byId("objectiveMotion").setAttribute("aria-pressed", String(objectivePaused));
  byId("objectiveMotion").setAttribute("aria-label", (objectivePaused ? "Play" : "Pause") + " illustrative gradient animation");
  byId("objectiveMotion").onclick = () => {objectivePaused = !objectivePaused;renderObjective();};
}
