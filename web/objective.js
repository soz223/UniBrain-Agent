"use strict";

function objectiveArt(kind) {
  const brain = '<path d="M29 5C17 2 8 12 9 23C2 33 10 47 21 48C27 55 38 53 42 46C54 46 60 34 54 24C58 12 46 3 37 6C34 3 31 3 29 5Z" fill="currentColor" fill-opacity=".12" stroke="currentColor" stroke-width="1.3"/><path d="M32 8V47M18 14L25 21L16 29L24 37L20 43M44 13L38 21L48 29L40 36L44 42" fill="none" stroke="currentColor" opacity=".5"/>';
  let drawing;
  if (kind === "registration" || kind === "regularization") drawing = '<g transform="translate(9 0) rotate(-9 32 29)" opacity=".35">' + brain + '</g><g transform="translate(46 0)">' + brain + '</g><path class="objective-pulse" d="M17 20H100M15 32H103M20 44H100M31 8V49M47 6V52M63 8V50M79 9V49" fill="none" stroke="currentColor" stroke-width=".6"/>';
  else if (kind === "classification") drawing = '<g transform="translate(0 1) scale(.9)">' + brain + '</g><path d="M57 29H69M65 25L69 29L65 33" fill="none" stroke="currentColor"/><rect x="78" y="27" width="10" height="22" rx="2" fill="currentColor" opacity=".35"/><rect class="objective-pulse" x="94" y="11" width="10" height="38" rx="2" fill="currentColor"/><path d="M74 50H110" stroke="currentColor" opacity=".4"/>';
  else drawing = '<g transform="translate(25 0)">' + brain + (kind === "segmentation" || kind === "parcellation" ? '<path d="M16 20L25 13L30 23L23 33Z" fill="#a9c7ed" opacity=".7"/><path d="M36 14L47 21L40 34L35 28Z" fill="#d2b9e6" opacity=".8"/><path d="M25 34L33 29L40 40L31 47Z" fill="#e2c08c" opacity=".7"/>' : '<path d="M22 26L29 33L43 20" fill="none" stroke="currentColor" stroke-width="2.5"/>') + '</g><path class="objective-scan" d="M22 28H85" stroke="currentColor" stroke-width="1"/>';
  return '<svg class="objective-art" viewBox="0 0 115 58" aria-hidden="true">' + drawing + '</svg>';
}

let objectiveScope = "selection";
let objectiveSelected = "";
let objectivePaused = typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
let objectiveRequest = 0;
let objectiveCacheKey = "";
let objectivePacket = null;
let objectivePreviousPacket = null;
let objectiveObservedCycle = null;
const objectiveColors = ["#a6d8c7", "#a9c7ed", "#d2b9e6", "#e2c08c", "#dfafb9"];

function objectiveFormula(packet) {
  return packet.terms.map(term => ({name: term.name, symbol: term.id, coefficient: term.weight,
    expression: packet.status === "declared" && ["taskforce", "roles"].includes(packet.coordination) ?
      (term.relative_weight || 1) + " × w(" + (term.weight_group || term.id) + ", t)" : String(term.weight)}));
}

function paintObjective(packet) {
  const previous = objectivePreviousPacket;
  objectivePacket = packet;
  if (packet) objectivePreviousPacket = packet;
  byId("objectiveProfile").textContent = packet ? packet.profile + " · " + (packet.status === "observed" ? "Observed update " + packet.step : "Backend definition preview") : "Connect the updated backend to resolve this selection";
  byId("objectiveFormula").innerHTML = packet ? '<span><i>L</i><sub>joint</sub> =</span>' + objectiveFormula(packet).map((term, i) =>
    (i ? '<span class="objective-plus">+</span>' : '') + '<span class="objective-part" style="--loss-color:' + objectiveColors[i % objectiveColors.length] + '"><span class="objective-weight">' + escapeHtml(term.expression) + '</span><i>L</i><sub>' + escapeHtml(term.symbol) + '</sub></span>').join('') : '';
  const terms = packet?.terms || [];
  const selected = terms.find(term => term.id === objectiveSelected) || terms[0];
  objectiveSelected = selected?.id || '';
  byId("objectiveTerms").classList.toggle("objective-terms-many", terms.length > 4);
  byId("objectiveTerms").innerHTML = terms.map((term, i) => '<button class="objective-term" data-loss-index="' + i + '" aria-pressed="' + (term.id === objectiveSelected) + '" style="--loss-color:' + objectiveColors[i % objectiveColors.length] + '">' + objectiveArt(term.id) + '<strong>' + escapeHtml(term.name) + '</strong><span class="objective-loss">' + escapeHtml(term.definition) + '</span></button>').join('');
  byId("objectiveDetail").textContent = selected ? selected.name + ': ' + selected.definition + '. Reference: ' + selected.supervision + (Number.isFinite(selected.value) ? '. Observed loss: ' + selected.value.toPrecision(5) + '; weighted: ' + selected.weighted_value.toPrecision(5) : '') : '';
  byId("objectiveBoundary").textContent = packet?.connection_evidence || 'No loss definition is available for this configuration. No stale formula is shown.';
  const changed = previous && packet ? packet.terms.filter(term => JSON.stringify(term) !== JSON.stringify(previous.terms.find(item => item.id === term.id))).map(term => term.name) : [];
  const modulesChanged = previous && packet ? packet.modules.filter(module => !previous.modules.some(item => item.id === module.id && item.name === module.name)).map(module => module.name) : [];
  byId("objectiveChanges").textContent = packet?.status === 'observed' ? 'Executed total: ' + packet.total.toPrecision(6) + '. Reconstructed from displayed terms: ' + packet.reconstructed_total.toPrecision(6) + '.' :
    (changed.length || modulesChanged.length ? 'Updated: ' + [...changed, ...modulesChanged].join(', ') + '. ' : '') + (packet?.explanation || '');
  byId("objectiveConnections").innerHTML = (packet?.modules || []).map(module => {
    const losses = (packet.connections || []).filter(edge => edge.module === module.id && edge.status !== 'zero_weight').map(edge => packet.terms.find(term => term.id === edge.loss)?.name || edge.loss);
    const gradient = packet.gradient_verification?.classification_to_candidates?.[module.id];
    return '<div><strong>' + escapeHtml(module.name) + '</strong><span>' + escapeHtml(packet.status === 'observed' ? (losses.length ? losses.join(' + ') + ' → autograd path' : 'No exposed autograd path from reported losses') : 'Connection awaiting execution') + (gradient ? '<small>' + escapeHtml(gradient.finite && gradient.nonzero ? 'Diagnosis gradient: finite and nonzero' : 'Diagnosis gradient: not confirmed active') + '</small>' : '') + '</span></div>';
  }).join('');
  byId("objectiveTerms").querySelectorAll('[data-loss-index]').forEach(node => node.onclick = () => {
    objectiveSelected = terms[Number(node.dataset.lossIndex)].id; paintObjective(packet);
  });
  byId("objectivePanel").classList.toggle('objective-paused', objectivePaused || !packet);
  byId("objectiveMotion").textContent = objectivePaused ? '▷ Play motion' : 'Ⅱ Pause motion';
  byId("objectiveMotion").setAttribute('aria-pressed', String(objectivePaused));
  byId("objectiveMotion").onclick = () => {objectivePaused = !objectivePaused; paintObjective(objectivePacket);};
}

async function renderObjective() {
  if (!library || !draft) return;
  const cycle = typeof fullCycle !== 'undefined' ? fullCycle : null;
  if (!cycle?.method) objectiveScope = 'selection';
  const selector = byId('objectiveScope');
  selector.hidden = !cycle?.method; selector.value = objectiveScope;
  selector.onchange = () => {objectiveScope = selector.value; objectiveCacheKey = ''; renderObjective();};
  const observed = typeof liveRun !== 'undefined' ? liveRun?.objective || liveRun?.history?.at(-1)?.objective : null;
  const key = JSON.stringify([draft, selectedCoordination(), activeBuild?.id, activeBuild?.status, objectiveScope, cycle?.objective, observed]);
  if (key === objectiveCacheKey) return;
  objectiveCacheKey = key;
  const request = ++objectiveRequest;
  if (objectiveScope === 'research') {paintObjective(cycle?.objective || null); return;}
  if (observed?.selection && library.stages.every(stage => observed.selection[stage.id] === draft[stage.id]) && observed.coordination === selectedCoordination()) {
    paintObjective(observed); return;
  }
  if (!backend.connected) {
    const baseline = appConfig.objective;
    paintObjective(baseline && JSON.stringify(baseline.selection) === JSON.stringify(draft) && selectedCoordination() === baseline.coordination ? baseline : null);
    return;
  }
  paintObjective(null);
  byId('objectiveProfile').textContent = 'Loading objective from backend…';
  try {
    const packet = await requestJson('/api/objective/preview', {selection: {...draft}, coordination: selectedCoordination(), build_id: activeBuild?.id});
    if (request === objectiveRequest) paintObjective(packet);
  } catch (error) {
    if (request === objectiveRequest) {objectiveCacheKey = ''; paintObjective(null); byId('objectiveChanges').textContent = 'Objective unavailable: ' + error.message;}
  }
}
