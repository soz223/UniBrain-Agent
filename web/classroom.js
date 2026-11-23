'use strict';

const el = id => document.getElementById(id);
const state = { backend: '', connected: false, run: null, generation: 0, timer: null, errors: 0, eventCount: 0, outputKey: '', busy: false, recording: false };
const names = { 'llm.started': 'LLM request', 'llm.completed': 'LLM decision', 'tool.started': 'Tool call', 'tool.completed': 'Tool result', 'run.completed': 'Run complete', 'run.failed': 'Run failed' };

function backendUrl(value) {
  const url = new URL(value.trim());
  const local = ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname);
  if (url.username || url.password || url.search || url.hash || (url.protocol !== 'https:' && !(url.protocol === 'http:' && local))) {
    throw new Error('Use an HTTPS backend, or HTTP on localhost, with no credentials or query parameters.');
  }
  return url.href.replace(/\/+$/, '');
}

function notice(text, error = false) {
  el('notice').textContent = text;
  el('notice').classList.toggle('error', error);
}

function accessState() {
  el('run').disabled = !state.connected || state.busy || state.recording || !el('token').value.trim();
  el('run').firstChild.textContent = state.busy ? (state.recording ? 'Replaying ' : 'Running ') : 'Run instruction ';
  el('replay').disabled = state.busy;
}

async function request(path, body) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 20000);
  try {
    const options = { credentials: 'omit', redirect: 'error', cache: 'no-store', signal: controller.signal, headers: {} };
    if (body !== undefined) {
      options.method = 'POST';
      options.headers = { 'Content-Type': 'application/json', Authorization: 'Bearer ' + el('token').value.trim() };
      options.body = JSON.stringify(body);
    }
    const response = await fetch(state.backend + path, options);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `Demo server returned HTTP ${response.status}.`);
    return payload;
  } catch (error) {
    if (error.name === 'AbortError') throw new Error('The demo server took too long to respond. Reconnect to check the existing run.');
    if (error instanceof TypeError) throw new Error('Cannot reach the demo backend. Check its address and whether the server is online.');
    throw error;
  } finally { clearTimeout(timeout); }
}

function setConnection(ok, message) {
  state.connected = ok;
  el('connectionStatus').textContent = message;
  el('connectionStatus').className = 'status ' + (ok ? 'ready' : 'failed');
  accessState();
}

function showConnection(show) {
  el('connectionPanel').hidden = !show;
  el('connectionToggle').setAttribute('aria-expanded', String(show));
}

function stopPolling() {
  clearTimeout(state.timer);
  state.generation += 1;
}

async function connect() {
  stopPolling();
  const generation = state.generation;
  state.recording = false;
  try {
    state.backend = backendUrl(el('backend').value);
    el('connectionStatus').textContent = 'Connecting';
    const health = await request('/demo/health');
    if (generation !== state.generation) return;
    if (health.mode !== 'isolated_inference') throw new Error('This address is not an isolated classroom backend.');
    setConnection(true, 'Demo backend online');
    el('providerLabel').textContent = `${health.provider} · ${health.model || 'Model not reported'}`;
    const selected = new URLSearchParams(location.search).get('run') || health.active_run || health.latest_run;
    if (selected) {
      state.run = selected;
      state.eventCount = 0;
      state.outputKey = '';
      await poll(state.generation, true);
    } else {
      state.busy = Boolean(health.active_run);
      notice(el('token').value ? 'Ready. Your instruction will call the real LLM and isolated model.' : 'Viewer mode. Open Connection to enter instructor access and start a run.');
    }
    accessState();
    el('retry').hidden = true;
  } catch (error) {
    if (generation !== state.generation) return;
    state.busy = false;
    setConnection(false, 'Demo backend offline');
    notice(error.message, true);
    el('retry').hidden = false;
    document.querySelector('.flow').classList.remove('running');
    showConnection(true);
  }
}

function safeDetails(value) {
  if (typeof value === 'string' && value.startsWith('data:image/')) return '[Image shown in Model output]';
  if (Array.isArray(value)) return value.map(safeDetails);
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, safeDetails(item)]));
  return value;
}

function renderEvents(events) {
  const open = new Set(Array.from(el('events').querySelectorAll('details[open]')).map(node => node.dataset.sequence));
  const fragment = document.createDocumentFragment();
  for (const event of events) {
    const li = document.createElement('li');
    li.className = event.type.endsWith('failed') ? 'failed' : event.type.endsWith('started') ? 'active' : '';
    const title = document.createElement('div');
    title.className = 'event-title';
    const name = document.createElement('span');
    name.textContent = `${names[event.type] || event.type}${event.tool ? ': ' + event.tool : ''}`;
    const time = document.createElement('time');
    const date = new Date(event.time);
    time.textContent = Number.isNaN(date.getTime()) ? '' : date.toLocaleTimeString('en-US', { hour12: false });
    title.append(name, time);
    li.append(title);
    if (event.message || event.action) {
      const description = document.createElement('p');
      description.className = 'event-description';
      description.textContent = event.message || `Action: ${event.action}`;
      li.append(description);
    }
    const details = document.createElement('details');
    details.dataset.sequence = String(event.sequence);
    details.open = open.has(String(event.sequence));
    const summary = document.createElement('summary');
    summary.textContent = event.type === 'tool.started' ? 'View arguments' : event.type === 'tool.completed' ? 'View returned JSON' : 'View event';
    const pre = document.createElement('pre');
    pre.textContent = JSON.stringify(safeDetails(event.type === 'tool.started' ? event.arguments : event.type === 'tool.completed' ? event.result : event), null, 2);
    details.append(summary, pre);
    li.append(details);
    fragment.append(li);
  }
  el('events').replaceChildren(fragment);
}

function metric(list, name, value) {
  const row = document.createElement('div');
  const dt = document.createElement('dt');
  const dd = document.createElement('dd');
  dt.textContent = name;
  dd.textContent = String(value);
  row.append(dt, dd);
  list.append(row);
}

function renderOutput(result) {
  if (!result || typeof result !== 'object') return;
  const key = JSON.stringify([result.evidence, result.prediction, result.summary]);
  if (key === state.outputKey) return;
  state.outputKey = key;
  el('images').replaceChildren();
  const images = result.visualization || result.images || [];
  for (const item of Array.isArray(images) ? images : []) {
    const src = item.image || item.src;
    if (typeof src !== 'string' || !/^data:image\/(png|jpeg|webp);base64,[A-Za-z0-9+/=]+$/.test(src)) continue;
    const figure = document.createElement('figure');
    const image = document.createElement('img');
    image.src = src;
    image.alt = item.title || 'UniBrain sample output';
    const caption = document.createElement('figcaption');
    caption.textContent = item.title || 'Model output';
    figure.append(image, caption);
    el('images').append(figure);
  }
  const list = document.createElement('dl');
  list.className = 'metric-list';
  const prediction = result.prediction;
  if (prediction && typeof prediction === 'object') {
    metric(list, 'Predicted class', prediction.class_names?.[prediction.class_index] ?? prediction.class_index ?? 'Not reported');
    if (Array.isArray(prediction.probabilities)) prediction.probabilities.forEach((value, index) => {
      metric(list, `${prediction.class_names?.[index] || 'Class ' + index} probability`, Number.isFinite(value) ? (100 * value).toFixed(2) + '%' : String(value));
    });
  }
  if (result.evidence) {
    const evidence = result.evidence;
    if (evidence.inference_seconds !== undefined) metric(list, 'Inference time', `${Number(evidence.inference_seconds).toFixed(2)} s`);
    if (evidence.device) metric(list, 'Device', evidence.device);
    if (evidence.parameters) metric(list, 'Model parameters', Number(evidence.parameters).toLocaleString('en-US'));
    if (evidence.input_shape) metric(list, 'Input shape', evidence.input_shape.join(' × '));
  }
  if (Array.isArray(result.stages)) for (const stage of result.stages) {
    if (stage.seconds !== undefined) metric(list, stage.name || 'Stage', `${Number(stage.seconds).toFixed(2)} s`);
  }
  el('modelOutput').className = '';
  el('modelOutput').replaceChildren(list);
  if (!list.children.length) {
    const pre = document.createElement('pre');
    pre.className = 'meta';
    pre.textContent = JSON.stringify(safeDetails(result), null, 2);
    el('modelOutput').append(pre);
  }
}

function renderFlow(events, busy, failed) {
  const ids = ['nodeLlm', 'nodeHarness', 'nodeModel', 'nodeResult'];
  ids.forEach(id => el(id).className = 'flow-node');
  document.querySelector('.flow').classList.toggle('running', busy);
  for (const event of events) {
    if (event.type === 'llm.completed') el('nodeLlm').classList.add('done');
    if (event.type === 'tool.started') el('nodeHarness').classList.add('done');
    if (event.type === 'tool.completed') {
      el('nodeResult').classList.add('done');
      if (event.tool === 'run_inference') el('nodeModel').classList.add('done');
    }
  }
  const last = events.at(-1);
  if (!last) return;
  if (!busy) return;
  if (failed) return;
  if (last.type === 'llm.started' || last.type === 'llm.completed') el('nodeLlm').classList.add('active');
  else if (last.type === 'tool.started') {
    el('nodeHarness').classList.add('done');
    el(last.tool === 'run_inference' ? 'nodeModel' : 'nodeHarness').classList.add('active');
  } else if (last.type === 'tool.completed') el('nodeResult').classList.add('active');
}

async function poll(generation, loadPrompt = false) {
  if (generation !== state.generation || !state.run) return;
  try {
    const run = await request('/demo/runs/' + encodeURIComponent(state.run));
    if (generation !== state.generation) return;
    state.errors = 0;
    setConnection(true, 'Demo backend online');
    state.busy = ['queued', 'running'].includes(run.status);
    el('runStatus').textContent = state.busy ? 'Live execution' : run.status === 'complete' ? 'Completed run' : run.status === 'failed' ? 'Failed run' : run.status;
    el('runStatus').className = 'pill ' + (state.busy ? 'active' : run.status === 'failed' ? 'failed' : '');
    el('runMeta').textContent = `Run ${run.id} · ${run.provider} · ${run.model || 'Model not reported'}`;
    el('providerLabel').textContent = `${run.provider} · ${run.model || 'Model not reported'}`;
    if (loadPrompt && run.prompt) el('prompt').value = run.prompt;
    const events = Array.isArray(run.events) ? run.events : [];
    if (events.length !== state.eventCount || loadPrompt) { renderEvents(events); state.eventCount = events.length; }
    renderFlow(events, state.busy, run.status === 'failed');
    const result = run.result || [...events].reverse().find(event => event.type === 'tool.completed' && event.tool === 'run_inference')?.result;
    renderOutput(result);
    if (run.summary) { el('answerSection').hidden = false; el('answer').textContent = run.summary; }
    if (run.status === 'failed') notice(run.error || 'The run failed. Inspect the trace, then retry after correcting the issue.', true);
    else if (state.busy) notice(events.at(-1)?.type === 'tool.started' && events.at(-1)?.tool === 'run_inference' ? 'UniBrain is running on the isolated demo worker. The trace updates automatically.' : 'The Agent is working. Tool calls and returned observations appear below.');
    else notice('This is a completed real execution. Enter instructor access to run a new instruction.');
    el('share').hidden = false;
    el('retry').hidden = true;
    accessState();
    if (state.busy) state.timer = setTimeout(() => poll(generation), 1200);
  } catch (error) {
    if (generation !== state.generation) return;
    state.errors += 1;
    notice(error.message + (state.errors < 4 ? ' Retrying the status request.' : ' Reconnect to resume monitoring; the server may still be running.'), true);
    if (state.errors < 4) state.timer = setTimeout(() => poll(generation), 2500);
    else {
      setConnection(false, 'Connection interrupted');
      el('retry').hidden = false;
      document.querySelector('.flow').classList.remove('running');
    }
  }
}

el('connectionToggle').addEventListener('click', () => showConnection(el('connectionPanel').hidden));
el('connect').addEventListener('click', connect);
el('retry').addEventListener('click', connect);
el('token').addEventListener('input', accessState);
el('forget').addEventListener('click', () => { el('token').value = ''; accessState(); notice('Instructor access cleared. You can continue viewing this run.'); });
el('requestForm').addEventListener('submit', async event => {
  event.preventDefault();
  if (!state.connected || state.busy || state.recording || !el('token').value.trim()) return;
  const prompt = el('prompt').value.trim();
  if (!prompt) { notice('Enter an instruction first.', true); return; }
  state.busy = true;
  accessState();
  notice('Sending your instruction to the Agent.');
  try {
    const run = await request('/demo/runs', { prompt });
    stopPolling();
    state.recording = false;
    state.run = run.id;
    state.eventCount = 0;
    state.outputKey = '';
    state.errors = 0;
    el('events').replaceChildren();
    el('images').replaceChildren();
    el('modelOutput').textContent = 'Waiting for the model output.';
    el('modelOutput').className = 'output-empty';
    el('answerSection').hidden = true;
    const url = new URL(location.href);
    url.searchParams.delete('recording');
    url.searchParams.set('run', run.id);
    history.replaceState(null, '', url);
    showConnection(false);
    await poll(state.generation);
  } catch (error) {
    state.busy = false;
    accessState();
    notice(error.message, true);
    el('retry').hidden = false;
  }
});
el('share').addEventListener('click', async () => {
  if (!state.run) return;
  const url = new URL(location.href);
  if (state.recording) { url.searchParams.delete('run'); url.searchParams.set('recording', '1'); }
  else { url.searchParams.delete('recording'); url.searchParams.set('run', state.run); }
  try { await navigator.clipboard.writeText(url.href); notice('Run link copied. It contains no instructor credential.'); }
  catch { notice('Copy this run link from your browser address bar.'); history.replaceState(null, '', url); }
});

async function replay() {
  if (state.busy) return;
  stopPolling();
  const generation = state.generation;
  state.recording = true;
  state.busy = true;
  accessState();
  notice('Recorded execution. Loading a verified real run for accelerated playback.');
  try {
    const response = await fetch('classroom-recording.json', { cache: 'no-store' });
    if (!response.ok) throw new Error('The saved recording could not be loaded.');
    const run = await response.json();
    if (run.status !== 'complete' || !Array.isArray(run.events) || !run.result) throw new Error('The saved recording is incomplete.');
    if (generation !== state.generation) return;
    state.run = run.id;
    state.outputKey = '';
    el('prompt').value = run.prompt;
    el('providerLabel').textContent = `${run.provider} · ${run.model} · Recorded`;
    el('runMeta').textContent = `Recorded run ${run.id} · ${run.provider} · ${run.model}`;
    el('runStatus').textContent = 'Recorded execution';
    el('runStatus').className = 'pill active';
    el('images').replaceChildren();
    el('modelOutput').textContent = 'Recorded model output will appear at the corresponding tool result.';
    el('modelOutput').className = 'output-empty';
    el('answerSection').hidden = true;
    el('share').hidden = false;
    el('retry').hidden = true;
    showConnection(false);
    const url = new URL(location.href);
    url.searchParams.delete('run');
    url.searchParams.set('recording', '1');
    history.replaceState(null, '', url);
    let count = 0;
    function frame() {
      if (generation !== state.generation) return;
      count += 1;
      const events = run.events.slice(0, count);
      const complete = count === run.events.length;
      renderEvents(events);
      renderFlow(events, !complete, false);
      const result = events.find(event => event.type === 'tool.completed' && event.tool === 'run_inference')?.result;
      if (result) renderOutput(result);
      if (complete) {
        state.busy = false;
        renderOutput(run.result);
        el('answerSection').hidden = false;
        el('answer').textContent = run.summary || '';
        el('runStatus').className = 'pill';
        notice('Recorded execution complete. No LLM or model calls were made. Open Connection to return to live mode.');
        accessState();
      } else {
        notice('Recorded execution · accelerated playback. Original event timestamps are preserved. No live LLM or model calls.');
        const gap = new Date(run.events[count].time) - new Date(events.at(-1).time);
        state.timer = setTimeout(frame, Math.max(650, Math.min(3500, gap / 3)));
      }
    }
    frame();
  } catch (error) {
    state.busy = false;
    accessState();
    notice('Recorded execution unavailable. ' + error.message, true);
  }
}

el('replay').addEventListener('click', replay);

(async () => {
  try {
    const response = await fetch('classroom-config.json', { cache: 'no-store' });
    const config = await response.json();
    el('backend').value = config.backend_url || '';
    if (new URLSearchParams(location.search).get('recording') === '1') {
      el('connectionStatus').textContent = 'Recorded replay';
      await replay();
    } else if (config.backend_url) await connect();
    else { setConnection(false, 'Backend not configured'); notice('Open Connection and enter the isolated demo backend address.'); showConnection(true); }
  } catch { setConnection(false, 'Backend not configured'); notice('Open Connection and enter the isolated demo backend address.'); showConnection(true); }
})();
