"use strict";

let fullCycle = null;
let cycleTimer = null;
let cycleStarting = false;

function cyclePayload(mode) {
  const value = {task:byId('cycleTask').value.trim(), stage:byId('cycleStage').value,
    source:byId('cycleSource').value.trim(), data:byId('cycleData').value.trim(),
    gpu:Number(byId('cycleGpu').value), acceptance_steps:Number(byId('cycleAcceptance').value),
    train_steps:Number(byId('cycleTraining').value), coordination:byId('cycleCoordination').checked?'taskforce':'none', mode};
  for (const [key,id] of [['plans','cyclePlans'],['dataset','cycleDataset'],['fold','cycleFold']]) {
    if (byId(id).value.trim()) value[key]=key==='fold'?Number(byId(id).value):byId(id).value.trim();
  }
  return value;
}

function cycleBusy(busy) {
  byId('cycleForm').querySelectorAll('input,textarea,select,button').forEach(node=>node.disabled=busy);
  byId('cycleRun').innerHTML=busy?'<span class="spinner"></span> Running…':'Run full research cycle';
}

function cyclePolicyStatus(record) {
  if(record.config?.coordination!=='taskforce')return '';
  const count=Number(record.coordination?.policy_updates||0);
  return count>0?'Policy learning · '+count+' updates':'Policy warming up · no policy update yet';
}

function showCycle(record) {
  fullCycle=record;
  window.dispatchEvent(new CustomEvent('cycletrainingtelemetry',{detail:record}));
  renderObjective();
  byId('cycleProgress').hidden=false;
  const busy=record.status==='running';
  cycleBusy(busy);
  byId('cycleNotice').textContent=(busy?'◌ ':record.status==='complete'||record.status==='ready'?'✓ ':'○ ')+titleCase(record.status)+' · '+Math.round(record.elapsed_seconds||0)+'s';
  const stages=[['auditing','Inspect inputs'],['selecting','Select method'],['generating_and_validating','Generate & check'],['joint_training','Joint training']];
  const current=record.phase==='training_preflight'?'joint_training':record.phase;
  byId('cyclePhases').innerHTML=stages.map(([key,label])=>'<li class="'+(current===key&&busy?'active':'')+'">'+(current===key&&busy?'<span class="spinner"></span> ':'')+label+'</li>').join('');
  const updates=record.actual_updates||{};
  let content='<p><strong>'+escapeHtml(record.method?titleCase(record.method):'Selecting from the audited library')+'</strong> · '+escapeHtml(titleCase(record.phase||'starting'))+'</p>';
  content+='<p class="muted">Acceptance: '+Number(updates.acceptance||0)+' / '+record.config.acceptance_steps+' updates · Training: '+Number(updates.training||0)+' / '+record.config.train_steps+' updates · '+escapeHtml(record.provider?.provider||'')+'</p>';
  const policyState=cyclePolicyStatus(record);
  if(policyState)content+='<p>'+escapeHtml(policyState)+'</p>';
  if(record.selection?.rationale)content+='<p>'+escapeHtml(record.selection.rationale)+'</p>';
  if(record.error)content+='<p class="cycle-error">'+escapeHtml(record.error)+'</p>';
  if(record.status==='ready')content+='<p>Prerequisites checked. No AI request or training was executed.</p>';
  if(record.status==='complete')content+='<p>Accepted adapter trained in the full host. Code unchanged: '+(record.adapter_unchanged===true?'verified':'not confirmed')+'. This short check does not estimate accuracy.</p>';
  if(record.status==='interrupted')content+='<p class="cycle-error">Backend restarted. Inspect the retained run before retrying; this page has not resumed it.</p>';
  byId('cycleSummary').innerHTML=content;
  byId('cycleCode').disabled=!record.accepted_adapter;
  const points=(record.updates||[]).filter(row=>Number.isFinite(row.total_loss));
  if(points.length) {
    const values=points.map(row=>row.total_loss),min=Math.min(...values),span=Math.max(...values)-min||1;
    const xy=points.map((row,i)=>[15+i*570/Math.max(1,points.length-1),110-(row.total_loss-min)/span*90]);
    byId('cycleChart').innerHTML='<svg viewBox="0 0 600 140" role="img" aria-label="Observed training loss"><polyline points="'+xy.map(p=>p.join(',')).join(' ')+'" fill="none" stroke="var(--accent)" stroke-width="2"/>'+xy.map(([x,y])=>'<circle cx="'+x+'" cy="'+y+'" r="3" fill="var(--accent)"/>').join('')+'<text x="15" y="135" fill="var(--muted)">Latest total loss: '+values.at(-1).toPrecision(5)+'</text></svg>';
  } else byId('cycleChart').innerHTML='';
  byId('cycleEvidence').innerHTML='<p>'+escapeHtml(record.scope||'')+'</p><pre>'+escapeHtml(JSON.stringify({checks:record.checks,proposals:record.proposal_count,calls:record.calls,selection_usage:record.selection_usage,coordination:record.coordination,gradients:record.gradients,blockers:record.blockers},null,2))+'</pre>';
}

async function pollCycle(identifier) {
  clearTimeout(cycleTimer);
  try {
    const record=await requestJson('/api/research-cycles/'+identifier);
    showCycle(record);
    if(record.status==='running')cycleTimer=setTimeout(()=>pollCycle(identifier),2500);
  } catch(error) {
    byId('cycleNotice').textContent='Progress unavailable: '+error.message;
    cycleTimer=setTimeout(()=>pollCycle(identifier),5000);
  }
}

async function submitCycle(mode) {
  if(cycleStarting||fullCycle?.status==='running')return;
  if(!backend.connected){openWorkspace('settings');return;}
  if(!aiSession){openWorkspace('settings');aiNotice('Connect your AI account before starting a full research cycle.');return;}
  if(!byId('cycleForm').reportValidity())return;
  const payload=cyclePayload(mode);
  cycleStarting=true;cycleBusy(true);
  byId('cycleNotice').textContent=mode==='preflight'?'Checking prerequisites, please wait…':'Starting the full model cycle, please wait…';
  try {
    const result=await requestJson('/api/research-cycles',payload);
    await pollCycle(result.id);
  } catch(error) {cycleBusy(false);byId('cycleNotice').textContent=error.message;}
  finally {cycleStarting=false;}
}

async function initializeCycle() {
  byId('cycleCoordination').onchange=()=>{byId('cyclePolicyHint').hidden=!byId('cycleCoordination').checked;};
  window.addEventListener('datasetchanged',async()=>{
    if(cycleStarting||fullCycle?.status==='running')return;
    try {
      const state=await requestJson('/api/research-cycles');
      byId('cycleData').value=state.defaults.data;
      byId('cycleNotice').textContent='○ Attached dataset updated';
    } catch(error){byId('cycleNotice').textContent=error.message;}
  });
  byId('cycleForm').onsubmit=event=>{event.preventDefault();submitCycle('run');};
  byId('cycleCheck').onclick=()=>submitCycle('preflight');
  byId('cycleCode').onclick=async()=>{
    try {const code=await requestJson('/api/research-cycles/'+fullCycle.id+'/code');showDetails('Accepted adapter','<p>SHA256 '+escapeHtml(code.sha256)+'</p><pre>'+escapeHtml(code.code)+'</pre>','FULL MODEL');}
    catch(error){byId('cycleNotice').textContent=error.message;}
  };
  byId('cycleExport').onclick=()=>{
    const url=URL.createObjectURL(new Blob([JSON.stringify(fullCycle,null,2)],{type:'application/json'}));
    const link=document.createElement('a');link.href=url;link.download='research-cycle-'+fullCycle.id+'.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  };
  if(!(await backendReady))return;
  try {
    const state=await requestJson('/api/research-cycles');
    byId('cycleSource').value=state.defaults.source;byId('cycleData').value=state.defaults.data;byId('cycleGpu').value=state.defaults.gpu;
    byId('cycleStage').innerHTML='<option value="all">Agent chooses the stage</option>'+state.defaults.stages.map(stage=>'<option value="'+stage+'">'+escapeHtml(titleCase(stage))+'</option>').join('');
    byId('cycleStage').value='classification';
    byId('cycleNotice').textContent='○ Ready · full architecture';
    if(state.job){
      if(state.job.status==='running') {
        byId('cycleCoordination').checked=state.job.config.coordination==='taskforce';
        byId('cyclePolicyHint').hidden=!byId('cycleCoordination').checked;
      }
      showCycle(state.job);if(state.job.status==='running')pollCycle(state.job.id);
    }
  } catch(error){byId('cycleNotice').textContent=error.message;}
}

initializeCycle();
