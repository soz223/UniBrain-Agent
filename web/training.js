"use strict";

const trainingTasks = ['extraction','registration','segmentation','classification'];
const trainingColors = ['#a6d8c7','#9dbfe5','#dfc08d','#c7b0e8'];
const trainingPhases = {
  observe:'Each actor receives its own training loss and one row of the gradient Gram matrix. Shared parameters reveal how tasks interact.',
  act:'Four actor logits become policy shares through one softmax. The recorded profile determines the coefficients applied to the model losses.',
  backward:'The weighted task losses backpropagate into connected UniBrain modules. Arrows show computational paths, not gradient vectors in physical space.',
  learn:'On the next training sample, loss improvement and the previous mixed gradient produce a shared reward. Replay trains four actors and four centralized critics.'
};
const trainingState = {packet:null, latestPacket:null, current:null, selected:null, task:'classification', phase:'observe', follow:true, playing:false, timer:null, poll:null, source:'current', generation:0, error:null};

function trainingPolicyDescription(policy) {
  if(!policy)return {label:'TASKFORCE · COOPERATIVE ACTOR CRITIC',formula:'The backend records the applied coefficient rule.',
    scope:'Gradient measurements await a recorded update.',method:'Select a recorded update to inspect its coordinator profile.',action:'a'};
  if(policy.profile==='paper_core')return {label:'TASKFORCE CORE · FULL GRADIENTS',
    formula:policy.gradient_normalization==='none'?'w = softmax(a) · Applied λ = w.':'w = softmax(a) · Applied λⱼ = wⱼ / ‖∇Lⱼ‖₂. Zero norms use scale 1.',
    scope:'Every trainable parameter is included. '+(policy.gradient_normalization==='none'?'Raw gradient Gram matrix.':'L2 normalized gradient Gram matrix.'),
    method:'TaskForce core profile: local actors, centralized critics, raw action replay, shared loss and gradient reward, OU exploration, and policy learning after the model update. Normalization choices are recorded in the run.',action:'a'};
  return {label:'TASKFORCE INSPIRED · LEGACY PROFILE',formula:'w = softmax(a) · Applied λ = T × w × base coefficient.',
    scope:'Sampled, base-weighted parameter gradients.',method:'Legacy profile: sampled parameter gradients, UniBrain base coefficients, normalized action replay and policy learning before the next model update. This run was not upgraded to the core profile.',action:'w'};
}

function trainingNumber(value, digits=3) {
  if(typeof value !== 'number' || !Number.isFinite(value))return 'N/A';
  if(digits===0)return Math.round(value).toLocaleString('en-US');
  return value !== 0 && (Math.abs(value)<.001||Math.abs(value)>=10000) ? value.toExponential(1) : value.toFixed(digits);
}

function trainingFrames(packet) {
  const rows = new Map();
  for(const row of packet?.history||[])if(Number.isFinite(row.step))rows.set(row.step,row);
  return [...rows.values()].sort((a,b)=>a.step-b.step).slice(-128);
}

function trainingCosine(policy, i, j) {
  const a=policy?.gradient_norms?.[trainingTasks[i]],b=policy?.gradient_norms?.[trainingTasks[j]];
  if(!(a>0) || !(b>0))return null;
  const value=policy?.cosine_similarity?.[i]?.[j];
  return Number.isFinite(value)?Math.max(-1,Math.min(1,value)):null;
}

function trainingSelected(frames, step, follow) {
  return follow ? frames.at(-1) : frames.find(row=>row.step===step)||frames[0];
}

function trainingStageModule(stage) {
  return {extraction:['ext_net'],registration:['reg_net'],segmentation:['seg_net'],parcellation:['par_net'],graph:['feature_net','graph_gen'],classification:['gnn_pred'],denoising:['denoise_net'],bias:['bias_net'],normalization:['norm_net']}[stage]||[];
}

function trainingDiagram(packet, task) {
  const stages=packet?.stages||[];
  if(!stages.length)return '<p class="training-empty">Select a pipeline to inspect its training structure.</p>';
  const main=stages.filter(stage=>!stage.branch),branches=stages.filter(stage=>stage.branch);
  const width=Math.max(900,main.length*165+30),positions=new Map(),gap=(width-50)/Math.max(main.length,1);
  main.forEach((stage,i)=>positions.set(stage.id,{x:25+i*gap,y:40,w:Math.min(145,gap-15)}));
  branches.forEach((stage,i)=>positions.set(stage.id,{x:25+(i+1)*gap,y:133,w:Math.min(145,gap-15)}));
  if(packet.forward_edges)positions.set('input',{x:25,y:133,w:Math.min(145,gap-15)});
  const edges=packet.status==='observed'?(packet.connections||[]).filter(edge=>edge.loss===task):[];
  const reached=new Set(edges.map(edge=>edge.module));
  let svg='<svg viewBox="0 0 '+width+' 320" role="img" aria-label="UniBrain forward computation and selected loss backpropagation"><defs><marker id="trainingForwardArrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#88bbac"/></marker><marker id="trainingBackwardArrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#bca7e9"/></marker></defs><text x="25" y="21" class="flow-label">UNIBRAIN · FORWARD COMPUTATION</text>';
  const forwardEdges=packet.forward_edges||main.slice(1).map((stage,i)=>[main[i].id,stage.id]);
  for(const [source,target] of forwardEdges){
    const a=positions.get(source),b=positions.get(target);if(!a||!b)continue;
    let path;
    if(a.y===b.y)path='M'+(a.x+a.w)+' '+(a.y+25)+' H'+b.x;
    else if(a.x===b.x)path='M'+(a.x+a.w/2)+' '+a.y+' V'+(b.y+50);
    else path='M'+(a.x+a.w)+' '+(a.y+12)+' C'+(a.x+a.w+20)+' 110 '+(b.x+b.w/2)+' 115 '+(b.x+b.w/2)+' '+(b.y+50);
    svg+='<path class="forward-path" marker-end="url(#trainingForwardArrow)" d="'+path+'"/>';
  }
  if(packet.forward_edges){const p=positions.get('input');svg+='<rect class="stage-box" x="'+p.x+'" y="'+p.y+'" width="'+p.w+'" height="50" rx="22"/><text x="'+(p.x+13)+'" y="'+(p.y+23)+'">Input MRI</text><text class="stage-method" x="'+(p.x+13)+'" y="'+(p.y+39)+'">Template inputs omitted</text>';}
  const lossX=25+trainingTasks.indexOf(task)*(width-50)/4+75;
  for(const stage of stages){
    const pos=positions.get(stage.id),connected=trainingStageModule(stage.id).some(module=>reached.has(module));
    if(connected)svg+='<path class="backward-path" marker-end="url(#trainingBackwardArrow)" d="M'+lossX+' 248 C'+lossX+' 209 '+(pos.x+pos.w/2)+' 215 '+(pos.x+pos.w/2)+' '+(pos.y+52)+'"/>';
  }
  for(const stage of stages){
    const p=positions.get(stage.id),connected=trainingStageModule(stage.id).some(module=>reached.has(module));
    svg+='<g><title>'+escapeHtml(stage.name+' · '+stage.method)+'</title><rect class="stage-box '+(connected?'stage-selected':'')+'" x="'+p.x+'" y="'+p.y+'" width="'+p.w+'" height="50" rx="9"/><text x="'+(p.x+10)+'" y="'+(p.y+21)+'">'+escapeHtml(stage.name)+'</text><text class="stage-method" x="'+(p.x+10)+'" y="'+(p.y+37)+'">'+escapeHtml((stage.method||'').slice(0,30))+'</text></g>';
  }
  trainingTasks.forEach((name,i)=>{
    const x=25+i*(width-50)/4,term=(packet.terms||[]).find(item=>item.id===name);
    svg+='<g><rect class="loss-box '+(name===task?'loss-selected':'')+'" x="'+x+'" y="248" width="150" height="41" rx="8"/><text x="'+(x+11)+'" y="265" style="fill:'+trainingColors[i]+'">L'+(i+1)+' · '+escapeHtml(titleCase(name))+'</text><text class="stage-method" x="'+(x+11)+'" y="281">'+escapeHtml(term?.definition||'Not recorded')+'</text></g>';
  });
  svg+='<text x="25" y="309" class="flow-label">'+(packet.status==='observed'?'RECORDED AUTOGRAD PATHS · SELECT AN ACTOR BELOW':'CONFIGURED STRUCTURE · GRADIENT PATHS AWAIT A FORWARD PASS')+'</text></svg>';
  return svg;
}

function renderTrainingMatrix(policy) {
  const mode=byId('trainingMatrixMode').value,selected=trainingTasks.indexOf(trainingState.task);
  let html='<div class="training-matrix"><span></span>'+trainingTasks.map(task=>'<span class="matrix-label">'+task.slice(0,3).toUpperCase()+'</span>').join('');
  trainingTasks.forEach((task,i)=>{
    html+='<span class="matrix-label">'+task.slice(0,3).toUpperCase()+'</span>';
    trainingTasks.forEach((other,j)=>{
      const cosine=trainingCosine(policy,i,j),value=mode==='cosine'?cosine:policy?.gram?.[i]?.[j];
      const alpha=cosine===null?.04:.1+.65*Math.abs(cosine);
      const color=cosine<0?'210,143,134':'126,182,162';
      const label=mode==='gram'&&Number.isFinite(value)?value.toPrecision(3):trainingNumber(value,2);
      html+='<span class="'+(i===selected?'matrix-selected':'')+'" style="background:rgba('+color+','+alpha+')" title="'+escapeHtml(task+' / '+other)+'">'+label+'</span>';
    });
  });
  byId('trainingMatrix').innerHTML=html+'</div>';
  byId('trainingMatrixNote').textContent=trainingPolicyDescription(policy).scope+' Green: aligned. Coral: opposing. N/A: unavailable or zero norm. A near-zero cosine can reflect disjoint parameters.';
}

function renderTrainingWeights(frames, selected) {
  const mode=byId('trainingWeightMode').value,rows=frames.filter(row=>row.gradient_policy?.[mode]);
  if(!rows.length){byId('trainingWeights').innerHTML='<p class="training-empty">Weight trajectories appear after recorded policy decisions.</p>';return;}
  const min=rows[0].step,max=rows.at(-1).step,values=rows.flatMap(row=>trainingTasks.map(task=>row.gradient_policy[mode][task])).filter(Number.isFinite);
  const limit=mode==='actions'?1:Math.max(...values,1)*1.1,x=step=>36+(step-min)*390/Math.max(1,max-min),y=value=>151-value/limit*126;
  let svg='<svg viewBox="0 0 450 180" role="img" aria-label="Recorded task weight trajectories">';
  [0,.5,1].forEach(fraction=>{svg+='<path d="M36 '+y(limit*fraction)+' H426" stroke="#363a3b"/><text x="2" y="'+(y(limit*fraction)+3)+'">'+(limit*fraction).toFixed(1)+'</text>';});
  trainingTasks.forEach((task,i)=>{
    let path='';let joined=false;
    for(const row of rows){const value=row.gradient_policy[mode][task];if(!Number.isFinite(value)){joined=false;continue;}path+=(joined?' L':' M')+x(row.step)+' '+y(value);joined=true;}
    svg+='<path d="'+path+'" fill="none" stroke="'+trainingColors[i]+'" stroke-width="1.7"/>';
  });
  if(selected)svg+='<path d="M'+x(selected.step)+' 20 V155" stroke="#a0a8a6" stroke-dasharray="2 4"/>';
  svg+='<text x="36" y="173">'+min+'</text><text x="390" y="173">'+max+'</text></svg><div class="training-weight-legend">'+trainingTasks.map((task,i)=>'<span style="color:'+trainingColors[i]+'">'+titleCase(task)+'</span>').join('')+'</div>';
  byId('trainingWeights').innerHTML=svg;
}

function renderTrainingAgent(policy) {
  const task=trainingState.task,i=trainingTasks.indexOf(task),observation=policy?.observations?.[task];
  const gram=policy?.gram?.[i];
  byId('trainingSelectedTask').textContent=titleCase(task);
  byId('trainingAgentDetail').innerHTML='<div class="agent-inside"><div><h3>Local observation o'+(i+1)+'</h3><p>Gram row '+(i+1)+' + own loss. The actor receives a normalized Gram row and log(1 + loss).</p><div class="observation-vector">'+(observation||[null,null,null,null,null]).map(value=>'<span>'+trainingNumber(value)+'</span>').join('')+'</div><p>'+(observation?'Recorded actor input.':'Normalized input was not logged in this run. Raw Gram row is shown below.')+'</p><div class="observation-vector">'+(gram||[null,null,null,null]).map(value=>'<span>'+trainingNumber(value)+'</span>').join('')+'</div></div><div><h3>Actor '+(i+1)+' ↔ centralized critic '+(i+1)+'</h3><p>Actor output a: '+trainingNumber(policy?.action_logits?.[task])+'<br>Policy share w: '+trainingNumber(policy?.actions?.[task])+'<br>Applied coefficient λ: '+trainingNumber(policy?.weights?.[task])+'</p><p>Critic input: all four observations + joint action.<br>Critic TD loss: '+trainingNumber(policy?.critic_td_losses?.[task])+'<br>Policy updates: '+trainingNumber(policy?.policy_updates,0)+'</p></div></div><p class="disclosure-note">Each critic receives the same shared reward. Target networks are softly updated after replay learning. N/A indicates a field absent from this record.</p>';
}

function trainingCritics(policy) {
  let svg='<svg viewBox="0 0 900 170" role="img" aria-label="Joint replay feeds four centralized critics, each training its own actor"><defs><marker id="criticArrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 L10 5 L0 10Z" fill="#a995c1"/></marker></defs>';
  for(let i=0;i<4;i++){
    const x=112+i*225;
    svg+='<path class="critic-path" d="M'+x+' 0 V22 H450 V40"/><path class="critic-path" marker-end="url(#criticArrow)" d="M450 74 V92 H'+x+' V112"/>';
    svg+='<path class="critic-path critic-feedback" marker-end="url(#criticArrow)" d="M'+(x+68)+' 132 C'+(x+112)+' 132 '+(x+108)+' 25 '+(x+65)+' 5"/><rect x="'+(x-68)+'" y="112" width="136" height="42" rx="9" fill="#26232d" stroke="#645773"/><text class="critic-title" x="'+x+'" y="130" text-anchor="middle">Critic '+(i+1)+' · Q'+(i+1)+'</text><text x="'+x+'" y="145" text-anchor="middle">All observations + actions</text>';
  }
  svg+='<rect x="273" y="40" width="354" height="34" rx="9" fill="#272c2c" stroke="#50635b"/><text x="450" y="61" text-anchor="middle">Shared replay (O, '+trainingPolicyDescription(policy).action+', R, O′) · next training sample</text><text x="450" y="168" text-anchor="middle">Each critic trains its corresponding actor · dotted return = policy update</text></svg>';
  return svg;
}

function renderTraining() {
  if(!byId('trainingLab'))return;
  const packet=trainingState.packet,frames=trainingFrames(packet),row=trainingSelected(frames,trainingState.selected,trainingState.follow);
  trainingState.selected=row?.step??null;
  const policy=row?.gradient_policy,objective=row?.objective||(!frames.length?appConfig.objective:null);
  const description=trainingPolicyDescription(policy);
  byId('trainingProfile').textContent=description.label;
  byId('trainingWeightRule').textContent=description.formula;
  byId('trainingMethodScope').textContent=description.method;
  byId('trainingMode').textContent=policy?'Recorded policy decision':row?'Model update · no policy record':'Schematic · no recorded updates';
  byId('trainingGraph').innerHTML=trainingDiagram(objective,trainingState.task);
  byId('trainingActors').innerHTML=trainingTasks.map((task,i)=>{
    const share=policy?.actions?.[task],loss=row?.losses?.[task]??policy?.task_losses?.[task];
    return '<button class="training-actor" data-training-task="'+task+'" aria-pressed="'+(task===trainingState.task)+'" style="--task-color:'+trainingColors[i]+'"><span class="actor-label">ACTOR '+(i+1)+'</span><strong>'+titleCase(task)+'</strong><span class="actor-io">Gram row + L → a → w</span><span class="actor-values"><span>L '+trainingNumber(loss)+'</span><span>w '+trainingNumber(share)+'</span></span><span class="actor-bar"><i style="width:'+(Number.isFinite(share)?Math.max(0,Math.min(100,share*100)):0)+'%"></i></span></button>';
  }).join('');
  byId('trainingActors').querySelectorAll('button').forEach(button=>button.onclick=()=>{trainingState.task=button.dataset.trainingTask;renderTraining();});
  byId('trainingCritics').innerHTML=trainingCritics(policy);
  const reward=policy?.reward_terms;
  byId('trainingCooperation').innerHTML='<article><span class="source-tag">JOINT ACTION</span><strong>G = Σ λⱼ ∇Lⱼ</strong><small>4 actors → softmax → 1 model update<br>Training loss: '+trainingNumber(row?.total_loss??row?.weighted_loss)+'</small></article><article><span class="source-tag">SHARED REWARD · PREVIOUS TRANSITION</span><strong>R '+trainingNumber(policy?.reward)+'</strong><small>Loss improvement: '+trainingNumber(reward?.training_loss_improvement)+'<br>Gradient penalty: '+trainingNumber(reward?.gradient_norm_penalty)+'</small></article><article><span class="source-tag">REPLAY → 4 CENTRALIZED CRITICS</span><strong>'+trainingNumber(policy?.policy_updates,0)+' policy updates</strong><small>Replay: '+trainingNumber(policy?.replay_size,0)+' transitions<br>Actor / critic steps: '+trainingNumber(policy?.actor_optimizer_steps,0)+' / '+trainingNumber(policy?.critic_optimizer_steps,0)+'</small></article>';
  byId('trainingStep').disabled=!frames.length;byId('trainingStep').max=Math.max(0,frames.length-1);byId('trainingStep').value=Math.max(0,frames.indexOf(row));
  byId('trainingStepLabel').textContent=row?'#'+row.step+(Number.isFinite(row.epoch)?' · epoch '+row.epoch:'')+(trainingState.follow?' · latest':' · replay'):'No records';
  byId('trainingLatest').disabled=trainingState.follow;
  renderTrainingMatrix(policy);renderTrainingWeights(frames,row);renderTrainingAgent(policy);
  const validation=packet?.validation||[];
  byId('trainingValidation').innerHTML=validation.length?table(['Epoch','Brain Dice','Tissue Dice','Atlas Dice','AUC'],validation.slice(-10).map(item=>[item.epoch,...['brain_dice','tissue_dice','atlas_dice','auc'].map(key=>trainingNumber(item.validation?.[key]))])):'<p>No completed epoch evaluation in this source.</p>';
  updateTrainingPhase();updateTrainingFreshness();
}

function updateTrainingFreshness(message) {
  const packet=trainingState.latestPacket||trainingState.packet;
  if(trainingState.error){byId('trainingFreshness').textContent=trainingState.error;return;}
  if(message){byId('trainingFreshness').textContent=message;return;}
  const observed=Date.parse(packet?.observed_at),age=Number.isFinite(observed)?Math.max(0,Math.round((Date.now()-observed)/1000)):null;
  byId('trainingFreshness').textContent=age!==null?'Last file update '+(age<120?age+'s':Math.round(age/60)+'m')+' ago · reported '+(packet.reported_status||'unknown')+(age>120?' · may be validating or offline':''):trainingFrames(packet).length?'Workspace telemetry · recorded updates':'Connect a backend or run a training cycle to receive measurements.';
}

function updateTrainingPhase() {
  const visible=!byId('rolesView').hidden;
  byId('trainingLab').dataset.phase=trainingState.phase;
  byId('trainingLab').classList.toggle('is-playing',trainingState.playing&&visible);
  byId('trainingPhaseText').textContent=trainingPhases[trainingState.phase];
  document.querySelectorAll('[data-training-phase]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.trainingPhase===trainingState.phase)));
  byId('trainingAnimate').textContent=trainingState.playing?'Pause flow':'Play flow';
  byId('trainingAnimate').setAttribute('aria-pressed',String(trainingState.playing));
}

function acceptTrainingTelemetry(value) {
  const packet={...value,history:(value.history||[]).map(row=>({...row,objective:row.objective||(row.step===value.objective?.step?value.objective:null)}))};
  trainingState.current=packet;
  if(trainingState.source==='current'){trainingState.latestPacket=packet;if(trainingState.follow)trainingState.packet=packet;renderTraining();}
}

async function pollTrainingRun() {
  clearTimeout(trainingState.poll);
  if(trainingState.source==='current'||!backend.connected)return;
  const source=trainingState.source,generation=trainingState.generation;
  try {
    const packet=await requestJson('/api/training-runs/'+encodeURIComponent(source));
    if(generation!==trainingState.generation)return;
    trainingState.error=null;trainingState.latestPacket=packet;if(trainingState.follow)trainingState.packet=packet;renderTraining();
  } catch(error){if(generation===trainingState.generation){trainingState.error='Telemetry unavailable · '+error.message;updateTrainingFreshness();}}
  if(generation===trainingState.generation&&!byId('rolesView').hidden)trainingState.poll=setTimeout(pollTrainingRun,10000);
}

async function initializeTraining() {
  document.querySelectorAll('[data-training-phase]').forEach(button=>button.onclick=()=>{trainingState.phase=button.dataset.trainingPhase;trainingState.playing=false;updateTrainingPhase();});
  byId('trainingAnimate').onclick=()=>{trainingState.playing=!trainingState.playing;updateTrainingPhase();};
  byId('trainingStep').oninput=()=>{trainingState.follow=false;trainingState.selected=trainingFrames(trainingState.packet)[Number(byId('trainingStep').value)]?.step;renderTraining();};
  byId('trainingLatest').onclick=()=>{trainingState.follow=true;trainingState.packet=trainingState.latestPacket;renderTraining();};
  byId('trainingMatrixMode').onchange=renderTraining;byId('trainingWeightMode').onchange=renderTraining;
  byId('trainingRun').onchange=()=>{
    trainingState.source=byId('trainingRun').value;trainingState.generation++;trainingState.follow=true;trainingState.error=null;clearTimeout(trainingState.poll);
    trainingState.packet=trainingState.source==='current'?trainingState.current:null;trainingState.latestPacket=trainingState.packet;renderTraining();pollTrainingRun();
  };
  window.addEventListener('trainingtelemetry',event=>acceptTrainingTelemetry(event.detail));
  window.addEventListener('cycletrainingtelemetry',event=>acceptTrainingTelemetry({history:event.detail.updates||[]}));
  window.addEventListener('workspacechange',event=>{updateTrainingPhase();if(event.detail==='roles')pollTrainingRun();else clearTimeout(trainingState.poll);});
  const reduced=window.matchMedia('(prefers-reduced-motion: reduce)');
  trainingState.playing=!reduced.matches;
  reduced.addEventListener('change',event=>{if(event.matches){trainingState.playing=false;updateTrainingPhase();}});
  trainingState.timer=setInterval(()=>{
    if(byId('rolesView').hidden||document.hidden)return;
    if(trainingState.playing){const phases=Object.keys(trainingPhases);trainingState.phase=phases[(phases.indexOf(trainingState.phase)+1)%phases.length];updateTrainingPhase();}
    updateTrainingFreshness();
  },3500);
  renderTraining();
  if(typeof liveRun!=='undefined'&&liveRun)acceptTrainingTelemetry(liveRun);
  if(!await backendReady)return;
  try {
    const result=await requestJson('/api/training-runs');
    byId('trainingRun').innerHTML+='<optgroup label="Server training runs">'+(result.runs||[]).map(run=>'<option value="'+escapeHtml(run.id)+'">'+escapeHtml(run.label)+'</option>').join('')+'</optgroup>';
    if(result.runs?.length&&!trainingFrames(trainingState.current).length){byId('trainingRun').value=result.runs[0].id;byId('trainingRun').onchange();}
  } catch(error){updateTrainingFreshness('This backend does not expose external training runs. Workspace telemetry remains available.');}
}

if(typeof document!=='undefined'&&document.getElementById('trainingLab'))initializeTraining();
