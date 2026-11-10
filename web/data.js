let dataConfig=null,dataJob=null,dataTimer=null,dataBusy=false;
const manifestExample={version:1,intensity:'preserve',labels:{diagnosis:{'0':'Define control','1':'Define case'},tissue:{'0':'Background','1':'Define tissue class 1','2':'Define tissue class 2','3':'Define tissue class 3'},atlas:'AAL116'},templates:{image:'templates/t1.nii.gz',tissue:'templates/tissue.nii.gz',atlas:'templates/aal116.nii.gz'},subjects:[{id:'subject001',modality:'T1w',split:'train',label:0,supervision:'manual',image:'sub001/T1w.nii.gz',brain:'sub001/brain.nii.gz',tissue:'sub001/tissue.nii.gz',atlas:'sub001/aal116.nii.gz'}]};

function dataControls(busy) {
  dataBusy=busy;['inspectData','buildData','askData'].forEach(id=>byId(id).disabled=busy||!backend.connected);
  document.querySelectorAll('[data-attach-dataset]').forEach(node=>node.disabled=busy);
}

function renderDataJob(job) {
  dataJob=job;dataControls(job.status==='running');
  const progress=byId('dataProgress'),failed=job.status==='failed';
  progress.hidden=job.status==='complete';progress.classList.toggle('assistant-error',failed);
  progress.innerHTML=(dataBusy?'<span class="spinner"></span>':'<span>!</span>')+'<span>'+escapeHtml(job.error||job.events?.at(-1)?.message||'Starting data checks…')+'</span>';
  byId('dataBadge').textContent=dataBusy?'◉ Working':failed?'! Needs attention':'✓ Checked';
  if(job.status!=='complete')return;
  let html='';
  if(job.mode==='discuss') {
    const result=job.result;
    html='<article class="data-answer"><p>'+escapeHtml(result.answer)+'</p><details class="disclosure"><summary>Preparation plan & model discussion</summary><h3>Preparation plan</h3><ol>'+result.plan.map(item=>'<li>'+escapeHtml(item)+'</li>').join('')+'</ol>'+(result.blockers.length?'<h3>Inputs to resolve</h3><p>'+escapeHtml(result.blockers.join('\n'))+'</p>':'')+'<h3>Model discussion</h3><p>'+escapeHtml(result.model_changes)+'</p></details><p class="settings-note">'+escapeHtml((job.calls||[]).map(call=>call.backend+' · '+call.seconds.toFixed(1)+'s').join(' · '))+'</p></article>';
  }
  const items=job.mode==='build'?[{path:job.result.dataset,kind:'Converted dataset',status:'ready',audit:job.result.audit}]:job.inventory?.items||[];
  if(items.length)html+='<details class="disclosure"'+(job.mode!=='discuss'?' open':'')+'><summary>Inspected sources <span>'+items.length+' entries</span></summary>';
  html+=items.slice(0,30).map((item,index)=>'<div class="data-result-row"><div><strong>'+escapeHtml((item.status==='ready'?'✓ ':item.status==='blocked'?'! ':'○ ')+item.kind)+'</strong><small>'+escapeHtml(item.path)+'</small><small>'+escapeHtml(item.reason||(item.audit?Object.entries(item.audit.splits).map(([split,value])=>split+': '+value.count).join(' · '):''))+(item.audit?.warnings?.length?' · '+escapeHtml(item.audit.warnings.join(' ')):'')+'</small></div>'+(item.status==='ready'?'<button class="text-button" data-attach-dataset="'+index+'">Use dataset ↗</button>':'')+'</div>').join('');
  if(items.length)html+='</details>';
  if(!items.length&&job.mode==='inspect')html='<p class="settings-note">No supported volumes found. Select NIfTI folders, DICOM series or a UniBrain NPY dataset.</p>';
  html+='<details class="disclosure"><summary>Data notebook <span>'+job.events.length+' events</span></summary><pre class="prompt-content">'+escapeHtml(job.events.map(event=>event.message).join('\n'))+'</pre></details>';
  byId('dataResult').innerHTML=html;
  document.querySelectorAll('[data-attach-dataset]').forEach(button=>button.onclick=async()=>{
    dataControls(true);
    try {const selected=await requestJson('/api/data/use',{id:job.id,dataset:items[Number(button.dataset.attachDataset)].path});dataConfig.selected=selected;byId('selectedDataset').textContent='✓ Attached dataset · '+selected.path;window.dispatchEvent(new Event('datasetchanged'));}
    catch(error){progress.hidden=false;progress.textContent=error.message;}
    finally{dataControls(false);}
  });
}

async function pollData(identifier) {
  try {const job=await requestJson('/api/data/jobs/'+identifier);renderDataJob(job);if(job.status==='running')dataTimer=setTimeout(()=>pollData(identifier),1200);}
  catch(error){renderDataJob({status:'failed',error:error.message});}
}

async function submitData(mode) {
  if(!backend.connected){openWorkspace('settings');return;}
  if(mode==='discuss'&&!aiSession){openWorkspace('settings');aiNotice('Connect your own AI account before starting the data agent.');return;}
  if(mode==='build'&&!byId('dataManifest').value.trim()){byId('dataSetup').open=true;byId('dataManifest').focus();return;}
  clearTimeout(dataTimer);renderDataJob({status:'running',events:[]});
  try {
    const value=await requestJson('/api/data/jobs',{mode,sources:byId('dataSources').value.split('\n').map(value=>value.trim()).filter(Boolean),manifest:byId('dataManifest').value.trim(),message:byId('dataQuestion').value,selection:draft||appConfig.catalog.defaults});
    await pollData(value.id);
  } catch(error){renderDataJob({status:'failed',error:error.message});}
}

async function initializeData() {
  byId('inspectData').onclick=()=>submitData('inspect');byId('buildData').onclick=()=>submitData('build');
  byId('dataChatForm').onsubmit=event=>{event.preventDefault();submitData('discuss');};
  byId('dataExample').onclick=()=>showDetails('Conversion manifest','<p>Paths are relative to the manifest on your backend. Add eligible subjects for train, val and test. Keep each subject in one split. Confirm class meanings and annotation provenance.</p><pre class="prompt-content">'+escapeHtml(JSON.stringify(manifestExample,null,2))+'</pre><a href="https://github.com/soz223/UniBrain-Agent/blob/main/docs/data-agent.md" target="_blank" rel="noopener">Data preparation guide ↗</a>','DATA BUILD AGENT');
  byId('viewDataContract').onclick=()=>showDetails('Model input contract','<pre class="prompt-content">'+escapeHtml(JSON.stringify({...appConfig.data_contract,selection:draft||appConfig.catalog.defaults},null,2))+'</pre>','SELECTED PIPELINE');
  if(!(await backendReady)){dataControls(false);return;}
  try {dataConfig=await requestJson('/api/data');if(dataConfig.selected)byId('selectedDataset').textContent='✓ Attached dataset · '+dataConfig.selected.path;if(dataConfig.job){byId('dataSources').value=(dataConfig.job.sources||[]).join('\n');byId('dataManifest').value=dataConfig.job.manifest||'';renderDataJob(dataConfig.job);if(dataBusy)pollData(dataConfig.job.id);}}
  catch(error){byId('selectedDataset').textContent='! '+error.message;}
}
initializeData();
