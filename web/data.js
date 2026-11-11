let dataConfig=null,dataJob=null,dataTimer=null,dataBusy=false;
const manifestExample={version:1,intensity:'preserve',labels:{diagnosis:{'0':'Define control','1':'Define case'},tissue:{'0':'Background','1':'Define tissue class 1','2':'Define tissue class 2','3':'Define tissue class 3'},atlas:'AAL116'},templates:{image:'templates/t1.nii.gz',tissue:'templates/tissue.nii.gz',atlas:'templates/aal116.nii.gz'},subjects:[{id:'subject001',modality:'T1w',split:'train',label:0,supervision:'manual',image:'sub001/T1w.nii.gz',brain:'sub001/brain.nii.gz',tissue:'sub001/tissue.nii.gz',atlas:'sub001/aal116.nii.gz'}]};

function dataControls(busy) {
  dataBusy=busy;['inspectData','buildData','askData'].forEach(id=>byId(id).disabled=busy||!backend.connected);
  document.querySelectorAll('[data-attach-dataset]').forEach(node=>node.disabled=busy);
}

function selectedDatasetLabel(selected) {
  if (!selected) return '○ Using configured sample data';
  const name=selected.path.split(/[\\/]/).filter(Boolean).at(-1);
  const counts=Object.entries(selected.audit?.splits||{}).map(([split,value])=>split+' '+value.count).join(' · ');
  return '✓ Attached '+name+(counts?' · '+counts:'');
}

function showSelectedDataset(selected) {
  byId('selectedDataset').textContent=selectedDatasetLabel(selected);
  const warning=selected?.audit?.warnings?.[0];
  byId('dataWarning').hidden=!warning;
  byId('dataWarning').textContent=warning?'! '+warning:'';
  byId('dataBadge').textContent=selected?'✓ Attached':'○ Sample data';
}

function renderDataJob(job) {
  dataJob=job;dataControls(job.status==='running');
  const progress=byId('dataProgress'),failed=job.status==='failed';
  progress.hidden=job.status==='complete';progress.classList.toggle('assistant-error',failed);
  progress.innerHTML=(dataBusy?'<span class="spinner"></span>':'<span>!</span>')+'<span>'+escapeHtml(job.error||job.events?.at(-1)?.message||'Starting data checks…')+'</span>';
  byId('dataBadge').textContent=dataBusy?'◉ Working':failed?'! Needs attention':dataConfig?.selected?'✓ Attached':'○ Sample data';
  if(job.status!=='complete'){byId('dataResult').innerHTML='';return;}
  let html='';
  if(job.mode==='discuss') {
    const result=job.result;
    const answer=result.answer||'';
    html='<details class="disclosure data-answer"><summary>Latest data agent answer</summary><p>'+escapeHtml(answer)+'</p>'+(result.blockers?.length?'<p class="data-blocker">Needs input: '+escapeHtml(result.blockers.slice(0,2).join(' · '))+'</p>':'')+'</details>';
  }
  const items=job.mode==='build'?[{path:job.result.dataset,kind:'Converted dataset',status:'ready',audit:job.result.audit}]:job.inventory?.items||[];
  const ordered=[...items].sort((a,b)=>(b.status==='ready')-(a.status==='ready'));
  if(job.mode!=='discuss') {
    const ready=items.filter(item=>item.status==='ready').length;
    html='<p class="data-summary">'+ready+' ready · '+(items.length-ready)+' need attention</p>';
    const row=(item,index)=>'<div class="data-result-row"><div><strong>'+escapeHtml((item.status==='ready'?'✓ ':'! ')+item.kind)+'</strong><small>'+escapeHtml(item.path.split(/[\\/]/).filter(Boolean).at(-1)||item.path)+'</small><small>'+escapeHtml(item.reason||(item.audit?Object.entries(item.audit.splits).map(([split,value])=>split+' '+value.count).join(' · '):''))+'</small></div>'+(item.status==='ready'?'<button class="text-button" data-attach-dataset="'+index+'">Use dataset ↗</button>':'')+'</div>';
    html+=ordered.slice(0,2).map(row).join('');
    if(ordered.length>2)html+='<details class="disclosure"><summary>More datasets <span>'+(ordered.length-2)+'</span></summary>'+ordered.slice(2).map((item,index)=>row(item,index+2)).join('')+'</details>';
    if(!items.length)html='<p class="settings-note">No supported data found in this source.</p>';
  }
  byId('dataResult').innerHTML=html;
  document.querySelectorAll('[data-attach-dataset]').forEach(button=>button.onclick=async()=>{
    dataControls(true);
    try {const selected=await requestJson('/api/data/use',{id:job.id,dataset:ordered[Number(button.dataset.attachDataset)].path});dataConfig.selected=selected;showSelectedDataset(selected);window.dispatchEvent(new Event('datasetchanged'));}
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
  byId('viewDataContract').onclick=()=>showDetails('Model input requirements',table(['Input','Required format'],[['Image','3D T1w · 96 × 96 × 96'],['Targets','Brain mask · tissue labels 0 to 3 · AAL116 atlas · binary diagnosis'],['Splits','Train · validation · test']])+'<p>Each subject needs aligned annotations and a consistent identity across sources.</p>','DATA CONTRACT');
  if(!(await backendReady)){dataControls(false);return;}
  try {dataConfig=await requestJson('/api/data');showSelectedDataset(dataConfig.selected);if(dataConfig.job){byId('dataSources').value=(dataConfig.job.sources||[]).join('\n');byId('dataManifest').value=dataConfig.job.manifest||'';renderDataJob(dataConfig.job);if(dataBusy)pollData(dataConfig.job.id);}}
  catch(error){byId('selectedDataset').textContent='! '+error.message;}
}
initializeData();
