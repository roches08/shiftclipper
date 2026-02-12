// ShiftClipper MVP - Web UI
// - Relative URLs (RunPod proxy-safe)
// - XHR upload for progress
// - Canvas overlay draws click circles

const $ = (id) => document.getElementById(id);

const state = {
  jobId: null,
  polling: false,
  selectMode: false,
  clicks: [], // {t, x, y} normalized
  lastStatus: null,
};

function setPill(text){ $('pill').textContent = text || 'idle'; }

function setBar(pct, label){
  const v = Math.max(0, Math.min(100, Number(pct ?? 0)));
  $('barFill').style.width = `${v}%`;
  // no fancy colors, keep simple
  $('barFill').style.background = v >= 90 ? '#0a7a2f' : '#2b7cff';
  $('barText').textContent = label || `${v}%`;
}

function setStatus(stage, pct, message){
  setPill(stage || 'idle');
  const lbl = `${stage || 'idle'} — ${Math.round(Number(pct ?? 0))}%${message ? ' • ' + message : ''}`;
  setBar(pct ?? 0, lbl);
}

function show(obj){
  $('out').textContent = JSON.stringify(obj ?? {}, null, 2);
}

function showClips(obj){
  $('clips').textContent = obj ? JSON.stringify(obj, null, 2) : '—';
  // render links
  const box = $('clipsLinks');
  box.innerHTML = '';
  if(!obj) return;
  const jobId = state.jobId;
  const mk = (href, text) => {
    const a = document.createElement('a');
    a.href = href;
    a.target = '_blank';
    a.rel = 'noreferrer';
    a.textContent = text;
    box.appendChild(a);
  };
  if(obj.combined_url) mk(obj.combined_url, '▶ Combined MP4');
  if(Array.isArray(obj.clips)){
    obj.clips.forEach((c, i)=>{
      if(c && c.url) mk(c.url, `▶ Clip ${String(i+1).padStart(2,'0')} (${c.start.toFixed(2)}–${c.end.toFixed(2)}s)`);
    });
  }
}

function updateButtons(meta){
  const haveJob = !!state.jobId;
  const haveFile = !!$('file').files?.length;
  const proxyReady = !!(meta && meta.proxy_ready);

  $('btnUpload').disabled = !haveJob || !haveFile;
  $('btnCancel').disabled = !haveJob;

  $('btnSelect').disabled = !proxyReady;
  $('btnClearClicks').disabled = state.clicks.length === 0;

  $('btnSave').disabled = !haveJob || state.clicks.length < 3;
  const isReady = meta && (meta.status === 'ready' || meta.stage === 'ready' || meta.status === 'uploaded' || meta.stage === 'uploaded' || meta.status === 'done');
  $('btnRun').disabled = !haveJob || !isReady || state.clicks.length < 3;
}

function renderClicks(){
  $('clickCount').textContent = String(state.clicks.length);
  $('clickWarn').textContent = state.clicks.length < 3 ? ' (need at least 3)' : '';
  if(state.clicks.length === 0){
    $('clickList').textContent = '—';
  }else{
    const last = state.clicks.slice(-8);
    $('clickList').textContent = last.map((c, i)=>{
      const idx = state.clicks.length - last.length + i + 1;
      return `#${idx}: t=${c.t.toFixed(2)}s x=${c.x.toFixed(3)} y=${c.y.toFixed(3)}`;
    }).join(' | ');
  }
  drawOverlay();
}

async function apiJson(method, url, body){
  const opt = {method, headers:{}};
  if(body !== undefined){
    opt.headers['Content-Type'] = 'application/json';
    opt.body = JSON.stringify(body);
  }
  const r = await fetch(url, opt);
  if(!r.ok){
    const txt = await r.text().catch(()=> '');
    throw new Error(`HTTP ${r.status} ${txt}`.trim());
  }
  return await r.json();
}

function setVideoSrc(url){
  const v = $('vid');
  if(!url) return;
  const bust = `ts=${Date.now()}`;
  const sep = url.includes('?') ? '&' : '?';
  v.src = `${url}${sep}${bust}`;
  v.load();
  // resize overlay after metadata loads
}

async function pollStatus(loop=false){
  if(!state.jobId) return;
  if(state.polling && loop) return;
  state.polling = loop;

  try{
    while(true){
      const meta = await apiJson('GET', `/jobs/${state.jobId}/status`);
      state.lastStatus = meta;
      show(meta);

      const stage = meta.stage || meta.status || 'unknown';
      const pct = meta.progress ?? 0;
      setStatus(stage, pct, meta.message || '');

      if(meta.proxy_ready && meta.proxy_url){
        $('previewHint').textContent = 'Proxy ready. Click Select Player to start collecting clicks.';
        if(!$('vid').src.includes(meta.proxy_url)){
          setVideoSrc(meta.proxy_url);
        }
      } else {
        $('previewHint').textContent = meta.message || 'Waiting for proxy…';
      }

      updateButtons(meta);

      if(!loop) break;
      if(meta.status === 'done' || meta.status === 'error') break;
      await new Promise(res=>setTimeout(res, 1200));
    }
  } catch(e){
    setStatus('error', 0, String(e));
    show({error: String(e)});
  } finally {
    state.polling = false;
  }
}

async function createJob(){
  const r = await apiJson('POST','/jobs');
  state.jobId = r.job_id;
  $('jobId').textContent = state.jobId;
  state.clicks = [];
  state.selectMode = false;
  $('btnSelect').textContent = 'Select Player (clicks OFF)';
  renderClicks();

  show(r);
  setStatus('created', 0, 'Job created');
  updateButtons({status:'created', stage:'created', progress:0, proxy_ready:false});
  await pollStatus(false);
}

function uploadVideo(){
  if(!state.jobId) return;
  const f = $('file').files[0];
  if(!f) return;

  const form = new FormData();
  form.append('file', f);

  // Reset UI
  setStatus('uploading', 0, 'Uploading…');
  $('previewHint').textContent = 'Uploading…';
  state.clicks = [];
  state.selectMode = false;
  $('btnSelect').textContent = 'Select Player (clicks OFF)';
  renderClicks();
  $('vid').removeAttribute('src');
  $('vid').load();

  const xhr = new XMLHttpRequest();
  xhr.open('POST', `/jobs/${state.jobId}/upload`);

  xhr.upload.onprogress = (e)=>{
    if(e.lengthComputable){
      // upload maps to 0-40%
      const p = Math.max(0, Math.min(40, Math.round((e.loaded/e.total)*40)));
      setStatus('uploading', p, `Uploading… ${Math.round((e.loaded/e.total)*100)}%`);
    }
  };

  xhr.onerror = ()=>{
    setStatus('error', 0, 'Upload failed (network error)');
  };

  xhr.onload = async ()=>{
    if(xhr.status >= 200 && xhr.status < 300){
      setStatus('proxy', 40, 'Upload complete. Building proxy…');
      try{
        await pollStatus(true); // will switch to proxy_ready when ready
      } catch(e){
        setStatus('error', 0, `Proxy poll error: ${String(e)}`);
      }
    } else {
      setStatus('error', 0, `Upload failed (HTTP ${xhr.status})`);
      show({error: xhr.responseText || `HTTP ${xhr.status}`});
    }
  };

  xhr.send(form);
}

async function saveSetup(){
  if(!state.jobId) return;
  const payload = {
    camera_mode: $('cameraMode').value,
    player_number: ($('playerNumber').value || '').trim(),
    jersey_color: $('jerseyColor').value,
    clicks: state.clicks,
    clicks_count: state.clicks.length,
  };
  const r = await apiJson('PUT', `/jobs/${state.jobId}/setup`, payload);
  show(r);
  await pollStatus(false);
}

async function runJob(){
  if(!state.jobId) return;
  setStatus('queued', 20, 'Queued…');
  const r = await apiJson('POST', `/jobs/${state.jobId}/run`);
  show(r);
  await pollStatus(true);

  try{
    const res = await apiJson('GET', `/jobs/${state.jobId}/results`);
    showClips(res);
  }catch(e){
    showClips({error:String(e)});
  }
}

async function cancelJob(){
  if(!state.jobId) return;
  const r = await apiJson('GET', `/jobs/${state.jobId}/cancel`);
  show(r);
  await pollStatus(false);
}

function toggleSelect(){
  state.selectMode = !state.selectMode;
  $('btnSelect').textContent = state.selectMode ? 'Select Player (clicks ON)' : 'Select Player (clicks OFF)';
  $('previewHint').textContent = state.selectMode ? 'Click the player in the video 3–8 times.' : 'Selection off.';
}

function clearClicks(){
  state.clicks = [];
  renderClicks();
  updateButtons(state.lastStatus);
}

function handleVideoClick(ev){
  if(!state.selectMode) return;
  const v = $('vid');
  if(!v || !v.videoWidth || !v.videoHeight) return;

  const rect = v.getBoundingClientRect();
  const x = (ev.clientX - rect.left) / rect.width;
  const y = (ev.clientY - rect.top) / rect.height;
  const cx = Math.max(0, Math.min(1, x));
  const cy = Math.max(0, Math.min(1, y));

  state.clicks.push({t: Number(v.currentTime || 0), x: cx, y: cy});
  renderClicks();
  updateButtons(state.lastStatus);
}

function drawOverlay(){
  const v = $('vid');
  const c = $('overlay');
  if(!c || !v) return;

  const rect = v.getBoundingClientRect();
  const w = Math.max(1, Math.round(rect.width));
  const h = Math.max(1, Math.round(rect.height));
  if(c.width !== w || c.height !== h){
    c.width = w;
    c.height = h;
  }
  const ctx = c.getContext('2d');
  ctx.clearRect(0,0,c.width,c.height);

  if(state.clicks.length === 0) return;

  const color = $('jerseyColor')?.value || '#00c853';
  ctx.lineWidth = 3;

  state.clicks.forEach((p, i)=>{
    const x = p.x * c.width;
    const y = p.y * c.height;
    const r = 12;

    // outer ring
    ctx.strokeStyle = color;
    ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2); ctx.stroke();

    // label bubble
    ctx.fillStyle = 'rgba(0,0,0,0.55)';
    ctx.beginPath(); ctx.arc(x,y,r+10,0,Math.PI*2); ctx.fill();

    ctx.fillStyle = '#fff';
    ctx.font = '12px system-ui';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(String(i+1), x, y);
  });
}

function wire(){
  $('btnCreate').addEventListener('click', createJob);
  $('btnUpload').addEventListener('click', uploadVideo);
  $('btnSave').addEventListener('click', saveSetup);
  $('btnRun').addEventListener('click', runJob);
  $('btnCancel').addEventListener('click', cancelJob);
  $('btnSelect').addEventListener('click', toggleSelect);
  $('btnClearClicks').addEventListener('click', clearClicks);

  $('file').addEventListener('change', ()=>updateButtons(state.lastStatus));
  $('jerseyColor').addEventListener('change', drawOverlay);

  $('vid').addEventListener('click', handleVideoClick);
  $('vid').addEventListener('loadedmetadata', drawOverlay);
  window.addEventListener('resize', drawOverlay);

  setStatus('idle', 0, '');
  show({});
  showClips(null);
  renderClicks();
  updateButtons(null);
}

wire();
