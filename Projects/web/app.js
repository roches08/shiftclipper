// ShiftClipper MVP - Web UI
// - Relative URLs (RunPod proxy-safe)
// - XHR upload for progress
// - Canvas overlay draws click circles
// - Setup includes: camera_mode, player_number, jersey_color, opponent_color, extend_sec, verify_mode, clicks
// - Min/Max clicks configurable (default min=2, max=6)

const $ = (id) => document.getElementById(id);

const state = {
  jobId: null,
  polling: false,
  selectMode: false,
  clicks: [], // {t, x, y}
  lastStatus: null,
};

function num(v, fallback){
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function getMinClicks(){ return num($('minClicks')?.value, 2); }
function getMaxClicks(){ return num($('maxClicks')?.value, 6); }

function setPill(text){ $('pill').textContent = text || 'idle'; }

function setBar(pct, label){
  const v = Math.max(0, Math.min(100, Number(pct ?? 0)));
  $('barFill').style.width = `${v}%`;
  $('barText').textContent = label || `${v}%`;
}

function show(obj){ $('out').textContent = JSON.stringify(obj ?? {}, null, 2); }
function showClips(obj){ $('clips').textContent = obj ? JSON.stringify(obj, null, 2) : '—'; }

function updateButtons(meta){
  const haveJob = !!state.jobId;
  const proxyReady = !!(meta && meta.proxy_ready);
  const clickCount = state.clicks.length;
  const minClicks = getMinClicks();

  $('btnUpload').disabled = !haveJob || !$('file').files?.length;
  $('btnSelect').disabled = !proxyReady;
  $('btnClearClicks').disabled = clickCount === 0;

  // Save requires >= minClicks (NOT hardcoded 3)
  $('btnSave').disabled = !haveJob || clickCount < minClicks;

  $('btnRun').disabled = !haveJob || !(meta && (meta.status === 'ready' || meta.stage === 'ready'));
  $('btnCancel').disabled = !haveJob;
}

function renderClicks(){
  const minClicks = getMinClicks();
  const maxClicks = getMaxClicks();

  $('clickCount').textContent = String(state.clicks.length);

  if(state.clicks.length < minClicks){
    $('clickWarn').textContent = ` (need at least ${minClicks})`;
  }else{
    $('clickWarn').textContent = '';
  }

  if(state.clicks.length >= maxClicks){
    $('clickWarn').textContent = ` (max ${maxClicks} reached)`;
  }

  if(state.clicks.length === 0){
    $('clickList').textContent = '—';
    return;
  }

  $('clickList').textContent = state.clicks
    .slice(-12)
    .map((c,i)=>{
      const idx = state.clicks.length - Math.min(12,state.clicks.length) + i + 1;
      return `#${idx}: t=${c.t.toFixed(2)}s x=${c.x.toFixed(3)} y=${c.y.toFixed(3)}`;
    })
    .join(' | ');
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
}

function resizeOverlay(){
  const v = $('vid');
  const c = $('overlay');
  if(!v || !c) return;
  const rect = v.getBoundingClientRect();
  c.width = Math.round(rect.width);
  c.height = Math.round(rect.height);
  c.style.width = `${Math.round(rect.width)}px`;
  c.style.height = `${Math.round(rect.height)}px`;
}

function drawOverlay(){
  const c = $('overlay');
  if(!c) return;
  const ctx = c.getContext('2d');
  ctx.clearRect(0,0,c.width,c.height);

  // click circles
  for(const click of state.clicks){
    const x = click.x * c.width;
    const y = click.y * c.height;
    ctx.beginPath();
    ctx.arc(x, y, 12, 0, Math.PI*2);
    ctx.lineWidth = 3;
    ctx.strokeStyle = '#00ff88';
    ctx.stroke();
    ctx.fillStyle = 'rgba(0,255,136,0.15)';
    ctx.fill();
  }
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
      setPill(meta.stage || meta.status || 'unknown');
      setBar(
        meta.progress ?? 0,
        `${meta.stage || meta.status || 'unknown'} — ${meta.progress ?? 0}% ${meta.message ? '• ' + meta.message : ''}`
      );

      if(meta.proxy_ready && meta.proxy_url){
        const minClicks = getMinClicks();
        const maxClicks = getMaxClicks();
        $('previewHint').textContent = `Proxy ready. Click Select Player then click the player ${minClicks}–${maxClicks} times (torso).`;
        if(!$('vid').src.includes(meta.proxy_url)){
          setVideoSrc(meta.proxy_url);
        }
      }else{
        $('previewHint').textContent = meta.message || 'Waiting for proxy…';
      }

      updateButtons(meta);

      if(!loop) break;
      if(meta.status === 'done' || meta.status === 'error') break;
      await new Promise(res=>setTimeout(res, 1200));
    }
  }catch(e){
    $('out').textContent = JSON.stringify({error: String(e)}, null, 2);
    setPill('error');
  }finally{
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
  drawOverlay();
  setPill('created');
  setBar(0, 'created — 0%');
  updateButtons({status:'created', stage:'created', progress:0, proxy_ready:false});
  await pollStatus(false);
}

function uploadVideo(){
  if(!state.jobId) return;
  const f = $('file').files[0];
  if(!f) return;

  const form = new FormData();
  form.append('file', f);

  setPill('uploading');
  setBar(1, 'Uploading…');

  const xhr = new XMLHttpRequest();
  xhr.open('POST', `/jobs/${state.jobId}/upload`);

  xhr.upload.onprogress = (e)=>{
    if(e.lengthComputable){
      const pct = Math.max(1, Math.min(40, Math.round((e.loaded/e.total)*40)));
      setBar(pct, `Uploading… ${Math.round((e.loaded/e.total)*100)}%`);
    }
  };

  xhr.onerror = ()=>{
    setPill('error');
    setBar(0, 'Upload failed (network).');
  };

  xhr.onload = async ()=>{
    if(xhr.status >= 200 && xhr.status < 300){
      setPill('proxy');
      setBar(45, 'Upload complete. Building proxy…');
      await pollStatus(true);
    } else {
      setPill('error');
      setBar(0, `Upload failed (HTTP ${xhr.status}).`);
      $('out').textContent = xhr.responseText || `HTTP ${xhr.status}`;
    }
  };

  xhr.send(form);
}

async function saveSetup(){
  if(!state.jobId) return;

  const minClicks = getMinClicks();
  const maxClicks = getMaxClicks();

  const payload = {
    camera_mode: $('cameraMode').value,
    player_number: ($('playerNumber').value || '').trim(),
    jersey_color: $('jerseyColor').value,
    opponent_color: $('oppColor')?.value || null,

    extend_sec: Number($('extendSec').value || 2),
    verify_mode: $('verifyMode').value === 'on',

    // hints for backend (safe if ignored)
    min_clicks: minClicks,
    max_clicks: maxClicks,
    use_color_filter: !!$('useColor')?.checked,
    use_number_tracking: !!$('useNumber')?.checked,

    clicks: state.clicks,
    clicks_count: state.clicks.length,
  };

  const r = await apiJson('PUT', `/jobs/${state.jobId}/setup`, payload);
  show(r);
  await pollStatus(false);
}

async function runJob(){
  if(!state.jobId) return;
  const r = await apiJson('POST', `/jobs/${state.jobId}/run`);
  show(r);
  await pollStatus(true);

  try{
    const res = await apiJson('GET', `/jobs/${state.jobId}/results`);
    showClips(res);
    renderClipsUi(res);
  }catch(e){
    showClips({error:String(e)});
    $('clipsUi').textContent = '—';
  }
}

async function cancelJob(){
  // UI-only placeholder
  setPill('cancel');
  setBar(0, 'Cancelled (UI only).');
}

function toggleSelect(){
  state.selectMode = !state.selectMode;
  $('btnSelect').textContent = state.selectMode ? 'Select Player (clicks ON)' : 'Select Player (clicks OFF)';
  if(state.selectMode){
    const minClicks = getMinClicks();
    const maxClicks = getMaxClicks();
    $('previewHint').textContent = `Clicks ON. Click the player ${minClicks}–${maxClicks} times (torso).`;
  }else{
    $('previewHint').textContent = 'Selection off.';
  }
}

function clearClicks(){
  state.clicks = [];
  renderClicks();
  drawOverlay();
  updateButtons(state.lastStatus);
}

function handleVideoClick(ev){
  if(!state.selectMode) return;

  const maxClicks = getMaxClicks();
  if(state.clicks.length >= maxClicks){
    renderClicks();
    updateButtons(state.lastStatus);
    return;
  }

  const v = $('vid');
  if(!v) return;

  const rect = v.getBoundingClientRect();
  const x = (ev.clientX - rect.left) / rect.width;
  const y = (ev.clientY - rect.top) / rect.height;

  const cx = Math.max(0, Math.min(1, x));
  const cy = Math.max(0, Math.min(1, y));
  state.clicks.push({t: Number(v.currentTime || 0), x: cx, y: cy});

  renderClicks();
  drawOverlay();
  updateButtons(state.lastStatus);
}

function renderClipsUi(res){
  const el = $('clipsUi');
  el.innerHTML = '';
  if(!res || !res.clips || res.clips.length === 0){
    el.textContent = '—';
    return;
  }

  const list = document.createElement('div');
  list.style.display = 'grid';
  list.style.gap = '8px';

  res.clips.forEach((c, i)=>{
    const a = document.createElement('a');
    a.href = c.url;
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
    a.textContent = `▶ Clip ${String(i+1).padStart(2,'0')} (${c.start.toFixed(2)}–${c.end.toFixed(2)}s)`;
    list.appendChild(a);
  });

  if(res.combined_url){
    const wrap = document.createElement('div');
    wrap.style.marginTop = '8px';
    const a = document.createElement('a');
    a.href = res.combined_url;
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
    a.textContent = `▶ Combined video`;
    wrap.appendChild(a);
    list.appendChild(wrap);
  }

  el.appendChild(list);
}

function wire(){
  $('btnCreate').addEventListener('click', createJob);
  $('btnUpload').addEventListener('click', uploadVideo);
  $('btnSave').addEventListener('click', saveSetup);
  $('btnRun').addEventListener('click', runJob);
  $('btnCancel').addEventListener('click', cancelJob);
  $('btnSelect').addEventListener('click', toggleSelect);
  $('btnClearClicks').addEventListener('click', clearClicks);

  $('file').addEventListener('change', ()=> updateButtons(state.lastStatus));
  $('vid').addEventListener('click', handleVideoClick);

  // re-render warnings when min/max changes
  $('minClicks')?.addEventListener('change', ()=>{
    renderClicks();
    updateButtons(state.lastStatus);
  });
  $('maxClicks')?.addEventListener('change', ()=>{
    // trim clicks if user lowers max below current count
    const maxClicks = getMaxClicks();
    if(state.clicks.length > maxClicks){
      state.clicks = state.clicks.slice(0, maxClicks);
      drawOverlay();
    }
    renderClicks();
    updateButtons(state.lastStatus);
  });

  window.addEventListener('resize', ()=>{ resizeOverlay(); drawOverlay(); });
  $('vid').addEventListener('loadedmetadata', ()=>{ resizeOverlay(); drawOverlay(); });

  setPill('idle');
  setBar(0, 'idle');
  show({});
  showClips(null);
  renderClicks();
  resizeOverlay();
  drawOverlay();
  updateButtons(null);
}

wire();

