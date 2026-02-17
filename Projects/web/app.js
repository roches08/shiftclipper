const $ = (id) => document.getElementById(id);
const state = { jobId: null, clicks: [], lastStatus: null };

const PRESET_HELP = {
  broadcast: "Broadcast: Standard TV side angle. OCR works intermittently.",
  broadcast_wide: "Broadcast (Wide/Youth): TV angle but wide. OCR weaker; relies more on motion + color.",
  tactical: "Tactical: High-wide overhead. OCR usually unreliable; relies on tracking.",
};
const TRACK_HELP = {
  clip: "Clips: Creates highlight clips when the target is confidently tracked.",
  shift: "Shifts: Tracks full shifts on ice; outputs shift start/end + total TOI.",
};
const CAMERA_DEFAULTS = {
  broadcast: { detect_stride: 1, ocr_min_conf: 0.20, lock_seconds_after_confirm: 4.0, gap_merge_seconds: 2.5, lost_timeout: 1.5, min_track_seconds: 0.75 },
  broadcast_wide: { detect_stride: 1, ocr_min_conf: 0.18, lock_seconds_after_confirm: 6.0, gap_merge_seconds: 3.0, lost_timeout: 1.9, min_track_seconds: 0.75 },
  tactical: { detect_stride: 3, ocr_min_conf: 0.30, lock_seconds_after_confirm: 5.0, gap_merge_seconds: 2.0, lost_timeout: 1.8, min_track_seconds: 0.75 },
};

async function j(method, path, body){
  const opt = { method, headers: {} };
  if(body !== undefined){ opt.headers['Content-Type'] = 'application/json'; opt.body = JSON.stringify(body); }
  const r = await fetch(path, opt);
  if(!r.ok) throw new Error(await r.text());
  return r.json();
}
function refreshHelp(){
  const cm = $('cameraMode').value; const tm = $('trackingMode').value;
  $('cameraHelp').textContent = PRESET_HELP[cm];
  $('trackingHelp').textContent = TRACK_HELP[tm];
  $('shiftOnly').style.display = tm === 'shift' ? 'inline-flex' : 'none';
  $('verifyBanner').style.display = $('verifyMode').value === 'on' ? 'block' : 'none';
}
function applyPreset(){
  const p = CAMERA_DEFAULTS[$('cameraMode').value];
  $('detectStride').value = p.detect_stride;
  $('ocrMinConf').value = p.ocr_min_conf;
  $('lockSeconds').value = p.lock_seconds_after_confirm;
  $('mergeGap').value = p.gap_merge_seconds;
  $('lostTimeout').value = p.lost_timeout;
  $('minTrack').value = p.min_track_seconds;
  refreshHelp();
}

async function createJob(){
  const r = await j('POST', '/jobs', { name: 'ui-job' });
  state.jobId = r.job_id;
  $('jobId').textContent = state.jobId;
  $('btnUpload').disabled = false;
  $('btnSave').disabled = false;
  $('btnRun').disabled = false;
}

async function upload(){
  const f = $('file').files[0]; if(!f || !state.jobId) return;
  const fd = new FormData(); fd.append('file', f);
  const r = await fetch(`/jobs/${state.jobId}/upload`, { method: 'POST', body: fd });
  if(!r.ok) throw new Error(await r.text());
  await poll();
}

function payload(){
  return {
    camera_mode: $('cameraMode').value,
    tracking_mode: $('trackingMode').value,
    verify_mode: $('verifyMode').value === 'on',
    player_number: $('playerNumber').value,
    jersey_color: $('jerseyColor').value,
    jersey_color_hex: $('jerseyColor').value,
    jersey_color_rgb: (() => { const v=$('jerseyColor').value.replace("#",""); return { r: parseInt(v.slice(0,2),16), g: parseInt(v.slice(2,4),16), b: parseInt(v.slice(4,6),16) }; })(),
    color_tolerance: Number($('colorTolerance').value),
    extend_sec: Number($('extendSec').value),
    detect_stride: Number($('detectStride').value),
    ocr_min_conf: Number($('ocrMinConf').value),
    lock_seconds_after_confirm: Number($('lockSeconds').value),
    lost_timeout: Number($('lostTimeout').value),
    gap_merge_seconds: Number($('mergeGap').value),
    min_track_seconds: Number($('minTrack').value),
    bench_zone_ratio: Number($('benchZone').value),
    debug_overlay: $('debugOverlay').checked,
    debug_timeline: $('debugTimeline').checked,
    clicks: state.clicks,
  };
}

async function save(){ await j('PUT', `/jobs/${state.jobId}/setup`, payload()); await poll(); }

async function run(){
  if($('verifyMode').value === 'on'){
    const ok = confirm('Verify mode will not create clips/combined video. Continue?\nCancel = Turn off verify + run');
    if(!ok){ $('verifyMode').value = 'off'; refreshHelp(); }
  }
  await save();
  await j('POST', `/jobs/${state.jobId}/run`);
  for(let i=0;i<200;i++){
    const s = await j('GET', `/jobs/${state.jobId}/status`);
    $('out').textContent = JSON.stringify(s, null, 2);
    if(['done','failed','cancelled','verified','done_no_clips'].includes(s.status)) break;
    await new Promise(r => setTimeout(r, 1200));
  }
  const res = await j('GET', `/jobs/${state.jobId}/results`);
  $('clips').textContent = JSON.stringify(res, null, 2);
  renderResults(res);
}

function renderResults(res){
  const msg = $('resultsMessage');
  if(res.status === 'verified') msg.textContent = 'Verify completed. No clips/combined generated.';
  else if(res.status === 'done_no_clips') msg.textContent = 'Run completed but no clips were found. See debug overlay/timeline.';
  else msg.textContent = res.message || 'Done';
  const a = $('artifacts'); a.innerHTML='';
  const art = res.artifacts || {};
  const add = (label, url) => { if(!url) return; const d=document.createElement('div'); d.innerHTML=`<a target='_blank' href='${url}'>${label}</a>`; a.appendChild(d); };
  (art.clips || []).forEach((c, i) => add(`clip ${i+1}`, c.url));
  add('combined mp4', art.combined_url || res.combined_url);
  add('debug overlay', art.debug_overlay_url);
  add('debug timeline', art.debug_timeline_url);
}

async function poll(){
  if(!state.jobId) return;
  const s = await j('GET', `/jobs/${state.jobId}/status`);
  $('out').textContent = JSON.stringify(s, null, 2);
  if(s.proxy_url) $('vid').src = s.proxy_url;
}

window.addEventListener('DOMContentLoaded', () => {
  $('btnCreate').onclick = createJob;
  $('btnUpload').onclick = upload;
  $('btnSave').onclick = save;
  $('btnRun').onclick = run;
  $('cameraMode').onchange = applyPreset;
  $('trackingMode').onchange = refreshHelp;
  $('verifyMode').onchange = refreshHelp;
  applyPreset();
});
