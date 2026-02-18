const $ = (id) => document.getElementById(id);
const state = {
  jobId: null,
  clicks: [],
  lastStatus: null,
  pollTimer: null,
  proxySrc: null,
  uploading: false,
  uploadXhr: null,
};

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
  broadcast: { detect_stride: 2, yolo_imgsz: 512, ocr_every_n: 12, ocr_min_conf: 0.20, lock_seconds_after_confirm: 4.0, gap_merge_seconds: 1.5, lost_timeout: 1.5, min_track_seconds: 0.75 },
  broadcast_wide: { detect_stride: 3, yolo_imgsz: 512, ocr_every_n: 16, ocr_min_conf: 0.18, lock_seconds_after_confirm: 6.0, gap_merge_seconds: 1.5, lost_timeout: 1.9, min_track_seconds: 0.75 },
  tactical: { detect_stride: 4, yolo_imgsz: 416, ocr_every_n: 20, ocr_min_conf: 0.30, lock_seconds_after_confirm: 5.0, gap_merge_seconds: 1.5, lost_timeout: 1.8, min_track_seconds: 0.75 },
};

const { getAdvancedPreset } = window.ShiftClipperPresets;

const STAGE_META = [
  { key: 'uploading', label: 'Uploading', icon: '⬆️' },
  { key: 'queued', label: 'Queued', icon: '🕒' },
  { key: 'tracking', label: 'Tracking', icon: '🎯' },
  { key: 'clips', label: 'Creating clips', icon: '✂️' },
  { key: 'combined', label: 'Combining video', icon: '🎬' },
  { key: 'done', label: 'Done', icon: '✅' },
  { key: 'failed', label: 'Failed', icon: '❌' },
];

async function j(method, path, body){
  const opt = { method, headers: {} };
  if(body !== undefined){ opt.headers['Content-Type'] = 'application/json'; opt.body = JSON.stringify(body); }
  const r = await fetch(path, opt);
  if(!r.ok) throw new Error(await r.text());
  return r.json();
}

function mb(n){ return (n / (1024 * 1024)).toFixed(1); }

function setJobId(jobId){
  const changed = state.jobId !== jobId;
  state.jobId = jobId;
  if(jobId) localStorage.setItem('shiftclipper.jobId', jobId);
  else localStorage.removeItem('shiftclipper.jobId');
  if(changed){
    state.proxySrc = null;
    state.clicks = [];
    drawClickMarkers();
  }
  $('jobId').textContent = jobId || '—';
  $('btnUpload').disabled = !jobId;
  $('btnSave').disabled = !jobId;
  $('btnCancel').disabled = !jobId;
  $('btnClear').disabled = !jobId;
  updateRunButtonState();
}

function updateRunButtonState(){
  const skipSeeding = $('skipSeeding').checked;
  const hasSeed = state.clicks.length > 0;
  $('btnRun').disabled = !state.jobId || (!hasSeed && !skipSeeding);
  $('seedStatus').textContent = `Seed clicks: ${state.clicks.length}`;
  const clickList = state.clicks
    .map((c, idx) => `#${idx + 1} t=${Number(c.t).toFixed(2)} x=${Number(c.x).toFixed(3)} y=${Number(c.y).toFixed(3)}`)
    .join(' | ');
  $('seedClicksList').textContent = clickList || '—';
}


function refreshHelp(){
  const cm = $('cameraMode').value; const tm = $('trackingMode').value;
  $('cameraHelp').textContent = PRESET_HELP[cm];
  $('trackingHelp').textContent = TRACK_HELP[tm];
  $('shiftOnly').style.display = tm === 'shift' ? 'inline-flex' : 'none';
  $('verifyBanner').style.display = $('verifyMode').value === 'on' ? 'block' : 'none';
  updateRunButtonState();
}

function applyAdvancedPreset(name){
  const p = getAdvancedPreset(name);
  $('scoreLockThreshold').value = p.score_lock_threshold;
  $('scoreUnlockThreshold').value = p.score_unlock_threshold;
  $('lostTimeout').value = p.lost_timeout;
  $('reacquireWindowSeconds').value = p.reacquire_window_seconds;
  $('reacquireScoreLockThreshold').value = p.reacquire_score_lock_threshold;
  $('mergeGap').value = p.gap_merge_seconds;
  $('lockSeconds').value = p.lock_seconds_after_confirm;
  $('minTrack').value = p.min_track_seconds;
  $('minClipSeconds').value = p.min_clip_seconds;
  $('allowUnconfirmedClips').checked = !!p.allow_unconfirmed_clips;
  $('allowSeedClips').checked = !!p.allow_seed_clips;
  $('seedLockSeconds').value = p.seed_lock_seconds;
  $('seedIouMin').value = p.seed_iou_min;
  $('seedDistMax').value = p.seed_dist_max;
  $('seedBonus').value = p.seed_bonus;
  $('seedWindowS').value = p.seed_window_s;
  $('ocrDisable').checked = !!p.ocr_disable;
  $('ocrEveryNFrames').value = p.ocr_every_n_frames;
  $('ocrMinConf').value = p.ocr_min_conf;
  $('ocrVetoConf').value = p.ocr_veto_conf;
  $('ocrVetoSeconds').value = p.ocr_veto_seconds;
  $('detectStride').value = p.detect_stride;
  $('yoloImgsz').value = p.yolo_imgsz;
  $('yoloBatch').value = p.yolo_batch;
  $('trackerType').value = p.tracker_type;
  $('reidEnable').checked = !!p.reid_enable;
  $('reidModel').value = p.reid_model;
  $('reidEveryNFrames').value = p.reid_every_n_frames;
  $('reidWeight').value = p.reid_weight;
  $('reidMinSim').value = p.reid_min_sim;
  $('reidCropExpand').value = p.reid_crop_expand;
  $('reidBatch').value = p.reid_batch;
  $('reidDevice').value = p.reid_device;
  $('swapGuardSeconds').value = p.swap_guard_seconds;
  $('swapGuardBonus').value = p.swap_guard_bonus;
}

function applyPreset(){
  const p = CAMERA_DEFAULTS[$('cameraMode').value];
  $('detectStride').value = p.detect_stride;
  $('ocrMinConf').value = p.ocr_min_conf;
  $('lockSeconds').value = p.lock_seconds_after_confirm;
  $('mergeGap').value = p.gap_merge_seconds;
  $('lostTimeout').value = p.lost_timeout;
  $('minTrack').value = p.min_track_seconds;
  applyAdvancedPreset('balanced');
  refreshHelp();
}

function setValueIfDefined(id, value){
  if(value === undefined || value === null) return;
  $(id).value = value;
}

function setCheckedIfDefined(id, value){
  if(value === undefined || value === null) return;
  $(id).checked = !!value;
}

function renderStepper(stage){
  const idx = Math.max(0, STAGE_META.findIndex(s => s.key === stage));
  const isFailed = stage === 'failed';
  const nodes = STAGE_META.filter(s => s.key !== 'failed' || isFailed).map((s, i) => {
    let cls = 'todo';
    if (s.key === stage) cls = isFailed ? 'failed' : 'active';
    else if (i < idx && !isFailed) cls = 'done';
    return `<div class="step ${cls}"><span>${s.icon}</span><span>${s.label}</span></div>`;
  });
  $('stepper').innerHTML = nodes.join('');
}

function updateProgressUi(status){
  const progress = Number(status.progress || 0);
  $('overallProgress').value = Math.max(0, Math.min(100, progress));
  $('overallProgressText').textContent = `${progress}%`;
  const stage = status.stage || status.status || 'queued';
  renderStepper(stage);
  const message = status.error || status.message || '—';
  $('progressMessage').textContent = message;
}

async function createJob(){
  const r = await j('POST', '/jobs', { name: 'ui-job' });
  setJobId(r.job_id);
  await loadSetup();
  await pollOnce();
}

async function upload(){
  const f = $('file').files[0]; if(!f || !state.jobId) return;
  if (state.uploading) return;
  const fd = new FormData(); fd.append('file', f);
  $('progressMessage').textContent = 'Uploading...';
  state.uploading = true;

  await new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    state.uploadXhr = xhr;
    xhr.open('POST', `/jobs/${state.jobId}/upload`, true);
    xhr.upload.onprogress = (evt) => {
      if (!evt.lengthComputable) return;
      const percent = Math.round((evt.loaded / evt.total) * 100);
      $('overallProgress').value = percent;
      $('overallProgressText').textContent = `${percent}%`;
      $('progressMessage').textContent = `Uploading: ${mb(evt.loaded)}/${mb(evt.total)} MB (${percent}%)`;
      renderStepper('uploading');
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        $('progressMessage').textContent = 'Processing…';
        renderStepper('queued');
        resolve();
      } else {
        reject(new Error(xhr.responseText || `Upload failed (${xhr.status})`));
      }
    };
    xhr.onerror = () => reject(new Error('Upload failed'));
    xhr.send(fd);
  }).finally(() => {
    state.uploading = false;
    state.uploadXhr = null;
  });

  startPolling();
}

function payload(){
  const toNumber = (id) => parseFloat($(id).value);
  const toInt = (id) => parseInt($(id).value, 10);
  return {
    camera_mode: $('cameraMode').value,
    tracking_mode: $('trackingMode').value,
    verify_mode: $('verifyMode').value === 'on',
    skip_seeding: $('skipSeeding').checked,
    player_number: $('playerNumber').value,
    jersey_color: $('jerseyColor').value,
    jersey_color_hex: $('jerseyColor').value,
    opponent_color: $('opponentColor').value,
    jersey_color_rgb: (() => { const v=$('jerseyColor').value.replace("#",""); return { r: parseInt(v.slice(0,2),16), g: parseInt(v.slice(2,4),16), b: parseInt(v.slice(4,6),16) }; })(),
    color_tolerance: toInt('colorTolerance'),
    extend_sec: toNumber('extendSec'),
    detect_stride: toInt('detectStride'),
    yolo_imgsz: toInt('yoloImgsz'),
    yolo_batch: toInt('yoloBatch'),
    tracker_type: $('trackerType').value || 'bytetrack',
    ocr_min_conf: toNumber('ocrMinConf'),
    ocr_disable: $('ocrDisable').checked,
    ocr_every_n_frames: toInt('ocrEveryNFrames'),
    ocr_veto_conf: toNumber('ocrVetoConf'),
    ocr_veto_seconds: toNumber('ocrVetoSeconds'),
    lock_seconds_after_confirm: toNumber('lockSeconds'),
    lost_timeout: toNumber('lostTimeout'),
    gap_merge_seconds: toNumber('mergeGap'),
    score_lock_threshold: toNumber('scoreLockThreshold'),
    score_unlock_threshold: toNumber('scoreUnlockThreshold'),
    reacquire_window_seconds: toNumber('reacquireWindowSeconds'),
    reacquire_score_lock_threshold: toNumber('reacquireScoreLockThreshold'),
    allow_unconfirmed_clips: $('allowUnconfirmedClips').checked,
    allow_seed_clips: $('allowSeedClips').checked,
    min_track_seconds: toNumber('minTrack'),
    min_clip_seconds: toNumber('minClipSeconds'),
    seed_lock_seconds: toNumber('seedLockSeconds'),
    seed_iou_min: toNumber('seedIouMin'),
    seed_dist_max: toNumber('seedDistMax'),
    seed_bonus: toNumber('seedBonus'),
    seed_window_s: toNumber('seedWindowS'),
    clicks_count: state.clicks.length,
    bench_zone_ratio: toNumber('benchZone'),
    debug_overlay: $('debugOverlay').checked,
    debug_timeline: $('debugTimeline').checked,
    transcode_enabled: $('transcodeEnabled').checked,
    transcode_scale_max: toInt('transcodeScaleMax'),
    transcode_fps: $('transcodeFps').value ? toInt('transcodeFps') : null,
    transcode_deinterlace: $('transcodeDeinterlace').checked,
    transcode_denoise: $('transcodeDenoise').checked,
    reid_enable: $('reidEnable').checked,
    reid_model: $('reidModel').value,
    reid_every_n_frames: toInt('reidEveryNFrames'),
    reid_weight: toNumber('reidWeight'),
    reid_min_sim: toNumber('reidMinSim'),
    reid_crop_expand: toNumber('reidCropExpand'),
    reid_batch: toInt('reidBatch'),
    reid_device: $('reidDevice').value,
    swap_guard_seconds: toNumber('swapGuardSeconds'),
    swap_guard_bonus: toNumber('swapGuardBonus'),
    clicks: state.clicks,
  };
}

async function save(){ await j('PUT', `/jobs/${state.jobId}/setup`, payload()); await pollOnce(); }

async function run(){
  if($('verifyMode').value === 'on'){
    const ok = confirm('Verify mode will not create clips/combined video. Continue?\nCancel = Turn off verify + run');
    if(!ok){ $('verifyMode').value = 'off'; refreshHelp(); }
  }
  if (state.clicks.length < 1 && !$('skipSeeding').checked){
    alert('Add at least one seed click or check Skip seeding before running.');
    return;
  }
  await save();
  await j('POST', `/jobs/${state.jobId}/run`);
  startPolling();
}

async function cancel(){
  if(!state.jobId) return;
  await j('POST', `/jobs/${state.jobId}/cancel`);
  if(state.pollTimer) clearInterval(state.pollTimer);
  state.pollTimer = null;
  state.uploading = false;
  if (state.uploadXhr) {
    state.uploadXhr.abort();
    state.uploadXhr = null;
  }
  await pollOnce();
}

function resetUiAfterClear(){
  const video = $('vid');
  video.removeAttribute('src');
  video.load();
  $('resultsMessage').textContent = '—';
  $('artifacts').innerHTML = '';
  $('clips').textContent = '—';
  $('out').textContent = '{}';
  $('overallProgress').value = 0;
  $('overallProgressText').textContent = '0%';
  $('progressMessage').textContent = '—';
  $('stepper').innerHTML = '';
}

async function clearJob(){
  if(!state.jobId) return;
  await j('DELETE', `/jobs/${state.jobId}`);
  if(state.pollTimer) clearInterval(state.pollTimer);
  state.pollTimer = null;
  setJobId(null);
  resetUiAfterClear();
  updateRunButtonState();
}

function clearSeedClicks(){
  state.clicks = [];
  drawClickMarkers();
}

function undoSeedClick(){
  if (state.clicks.length < 1) return;
  state.clicks.pop();
  drawClickMarkers();
}

function drawClickMarkers(){
  const video = $('vid');
  const canvas = $('seedCanvas');
  const wrap = canvas.parentElement;
  const rect = wrap.getBoundingClientRect();
  const w = Math.max(1, Math.round(rect.width));
  const h = Math.max(1, Math.round(rect.height));
  canvas.width = w;
  canvas.height = h;
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;

  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, w, h);
  const current = Number(video.currentTime || 0);
  state.clicks.forEach((c, idx) => {
    const x = c.x * w;
    const y = c.y * h;
    const isNearCurrent = Math.abs(Number(c.t || 0) - current) < 1.0;
    ctx.beginPath();
    ctx.arc(x, y, isNearCurrent ? 8 : 6, 0, Math.PI * 2);
    ctx.fillStyle = isNearCurrent ? 'rgba(55, 185, 255, 0.95)' : 'rgba(255, 84, 84, 0.9)';
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = '#ffffff';
    ctx.font = '12px Arial';
    ctx.fillText(`#${idx + 1}`, x + 10, y - 10);
  });
  updateRunButtonState();
}

function registerSeedClick(evt){
  const vid = $('vid');
  if (!vid.src) return;
  const rect = vid.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return;
  const clickX = evt.clientX - rect.left;
  const clickY = evt.clientY - rect.top;
  const vidWidth = rect.width;
  const vidHeight = rect.height;
  state.clicks.push({
    t: vid.currentTime,
    x: clickX / vidWidth,
    y: clickY / vidHeight,
  });
  drawClickMarkers();
}

async function loadSetup(){
  if (!state.jobId) return;
  applyPreset();
  try {
    const resp = await j('GET', `/jobs/${state.jobId}/setup`);
    const setup = resp.setup || {};
    setValueIfDefined('cameraMode', setup.camera_mode);
    setValueIfDefined('trackingMode', setup.tracking_mode);
    setValueIfDefined('verifyMode', setup.verify_mode ? 'on' : 'off');
    setValueIfDefined('playerNumber', setup.player_number);
    setValueIfDefined('jerseyColor', setup.jersey_color_hex || setup.jersey_color);
    setValueIfDefined('opponentColor', setup.opponent_color);
    setValueIfDefined('colorTolerance', setup.color_tolerance);
    setValueIfDefined('extendSec', setup.extend_sec);
    setValueIfDefined('detectStride', setup.detect_stride);
    setValueIfDefined('ocrMinConf', setup.ocr_min_conf);
    setValueIfDefined('lockSeconds', setup.lock_seconds_after_confirm);
    setValueIfDefined('lostTimeout', setup.lost_timeout);
    setValueIfDefined('mergeGap', setup.gap_merge_seconds);
    setValueIfDefined('scoreLockThreshold', setup.score_lock_threshold);
    setValueIfDefined('scoreUnlockThreshold', setup.score_unlock_threshold);
    setValueIfDefined('reacquireWindowSeconds', setup.reacquire_window_seconds);
    setValueIfDefined('reacquireScoreLockThreshold', setup.reacquire_score_lock_threshold);
    setValueIfDefined('minTrack', setup.min_track_seconds);
    setValueIfDefined('minClipSeconds', setup.min_clip_seconds);
    setValueIfDefined('seedLockSeconds', setup.seed_lock_seconds);
    setValueIfDefined('seedIouMin', setup.seed_iou_min);
    setValueIfDefined('seedDistMax', setup.seed_dist_max);
    setValueIfDefined('seedBonus', setup.seed_bonus);
    setValueIfDefined('seedWindowS', setup.seed_window_s);
    setValueIfDefined('ocrEveryNFrames', setup.ocr_every_n_frames);
    setValueIfDefined('ocrVetoConf', setup.ocr_veto_conf);
    setValueIfDefined('ocrVetoSeconds', setup.ocr_veto_seconds);
    setValueIfDefined('yoloImgsz', setup.yolo_imgsz);
    setValueIfDefined('yoloBatch', setup.yolo_batch);
    setValueIfDefined('trackerType', setup.tracker_type);
    setCheckedIfDefined('reidEnable', setup.reid_enable);
    setValueIfDefined('reidModel', setup.reid_model);
    setValueIfDefined('reidEveryNFrames', setup.reid_every_n_frames);
    setValueIfDefined('reidWeight', setup.reid_weight);
    setValueIfDefined('reidMinSim', setup.reid_min_sim);
    setValueIfDefined('reidCropExpand', setup.reid_crop_expand);
    setValueIfDefined('reidBatch', setup.reid_batch);
    setValueIfDefined('reidDevice', setup.reid_device);
    setValueIfDefined('swapGuardSeconds', setup.swap_guard_seconds);
    setValueIfDefined('swapGuardBonus', setup.swap_guard_bonus);
    setValueIfDefined('benchZone', setup.bench_zone_ratio);
    setCheckedIfDefined('allowUnconfirmedClips', setup.allow_unconfirmed_clips);
    setCheckedIfDefined('allowSeedClips', setup.allow_seed_clips);
    setCheckedIfDefined('ocrDisable', setup.ocr_disable);
    setCheckedIfDefined('debugOverlay', setup.debug_overlay);
    setCheckedIfDefined('debugTimeline', setup.debug_timeline);
    setCheckedIfDefined('transcodeEnabled', setup.transcode_enabled);
    setValueIfDefined('transcodeScaleMax', setup.transcode_scale_max);
    setValueIfDefined('transcodeFps', setup.transcode_fps);
    setCheckedIfDefined('transcodeDeinterlace', setup.transcode_deinterlace);
    setCheckedIfDefined('transcodeDenoise', setup.transcode_denoise);
    state.clicks = Array.isArray(setup.clicks) ? setup.clicks : [];
    setCheckedIfDefined('skipSeeding', setup.skip_seeding);
    drawClickMarkers();
  } catch (_) {
    state.clicks = [];
    drawClickMarkers();
  }
  refreshHelp();
  updateRunButtonState();
}

function renderResults(res){
  const msg = $('resultsMessage');
  if(res.status === 'verified') msg.textContent = 'Verify completed. No clips/combined generated.';
  else if(res.status === 'done_no_clips') msg.textContent = 'Run completed but no clips were found. See debug overlay/timeline.';
  else if(res.status === 'done_no_shifts') msg.textContent = 'Run completed but no shifts were found. See debug overlay/timeline.';
  else msg.textContent = res.message || 'Done';
  const a = $('artifacts'); a.innerHTML='';
  const art = res.artifacts || {};
  const add = (label, url) => { if(!url) return; const d=document.createElement('div'); d.innerHTML=`<a target='_blank' href='${url}'>${label}</a>`; a.appendChild(d); };
  (art.clips || []).forEach((c, i) => add(`clip ${i+1}`, c.url));
  add('combined mp4', art.combined_url || res.combined_url);
  add('debug overlay', art.debug_overlay_url);
  add('debug timeline', art.debug_timeline_url);
}

async function pollOnce(){
  if(!state.jobId) return null;
  const s = await j('GET', `/jobs/${state.jobId}/status`);
  state.lastStatus = s;
  $('out').textContent = JSON.stringify(s, null, 2);
  if(!state.uploading && s.proxy_ready && s.proxy_url && state.proxySrc !== s.proxy_url){
    const vid = $('vid');
    const previousTime = Number(vid.currentTime || 0);
    const wasPaused = vid.paused;
    state.proxySrc = s.proxy_url;
    vid.src = state.proxySrc;
    vid.addEventListener('loadedmetadata', () => {
      if (Number.isFinite(previousTime) && previousTime > 0) {
        try { vid.currentTime = Math.min(previousTime, Math.max(0, (vid.duration || previousTime) - 0.1)); } catch (_) {}
      }
      if (!wasPaused) {
        vid.play().catch(() => {});
      }
    }, { once: true });
  }
  if(!state.uploading){
    updateProgressUi(s);
  }

  if(['done','failed','cancelled','verified','done_no_clips','done_no_shifts'].includes(s.status)){
    clearInterval(state.pollTimer);
    state.pollTimer = null;
    try {
      const res = await j('GET', `/jobs/${state.jobId}/results`);
      $('clips').textContent = JSON.stringify(res, null, 2);
      renderResults(res);
    } catch (_) {}
  }
  return s;
}

function startPolling(){
  if(state.pollTimer) clearInterval(state.pollTimer);
  pollOnce();
  state.pollTimer = setInterval(pollOnce, 1200);
}

window.addEventListener('DOMContentLoaded', async () => {
  $('btnCreate').onclick = createJob;
  $('btnUpload').onclick = upload;
  $('btnSave').onclick = save;
  $('btnRun').onclick = run;
  $('btnCancel').onclick = cancel;
  $('btnClear').onclick = clearJob;
  $('btnClearClicks').onclick = clearSeedClicks;
  $('btnUndoClick').onclick = undoSeedClick;
  $('presetBalanced').onclick = () => applyAdvancedPreset('balanced');
  $('presetMoreClips').onclick = () => applyAdvancedPreset('more_clips');
  $('presetTrackQuality').onclick = () => applyAdvancedPreset('track_quality');
  $('cameraMode').onchange = applyPreset;
  $('trackingMode').onchange = refreshHelp;
  $('verifyMode').onchange = refreshHelp;
  $('skipSeeding').onchange = updateRunButtonState;
  $('vid').addEventListener('click', registerSeedClick);
  $('vid').addEventListener('loadedmetadata', drawClickMarkers);
  $('vid').addEventListener('seeked', drawClickMarkers);
  window.addEventListener('resize', drawClickMarkers);
  applyPreset();

  const existingJobId = localStorage.getItem('shiftclipper.jobId');
  if(existingJobId){
    setJobId(existingJobId);
    await loadSetup();
    try {
      const s = await pollOnce();
      if (s && !['done','failed','cancelled','verified','done_no_clips','done_no_shifts'].includes(s.status)) startPolling();
    } catch (_) {
      localStorage.removeItem('shiftclipper.jobId');
      setJobId(null);
    }
  }
});
