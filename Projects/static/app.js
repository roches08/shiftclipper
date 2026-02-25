function $(id) {
  return document.getElementById(id)
    || document.getElementById(id.replaceAll('_', '-'))
    || document.getElementById(id.replaceAll('-', '_'));
}

function getChecked(id, fallback = false) {
  const el = $(id);
  if (!el) return fallback;
  return !!el.checked;
}

function getValue(id, fallback = '') {
  const el = $(id);
  if (!el) return fallback;
  return (el.value ?? fallback);
}

function getNumber(id, fallback = 0) {
  const v = getValue(id, '');
  const n = parseFloat(v);
  return Number.isFinite(n) ? n : fallback;
}

const readChecked = getChecked;
const readValue = getValue;
const readNumber = getNumber;
const state = {
  jobId: null,
  clicks: [],
  maskEditorActive: false,
  drawMode: null,
  drawPointer: null,
  maskPolygons: { rink_polygon: [], bench_polygons: [[], []], penalty_polygons: [[], []] },
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

const { getVideoTypePreset } = window.ShiftClipperPresets;

const REID_WEIGHTS_DEFAULT_PATH = '/workspace/shiftclipper/Projects/models/reid/osnet_x0_25_msmt17.pth';
const REID_WEIGHTS_DEFAULT_URL = 'https://huggingface.co/kaiyangzhou/osnet/resolve/main/osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth';
const SETUP_STORAGE_KEY = 'shiftclipper_setup';
const SETUP_STORAGE_PREFIX = 'setup:';
const MASK_STORAGE_KEY = 'shiftclipper_mask_polygons';
const DEFAULT_SETUP = {
  video_type: 'wide_single_cam_working_v1',
  score_lock_threshold: 0.45,
  score_unlock_threshold: 0.25,
  clip_continue_threshold: 0.25,
  lock_threshold_normal: 0.45,
  lock_threshold_reacquire: 0.30,
  lock_threshold_seed: 0.30,
  lost_timeout: 18,
  locked_grace_seconds: 2.0,
  reacquire_window_seconds: 90,
  reacquire_score_lock_threshold: 0.30,
  reacquire_max_sec: 25,
  reacquire_confirm_frames: 5,
  loss_timeout_sec: 2,
  gap_merge_seconds: 12,
  lock_seconds_after_confirm: 4,
  min_track_seconds: 0.85,
  min_clip_seconds: 1.0,
  extend_sec: 1.5,
  allow_bench_reacquire: false,
  reid_enable: true,
  reid_every_n_frames: 3,
  reid_weight: 0.65,
  reid_min_sim: 0.45,
  reid_crop_expand: 0.20,
  reid_batch: 16,
  reid_device: 'cuda:0',
  allow_seed_clips: false,
  export_seed_clips: false,
  seed_lock_seconds: 12,
  seed_iou_min: 0.18,
  seed_dist_max: 0.16,
  seed_bonus: 0.8,
  seed_window_s: 45,
  max_clip_len_sec: 0,
  cold_lock_mode: 'allow',
  cold_lock_reid_min_similarity: 0.5,
  cold_lock_margin_min: 0.08,
  cold_lock_max_seconds: 3,
  preset_name: 'Wide Single Cam — Working (Test 2 profile)',
  preset_version: 'v1',
  use_rink_mask: true,
  use_bench_mask: true,
  use_penalty_mask: false,
  rink_polygon: [],
  bench_polygons: [[], []],
  penalty_polygons: [[], []],
};

const HOCKEY_TUNE_PRESET = {
  stability: 0.47,
  clipAlive: 0.23,
  reacquireStrength: 0.30,
  reacquireWindow: 110,
  lostTimeout: 20,
  gapMerge: 10,
  reidStrength: 0.70,
  swapGuard: 12,
};

const HOCKEY_QUICK_TUNE_MAP = {
  qtStability: ['scoreLockThreshold', 'lockThresholdNormal'],
  qtClipAlive: ['scoreUnlockThreshold', 'clipContinueThreshold'],
  qtReacquireStrength: ['lockThresholdReacquire', 'reacquireScoreLockThreshold'],
  qtReacquireWindow: ['reacquireWindowSeconds'],
  qtLostTimeout: ['lostTimeout'],
  qtGapMerge: ['mergeGap'],
};

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

// Robust job id getter: if state gets out of sync (refresh / hot reload),
// recover the active job id so buttons don't appear "dead".
function currentJobId(){
  if(state.jobId) return state.jobId;
  try{
    const ls = localStorage.getItem('shiftclipper.jobId');
    if(ls) return ls;
  }catch{}
  const shown = ($('jobId')?.textContent || '').trim();
  if(shown && shown !== '—' && shown !== '-') return shown;
  return null;
}

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
  const skipSeeding = readChecked('skipSeeding', false);
  const hasSeed = state.clicks.length > 0;
  $('btnRun').disabled = !currentJobId() || (!hasSeed && !skipSeeding);
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

function updatePresetLabel(name, version){
  const el = $('presetLabel');
  el.textContent = `Preset: ${name} (${version})`;
  el.dataset.presetName = name;
  el.dataset.presetVersion = version;
}

function applyVideoTypePreset(videoType){
  const preset = getVideoTypePreset(videoType);
  const p = preset.values;
  const val = (k, fallback) => (p[k] !== undefined ? p[k] : fallback);
  $('videoType').value = videoType;
  $('scoreLockThreshold').value = val('score_lock_threshold', $('scoreLockThreshold').value);
  $('scoreUnlockThreshold').value = val('score_unlock_threshold', $('scoreUnlockThreshold').value);
  $('clipContinueThreshold').value = val('clip_continue_threshold', $('clipContinueThreshold').value);
  $('lockThresholdNormal').value = val('lock_threshold_normal', $('lockThresholdNormal').value);
  $('lockThresholdReacquire').value = val('lock_threshold_reacquire', $('lockThresholdReacquire').value);
  $('lockThresholdSeed').value = val('lock_threshold_seed', $('lockThresholdSeed').value);
  $('lostTimeout').value = val('lost_timeout', $('lostTimeout').value);
  $('reacquireWindowSeconds').value = val('reacquire_window_seconds', val('reacquire_window_s', $('reacquireWindowSeconds').value));
  $('reacquireScoreLockThreshold').value = val('reacquire_score_lock_threshold', $('reacquireScoreLockThreshold').value);
  $('mergeGap').value = val('gap_merge_seconds', $('mergeGap').value);
  $('extendSec').value = val('extend_sec', $('extendSec').value);
  setCheckedIfPresent('allowBenchReacquire',
    !!val('allow_bench_reacquire', getChecked('allowBenchReacquire', false))
  );
  setCheckedIfPresent('allowSeedClips',
    !!val('export_seed_clips', val('allow_seed_clips', getChecked('allowSeedClips', false)))
  );
  $('seedLockSeconds').value = val('seed_lock_seconds', $('seedLockSeconds').value);
  $('seedIouMin').value = val('seed_iou_min', $('seedIouMin').value);
  $('seedDistMax').value = val('seed_dist_max', $('seedDistMax').value);
  $('seedBonus').value = val('seed_bonus', $('seedBonus').value);
  $('seedWindowS').value = val('seed_window_s', $('seedWindowS').value);
  setCheckedIfPresent('reidEnable',
    !!val('reid_enable', val('reid_enabled', getChecked('reidEnable', true)))
  );
  $('reidModel').value = val('reid_model', $('reidModel').value);
  $('reidEveryNFrames').value = val('reid_every_n_frames', $('reidEveryNFrames').value);
  $('reidWeight').value = val('reid_weight', $('reidWeight').value);
  $('reidMinSim').value = val('reid_min_sim', val('reid_min_similarity', $('reidMinSim').value));
  $('reidCropExpand').value = val('reid_crop_expand', $('reidCropExpand').value);
  $('reidBatch').value = val('reid_batch', $('reidBatch').value);
  $('reidDevice').value = val('reid_device', $('reidDevice').value);
  $('reidMinPx').value = val('reid_min_px', $('reidMinPx').value);
  $('reidSharpnessThreshold').value = val('reid_sharpness_threshold', $('reidSharpnessThreshold').value);
  $('swapGuardSeconds').value = val('swap_guard_seconds', $('swapGuardSeconds').value);
  $('swapGuardBonus').value = val('swap_guard_bonus', $('swapGuardBonus').value);
  $('coldLockMode').value = val('cold_lock_mode', $('coldLockMode').value);
  $('coldLockReidMinSimilarity').value = val('cold_lock_reid_min_similarity', $('coldLockReidMinSimilarity').value);
  $('coldLockMarginMin').value = val('cold_lock_margin_min', $('coldLockMarginMin').value);
  $('coldLockMaxSeconds').value = val('cold_lock_max_seconds', $('coldLockMaxSeconds').value);
  $('maxClipLenSec').value = val('max_clip_len_sec', $('maxClipLenSec').value);
  setValueIfDefined('postRoll', p.post_roll);
  setValueIfDefined('colorTolerance', p.color_tolerance);
  setValueIfDefined('yoloBatch', p.yolo_batch);
  setCheckedIfDefined('ocrDisable', p.ocr_disable);
  setValueIfDefined('ocrEveryNFrames', p.ocr_every_n_frames);
  setValueIfDefined('ocrVetoConf', p.ocr_veto_conf);
  setValueIfDefined('ocrVetoSeconds', p.ocr_veto_seconds);
  setValueIfDefined('lockedGraceSeconds', p.locked_grace_seconds);
  setValueIfDefined('reacquireMaxSec', p.reacquire_max_sec);
  setValueIfDefined('lossTimeoutSec', p.loss_timeout_sec);
  setCheckedIfDefined('allowUnconfirmedClips', p.allow_unconfirmed_clips);
  setCheckedIfDefined('debugOverlay', p.debug_overlay);
  setCheckedIfDefined('debugTimeline', p.debug_timeline);
  setCheckedIfDefined('generateCombined', p.generate_combined);
  updatePresetLabel(preset.preset_name, preset.preset_version);
  syncQuickTuneFromAdvanced();
}

function getVideoTypeSetupKey(videoType){
  return `${SETUP_STORAGE_PREFIX}${videoType || 'wide_single_cam_working_v1'}`;
}

function clamp01(v){
  return Math.max(0, Math.min(1, Number(v) || 0));
}

function syncQuickTuneFromAdvanced(){
  setValueIfDefined('qtStability', getValue('scoreLockThreshold', HOCKEY_TUNE_PRESET.stability));
  setValueIfDefined('qtClipAlive', getValue('scoreUnlockThreshold', HOCKEY_TUNE_PRESET.clipAlive));
  setValueIfDefined('qtReacquireStrength', getValue('lockThresholdReacquire', HOCKEY_TUNE_PRESET.reacquireStrength));
  setValueIfDefined('qtReacquireWindow', getValue('reacquireWindowSeconds', HOCKEY_TUNE_PRESET.reacquireWindow));
  setValueIfDefined('qtLostTimeout', getValue('lostTimeout', HOCKEY_TUNE_PRESET.lostTimeout));
  setValueIfDefined('qtGapMerge', getValue('mergeGap', HOCKEY_TUNE_PRESET.gapMerge));
  setValueIfDefined('qtReidStrength', getValue('reidWeight', HOCKEY_TUNE_PRESET.reidStrength));
  setValueIfDefined('qtSwapGuard', getValue('swapGuardSeconds', HOCKEY_TUNE_PRESET.swapGuard));
}

function applyQuickTuneToAdvanced(){
  Object.entries(HOCKEY_QUICK_TUNE_MAP).forEach(([quickId, advancedIds]) => {
    const val = getValue(quickId, '');
    advancedIds.forEach((advancedId) => setValueIfDefined(advancedId, val));
  });

  const reidStrength = clamp01(getNumber('qtReidStrength', HOCKEY_TUNE_PRESET.reidStrength));
  setValueIfDefined('reidWeight', reidStrength);
  setValueIfDefined('reidMinSim', clamp01(reidStrength - 0.22));

  const swapGuardSeconds = Math.max(0, getNumber('qtSwapGuard', HOCKEY_TUNE_PRESET.swapGuard));
  setValueIfDefined('swapGuardSeconds', swapGuardSeconds);
  setValueIfDefined('swapGuardBonus', clamp01(swapGuardSeconds / 40));
}

function applyHockeyTunePreset(){
  setValueIfDefined('qtStability', HOCKEY_TUNE_PRESET.stability);
  setValueIfDefined('qtClipAlive', HOCKEY_TUNE_PRESET.clipAlive);
  setValueIfDefined('qtReacquireStrength', HOCKEY_TUNE_PRESET.reacquireStrength);
  setValueIfDefined('qtReacquireWindow', HOCKEY_TUNE_PRESET.reacquireWindow);
  setValueIfDefined('qtLostTimeout', HOCKEY_TUNE_PRESET.lostTimeout);
  setValueIfDefined('qtGapMerge', HOCKEY_TUNE_PRESET.gapMerge);
  setValueIfDefined('qtReidStrength', HOCKEY_TUNE_PRESET.reidStrength);
  setValueIfDefined('qtSwapGuard', HOCKEY_TUNE_PRESET.swapGuard);
  applyQuickTuneToAdvanced();
  persistSetup();
}

function collectSetupFromUI(){
  return payload();
}

function getSetupForStorage(){
  const setup = collectSetupFromUI();
  delete setup.clicks;
  delete setup.clicks_count;
  return setup;
}

function persistSetup(){
  const setup = getSetupForStorage();
  localStorage.setItem(SETUP_STORAGE_KEY, JSON.stringify(setup));
  localStorage.setItem(getVideoTypeSetupKey(setup.video_type), JSON.stringify(setup));
  persistMaskPolygons();
}

function applySetupValues(setup){
  if(!setup) return;
  setValueIfDefined('videoType', setup.video_type);
  setValueIfDefined('scoreLockThreshold', setup.score_lock_threshold);
  setValueIfDefined('scoreUnlockThreshold', setup.score_unlock_threshold);
  setValueIfDefined('clipContinueThreshold', setup.clip_continue_threshold);
  setValueIfDefined('lostTimeout', setup.lost_timeout);
  setValueIfDefined('lockedGraceSeconds', setup.locked_grace_seconds);
  setValueIfDefined('reacquireWindowSeconds', setup.reacquire_window_seconds);
  setValueIfDefined('reacquireScoreLockThreshold', setup.reacquire_score_lock_threshold);
  setValueIfDefined('reacquireMaxSec', setup.reacquire_max_sec);
  setValueIfDefined('lossTimeoutSec', setup.loss_timeout_sec);
  setValueIfDefined('mergeGap', setup.gap_merge_seconds);
  setValueIfDefined('lockSeconds', setup.lock_seconds_after_confirm);
  setValueIfDefined('minTrack', setup.min_track_seconds);
  setValueIfDefined('minClipSeconds', setup.min_clip_seconds);
  setValueIfDefined('extendSec', setup.extend_sec);
  setCheckedIfDefined('allowBenchReacquire', setup.allow_bench_reacquire);
  setCheckedIfDefined('reidEnable', setup.reid_enable);
  setValueIfDefined('reidEveryNFrames', setup.reid_every_n_frames);
  setValueIfDefined('reidWeight', setup.reid_weight);
  setValueIfDefined('reidMinSim', setup.reid_min_sim);
  setValueIfDefined('reidCropExpand', setup.reid_crop_expand);
  setValueIfDefined('reidBatch', setup.reid_batch);
  setValueIfDefined('reidDevice', setup.reid_device);
  setCheckedIfDefined('allowSeedClips', setup.export_seed_clips ?? setup.allow_seed_clips);
  setValueIfDefined('seedLockSeconds', setup.seed_lock_seconds);
  setValueIfDefined('seedIouMin', setup.seed_iou_min);
  setValueIfDefined('seedDistMax', setup.seed_dist_max);
  setValueIfDefined('seedBonus', setup.seed_bonus);
  setValueIfDefined('seedWindowS', setup.seed_window_s);
  setValueIfDefined('maxClipLenSec', setup.max_clip_len_sec);
  setValueIfDefined('lockThresholdNormal', setup.lock_threshold_normal);
  setValueIfDefined('lockThresholdReacquire', setup.lock_threshold_reacquire);
  setValueIfDefined('lockThresholdSeed', setup.lock_threshold_seed);
  setValueIfDefined('coldLockMode', setup.cold_lock_mode);
  setValueIfDefined('coldLockReidMinSimilarity', setup.cold_lock_reid_min_similarity);
  setValueIfDefined('coldLockMarginMin', setup.cold_lock_margin_min);
  setValueIfDefined('coldLockMaxSeconds', setup.cold_lock_max_seconds);
  setCheckedIfDefined('generateCombined', setup.generate_combined);
  setCheckedIfDefined('useRinkMask', setup.use_rink_mask);
  setCheckedIfDefined('useBenchMask', setup.use_bench_mask);
  setCheckedIfDefined('usePenaltyMask', setup.use_penalty_mask);
  syncQuickTuneFromAdvanced();
  state.maskPolygons = {
    rink_polygon: normalizePolygon(setup.rink_polygon || state.maskPolygons.rink_polygon || []),
    bench_polygons: normalizePolygonList(setup.bench_polygons || state.maskPolygons.bench_polygons || []),
    penalty_polygons: normalizePolygonList(setup.penalty_polygons || state.maskPolygons.penalty_polygons || []),
  };
  drawMaskOverlay();
  updateMaskWarning();
}

function loadSavedSetupOrDefaults(){
  const selectedVideoType = $('videoType')?.value || DEFAULT_SETUP.video_type;
  const scopedRaw = localStorage.getItem(getVideoTypeSetupKey(selectedVideoType));
  if (scopedRaw){
    try {
      applySetupValues(JSON.parse(scopedRaw));
      return;
    } catch (_) {
      localStorage.removeItem(getVideoTypeSetupKey(selectedVideoType));
    }
  }
  const raw = localStorage.getItem(SETUP_STORAGE_KEY);
  if(!raw){
    applySetupValues(DEFAULT_SETUP);
    return;
  }
  try {
    applySetupValues(JSON.parse(raw));
  } catch (_) {
    localStorage.removeItem(SETUP_STORAGE_KEY);
    applySetupValues(DEFAULT_SETUP);
  }
}

function resetSettings(){
  localStorage.removeItem(SETUP_STORAGE_KEY);
  applySetupValues(DEFAULT_SETUP);
  refreshHelp();
  updateRunButtonState();
  updateMaskWarning();
}

function applyPreset(){
  const p = CAMERA_DEFAULTS[$('cameraMode').value];
  $('detectStride').value = p.detect_stride;
  $('ocrMinConf').value = p.ocr_min_conf;
  $('lockSeconds').value = p.lock_seconds_after_confirm;
  $('mergeGap').value = p.gap_merge_seconds;
  $('lostTimeout').value = p.lost_timeout;
  $('minTrack').value = p.min_track_seconds;
  applyVideoTypePreset($('videoType').value || 'wide_single_cam_working_v1');
  applySetupValues(DEFAULT_SETUP);
  refreshHelp();
}

function setValueIfDefined(id, value){
  if(value === undefined || value === null) return;
  const el = $(id);
  if(!el) return;
  el.value = value;
}

function setCheckedIfDefined(id, value){
  if(value === undefined || value === null) return;
  const el = $(id);
  if(!el) return;
  el.checked = !!value;
}

function setCheckedIfPresent(id, value){
  const el = $(id);
  if(el) el.checked = !!value;
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
  loadSavedSetupOrDefaults();
  await saveSetupToServer();
  await loadSetup();
  await pollOnce();
}

async function saveSetupToServer(){
  const jid = currentJobId();
  if (!jid) return;
  const response = await fetch(`/jobs/${jid}/setup`, {
    method: 'PUT',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(collectSetupFromUI()),
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
}

async function upload(){
  const jid = currentJobId();
  const f = $('file').files[0];
  if(!f || !jid) return;
  if (state.uploading) return;
  const fd = new FormData(); fd.append('file', f);
  $('progressMessage').textContent = 'Uploading...';
  state.uploading = true;

  await new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    state.uploadXhr = xhr;
    xhr.open('POST', `/jobs/${jid}/upload`, true);
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


function normalizePoint(point){
  if (!point || typeof point !== 'object') return null;
  const x = Number(point.x);
  const y = Number(point.y);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
  return { x: Math.max(0, Math.min(1, x)), y: Math.max(0, Math.min(1, y)) };
}

function normalizePolygon(poly){
  if (!Array.isArray(poly)) return [];
  return poly.map(normalizePoint).filter(Boolean);
}

function normalizePolygonList(polys){
  if (!Array.isArray(polys)) return [[], []];
  const out = polys.map(normalizePolygon);
  if (!out.length) return [[], []];
  while(out.length < 2) out.push([]);
  return out.slice(0, 2);
}

function loadMaskPolygonsFromStorage(){
  try {
    const raw = localStorage.getItem(MASK_STORAGE_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    state.maskPolygons = {
      rink_polygon: normalizePolygon(parsed.rink_polygon || []),
      bench_polygons: normalizePolygonList(parsed.bench_polygons || []),
      penalty_polygons: normalizePolygonList(parsed.penalty_polygons || []),
    };
  } catch (_) {}
}

function persistMaskPolygons(){
  localStorage.setItem(MASK_STORAGE_KEY, JSON.stringify(state.maskPolygons));
}

function getMaskTargetRef(){
  const target = getValue('maskTarget', 'rink');
  if (target === 'rink') return { key: 'rink_polygon', index: null };
  const [kind, idx] = target.split('_');
  const index = Number(idx);
  if (!Number.isInteger(index)) return { key: 'rink_polygon', index: null };
  return { key: `${kind}_polygons`, index };
}

function getMaskPolygonByTarget(){
  const ref = getMaskTargetRef();
  if (ref.index === null) return state.maskPolygons.rink_polygon;
  const arr = state.maskPolygons[ref.key] || [];
  while(arr.length <= ref.index) arr.push([]);
  return arr[ref.index];
}

function drawMaskOverlay(){
  const vid = $('vid');
  const canvas = $('maskCanvas');
  if (!vid || !canvas) return;
  const rect = vid.getBoundingClientRect();
  const vw = Math.max(1, rect.width);
  const vh = Math.max(1, rect.height);
  canvas.width = Math.round(vw);
  canvas.height = Math.round(vh);
  canvas.style.width = `${vw}px`;
  canvas.style.height = `${vh}px`;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const drawPoly = (poly, color) => {
    if (!Array.isArray(poly) || !poly.length) return;
    ctx.strokeStyle = color;
    ctx.fillStyle = color.replace('1)', '0.2)');
    ctx.lineWidth = 2;
    ctx.beginPath();
    poly.forEach((pt, idx) => {
      const x = pt.x * canvas.width;
      const y = pt.y * canvas.height;
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    if (poly.length > 2) ctx.closePath();
    ctx.stroke();
    if (poly.length > 2) ctx.fill();
    poly.forEach((pt) => {
      ctx.beginPath();
      ctx.arc(pt.x * canvas.width, pt.y * canvas.height, 4, 0, Math.PI * 2);
      ctx.fill();
    });
  };

  drawPoly(state.maskPolygons.rink_polygon, 'rgba(32,125,255,1)');
  (state.maskPolygons.bench_polygons || []).forEach((poly) => drawPoly(poly, 'rgba(255,167,38,1)'));
  (state.maskPolygons.penalty_polygons || []).forEach((poly) => drawPoly(poly, 'rgba(233,30,99,1)'));

  if (state.drawMode) {
    const activePoly = state.drawMode === 'rink'
      ? state.maskPolygons.rink_polygon
      : (state.maskPolygons.bench_polygons?.[0] || []);
    if (activePoly.length && state.drawPointer) {
      const color = state.drawMode === 'rink' ? 'rgba(32,125,255,0.85)' : 'rgba(255,167,38,0.85)';
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      const last = activePoly[activePoly.length - 1];
      ctx.moveTo(last.x * canvas.width, last.y * canvas.height);
      ctx.lineTo(state.drawPointer.x * canvas.width, state.drawPointer.y * canvas.height);
      ctx.stroke();
    }
  }

  if (state.maskEditorActive || state.drawMode) {
    canvas.style.display = 'block';
    canvas.style.pointerEvents = state.drawMode ? 'auto' : 'none';
  } else {
    canvas.style.display = 'none';
    canvas.style.pointerEvents = 'none';
  }
}

function getMaskWarnings(setup){
  const warnings = [];
  if (setup.use_rink_mask && !setup.rink_polygon.length) warnings.push('use_rink_mask enabled but rink polygon is empty.');
  if (setup.use_bench_mask && !(setup.bench_polygons || []).some((p) => p.length >= 3)) warnings.push('use_bench_mask enabled but bench polygons are empty.');
  if (setup.use_penalty_mask && !(setup.penalty_polygons || []).some((p) => p.length >= 3)) warnings.push('use_penalty_mask enabled but penalty polygons are empty.');
  return warnings;
}

function updateMaskWarning(){
  const warningEl = $('maskWarning');
  if (!warningEl) return;
  const warnings = getMaskWarnings({
    use_rink_mask: getChecked('useRinkMask', true),
    use_bench_mask: getChecked('useBenchMask', true),
    use_penalty_mask: getChecked('usePenaltyMask', false),
    rink_polygon: state.maskPolygons.rink_polygon || [],
    bench_polygons: state.maskPolygons.bench_polygons || [],
    penalty_polygons: state.maskPolygons.penalty_polygons || [],
  });
  warningEl.textContent = warnings.join(' ');
  warningEl.style.display = warnings.length ? 'block' : 'none';
}

function toggleMaskEditor(){
  state.maskEditorActive = !state.maskEditorActive;
  if (!state.maskEditorActive) stopDrawingMode();
  const btn = $('btnEditPolygons');
  if (btn) btn.textContent = state.maskEditorActive ? 'Stop Editing' : 'Edit Polygons';
  setMaskDrawButtons();
  drawMaskOverlay();
}

function setMaskDrawButtons(){
  const rinkBtn = $('btnDrawRink');
  const benchBtn = $('btnDrawBench');
  if (rinkBtn) rinkBtn.disabled = !state.maskEditorActive;
  if (benchBtn) benchBtn.disabled = !state.maskEditorActive;
}

function stopDrawingMode(){
  state.drawMode = null;
  state.drawPointer = null;
  setMaskDrawButtons();
  drawMaskOverlay();
}

function startDrawingMode(mode){
  if (!state.maskEditorActive) state.maskEditorActive = true;
  state.drawMode = mode;
  state.drawPointer = null;
  if (mode === 'rink') $('maskTarget').value = 'rink';
  if (mode === 'bench') $('maskTarget').value = 'bench_0';
  const btn = $('btnEditPolygons');
  if (btn) btn.textContent = 'Stop Editing';
  setMaskDrawButtons();
  drawMaskOverlay();
}

function clearSelectedPolygon(){
  const ref = getMaskTargetRef();
  if (ref.index === null) state.maskPolygons.rink_polygon = [];
  else {
    const arr = state.maskPolygons[ref.key] || [];
    while(arr.length <= ref.index) arr.push([]);
    arr[ref.index] = [];
    state.maskPolygons[ref.key] = arr;
  }
  drawMaskOverlay();
  updateMaskWarning();
}

function savePolygons(){
  persistMaskPolygons();
  persistSetup();
  updateMaskWarning();
}

function normalizedPointFromCanvasEvent(evt){
  const canvas = $('maskCanvas');
  if (!canvas) return null;
  const rect = canvas.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return null;
  const nx = (evt.clientX - rect.left) / rect.width;
  const ny = (evt.clientY - rect.top) / rect.height;
  return {
    x: Math.max(0, Math.min(1, nx)),
    y: Math.max(0, Math.min(1, ny)),
  };
}

function registerMaskPoint(evt){
  if (!state.drawMode) return;
  evt.preventDefault();
  const point = normalizedPointFromCanvasEvent(evt);
  if (!point) return;
  const poly = getMaskPolygonByTarget();
  if (!poly) return;
  if (evt.button === 2) poly.pop();
  else poly.push(point);
  state.drawPointer = point;
  drawMaskOverlay();
  updateMaskWarning();
}

function updateMaskPointer(evt){
  if (!state.drawMode) return;
  const point = normalizedPointFromCanvasEvent(evt);
  if (!point) return;
  state.drawPointer = point;
  drawMaskOverlay();
}

function endMaskPointer(evt){
  if (!state.drawMode) return;
  if (evt) evt.preventDefault();
}



function payload(){
  const toNumber = (id, fallback = 0) => getNumber(id, fallback);
  const toInt = (id, fallback = 0) => Math.trunc(getNumber(id, fallback));
  const jerseyColor = getValue('jerseyColor', '#000000');
  const jerseyColorHex = /^#[0-9a-fA-F]{6}$/.test(jerseyColor) ? jerseyColor : '#000000';
  const jerseyRgbHex = jerseyColorHex.replace('#', '');
  return {
    video_type: getValue('videoType', DEFAULT_SETUP.video_type),
    camera_mode: getValue('cameraMode', 'broadcast_wide'),
    tracking_mode: getValue('trackingMode', 'clip'),
    verify_mode: getValue('verifyMode', 'off') === 'on',
    skip_seeding: getChecked('skipSeeding', false),
    player_number: getValue('playerNumber', ''),
    jersey_color: jerseyColorHex,
    jersey_color_hex: jerseyColorHex,
    opponent_color: getValue('opponentColor', '#000000'),
    jersey_color_rgb: (() => ({ r: parseInt(jerseyRgbHex.slice(0,2),16), g: parseInt(jerseyRgbHex.slice(2,4),16), b: parseInt(jerseyRgbHex.slice(4,6),16) }))(),
    color_tolerance: toInt('colorTolerance'),
    extend_sec: toNumber('extendSec'),
    detect_stride: toInt('detectStride'),
    yolo_imgsz: toInt('yoloImgsz'),
    yolo_batch: toInt('yoloBatch'),
    tracker_type: getValue('trackerType', 'bytetrack') || 'bytetrack',
    ocr_min_conf: toNumber('ocrMinConf'),
    ocr_disable: getChecked('ocrDisable', false),
    ocr_every_n_frames: toInt('ocrEveryNFrames'),
    ocr_veto_conf: toNumber('ocrVetoConf'),
    ocr_veto_seconds: toNumber('ocrVetoSeconds'),
    lock_seconds_after_confirm: toNumber('lockSeconds'),
    lost_timeout: toNumber('lostTimeout', DEFAULT_SETUP.lost_timeout),
    gap_merge_seconds: toNumber('mergeGap', DEFAULT_SETUP.gap_merge_seconds),
    score_lock_threshold: toNumber('scoreLockThreshold', DEFAULT_SETUP.score_lock_threshold),
    score_unlock_threshold: toNumber('scoreUnlockThreshold', DEFAULT_SETUP.score_unlock_threshold),
    clip_continue_threshold: toNumber('clipContinueThreshold', DEFAULT_SETUP.clip_continue_threshold),
    reacquire_window_seconds: toNumber('reacquireWindowSeconds', DEFAULT_SETUP.reacquire_window_seconds),
    reacquire_score_lock_threshold: toNumber('reacquireScoreLockThreshold'),
    lock_threshold_normal: toNumber('lockThresholdNormal'),
    lock_threshold_reacquire: toNumber('lockThresholdReacquire', DEFAULT_SETUP.lock_threshold_reacquire),
    lock_threshold_seed: toNumber('lockThresholdSeed', DEFAULT_SETUP.lock_threshold_seed),
    locked_grace_seconds: toNumber('lockedGraceSeconds', DEFAULT_SETUP.locked_grace_seconds),
    reacquire_max_sec: toNumber('reacquireMaxSec', DEFAULT_SETUP.reacquire_max_sec),
    reacquire_confirm_frames: toInt('reacquireConfirmFrames', DEFAULT_SETUP.reacquire_confirm_frames),
    loss_timeout_sec: toNumber('lossTimeoutSec'),
    allow_bench_reacquire: getChecked('allowBenchReacquire', false),
    use_rink_mask: getChecked('useRinkMask', true),
    use_bench_mask: getChecked('useBenchMask', true),
    use_penalty_mask: getChecked('usePenaltyMask', false),
    polygon_coords_normalized: true,
    rink_polygon: normalizePolygon(state.maskPolygons.rink_polygon),
    bench_polygons: normalizePolygonList(state.maskPolygons.bench_polygons),
    penalty_polygons: normalizePolygonList(state.maskPolygons.penalty_polygons),
    allow_unconfirmed_clips: getChecked('allowUnconfirmedClips', false),
    allow_seed_clips: getChecked('allowSeedClips', false),
    export_seed_clips: getChecked('allowSeedClips', false),
    min_track_seconds: toNumber('minTrack'),
    min_clip_seconds: toNumber('minClipSeconds'),
    seed_lock_seconds: toNumber('seedLockSeconds', DEFAULT_SETUP.seed_lock_seconds),
    seed_iou_min: toNumber('seedIouMin'),
    seed_dist_max: toNumber('seedDistMax'),
    seed_bonus: toNumber('seedBonus'),
    seed_window_s: toNumber('seedWindowS', DEFAULT_SETUP.seed_window_s),
    max_clip_len_sec: toNumber('maxClipLenSec'),
    cold_lock_mode: getValue('coldLockMode', DEFAULT_SETUP.cold_lock_mode),
    cold_lock_reid_min_similarity: toNumber('coldLockReidMinSimilarity'),
    cold_lock_margin_min: toNumber('coldLockMarginMin'),
    cold_lock_max_seconds: toNumber('coldLockMaxSeconds'),
    clicks_count: state.clicks.length,
    bench_zone_ratio: toNumber('benchZone'),
    debug_overlay: getChecked('debugOverlay', false),
    debug_timeline: getChecked('debugTimeline', true),
    generate_combined: getChecked('generateCombined', false),
    transcode_enabled: getChecked('transcodeEnabled', false),
    transcode_scale_max: toInt('transcodeScaleMax'),
    transcode_fps: getValue('transcodeFps', '') ? toInt('transcodeFps') : null,
    transcode_deinterlace: getChecked('transcodeDeinterlace', true),
    transcode_denoise: getChecked('transcodeDenoise', false),
    reid_enable: getChecked('reidEnable', true),
    reid_model: getValue('reidModel', ''),
    reid_every_n_frames: toInt('reidEveryNFrames'),
    reid_weight: toNumber('reidWeight'),
    reid_min_sim: toNumber('reidMinSim'),
    reid_crop_expand: toNumber('reidCropExpand'),
    reid_min_px: toInt('reidMinPx'),
    reid_sharpness_threshold: toNumber('reidSharpnessThreshold'),
    reid_batch: toInt('reidBatch'),
    reid_device: getValue('reidDevice', 'cuda:0'),
    reid_fail_policy: 'disable',
    reid_auto_download: true,
    reid_weights_path: REID_WEIGHTS_DEFAULT_PATH,
    reid_weights_url: REID_WEIGHTS_DEFAULT_URL,
    swap_guard_seconds: toNumber('swapGuardSeconds'),
    swap_guard_bonus: toNumber('swapGuardBonus'),
    preset_name: $('presetLabel')?.dataset.presetName || 'Wide Single Cam — Working (Test 2 profile)',
    preset_version: $('presetLabel')?.dataset.presetVersion || 'v1',
    clicks: state.clicks,
  };
}

async function save(){
  const jid = currentJobId();
  if(!jid) return;
  persistSetup();
  await j('PUT', `/jobs/${jid}/setup`, payload());
  await pollOnce();
}

async function run(){
  const jid = currentJobId();
  if(!jid) return;
  if(readValue('verifyMode', 'off') === 'on'){
    const ok = confirm('Verify mode will not create clips/combined video. Continue?\nCancel = Turn off verify + run');
    if(!ok){ $('verifyMode').value = 'off'; refreshHelp(); }
  }
  if (state.clicks.length < 1 && !readChecked('skipSeeding', false)){
    alert('Add at least one seed click or check Skip seeding before running.');
    return;
  }
  await save();
  await j('POST', `/jobs/${jid}/run`);
  startPolling();
}

async function cancel(){
  const jid = currentJobId();
  if(!jid) return;
  await j('POST', `/jobs/${jid}/cancel`);
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
  const jid = currentJobId();
  if(!jid) return;
  await j('DELETE', `/jobs/${jid}`);
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
  if (state.drawMode) return;
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
  const jid = currentJobId();
  if (!jid) return;
  try {
    const resp = await j('GET', `/jobs/${jid}/setup`);
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
  setValueIfDefined('clipContinueThreshold', setup.clip_continue_threshold);
    setValueIfDefined('reacquireWindowSeconds', setup.reacquire_window_seconds);
    setValueIfDefined('reacquireScoreLockThreshold', setup.reacquire_score_lock_threshold);
    setValueIfDefined('lockedGraceSeconds', setup.locked_grace_seconds);
    setValueIfDefined('reacquireMaxSec', setup.reacquire_max_sec);
    setValueIfDefined('lossTimeoutSec', setup.loss_timeout_sec);
    setValueIfDefined('minTrack', setup.min_track_seconds);
    setValueIfDefined('minClipSeconds', setup.min_clip_seconds);
    setValueIfDefined('seedLockSeconds', setup.seed_lock_seconds);
    setValueIfDefined('seedIouMin', setup.seed_iou_min);
    setValueIfDefined('seedDistMax', setup.seed_dist_max);
    setValueIfDefined('seedBonus', setup.seed_bonus);
    setValueIfDefined('seedWindowS', setup.seed_window_s);
    setValueIfDefined('maxClipLenSec', setup.max_clip_len_sec);
    setValueIfDefined('lockThresholdNormal', setup.lock_threshold_normal);
    setValueIfDefined('lockThresholdReacquire', setup.lock_threshold_reacquire);
    setValueIfDefined('lockThresholdSeed', setup.lock_threshold_seed);
    setValueIfDefined('coldLockMode', setup.cold_lock_mode);
    setValueIfDefined('coldLockReidMinSimilarity', setup.cold_lock_reid_min_similarity);
    setValueIfDefined('coldLockMarginMin', setup.cold_lock_margin_min);
    setValueIfDefined('coldLockMaxSeconds', setup.cold_lock_max_seconds);
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
    setValueIfDefined('reidMinPx', setup.reid_min_px);
    setValueIfDefined('reidSharpnessThreshold', setup.reid_sharpness_threshold);
    setValueIfDefined('reidBatch', setup.reid_batch);
    setValueIfDefined('reidDevice', setup.reid_device);
    setValueIfDefined('swapGuardSeconds', setup.swap_guard_seconds);
    setValueIfDefined('swapGuardBonus', setup.swap_guard_bonus);
    if (setup.preset_name && setup.preset_version) updatePresetLabel(setup.preset_name, setup.preset_version);
    setValueIfDefined('benchZone', setup.bench_zone_ratio);
    setCheckedIfDefined('allowUnconfirmedClips', setup.allow_unconfirmed_clips);
    setCheckedIfDefined('allowSeedClips', setup.export_seed_clips ?? setup.allow_seed_clips);
    setCheckedIfDefined('allowBenchReacquire', setup.allow_bench_reacquire);
    setCheckedIfDefined('ocrDisable', setup.ocr_disable);
    setCheckedIfDefined('debugOverlay', setup.debug_overlay);
    setCheckedIfDefined('debugTimeline', setup.debug_timeline);
    setCheckedIfDefined('generateCombined', setup.generate_combined);
    setCheckedIfDefined('transcodeEnabled', setup.transcode_enabled);
    setValueIfDefined('transcodeScaleMax', setup.transcode_scale_max);
    setValueIfDefined('transcodeFps', setup.transcode_fps);
    setCheckedIfDefined('transcodeDeinterlace', setup.transcode_deinterlace);
    setCheckedIfDefined('transcodeDenoise', setup.transcode_denoise);
    setCheckedIfDefined('useRinkMask', setup.use_rink_mask);
    setCheckedIfDefined('useBenchMask', setup.use_bench_mask);
    setCheckedIfDefined('usePenaltyMask', setup.use_penalty_mask);
  syncQuickTuneFromAdvanced();
    const rawSetup = setup.config_ui_raw || {};
    state.maskPolygons = {
      rink_polygon: normalizePolygon(setup.rink_polygon || rawSetup.rink_polygon || state.maskPolygons.rink_polygon || []),
      bench_polygons: normalizePolygonList(setup.bench_polygons || rawSetup.bench_polygons || state.maskPolygons.bench_polygons || []),
      penalty_polygons: normalizePolygonList(setup.penalty_polygons || rawSetup.penalty_polygons || state.maskPolygons.penalty_polygons || []),
    };
    drawMaskOverlay();
    updateMaskWarning();
    state.clicks = Array.isArray(setup.clicks) ? setup.clicks : [];
    setCheckedIfDefined('skipSeeding', setup.skip_seeding);
    drawClickMarkers();
  } catch (_) {
    state.clicks = [];
    drawClickMarkers();
  }
  syncQuickTuneFromAdvanced();
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
  const jid = currentJobId();
  if(!jid) return null;
  const s = await j('GET', `/jobs/${jid}/status`);
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
      const res = await j('GET', `/jobs/${jid}/results`);
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
  const bindClick = (id, handler) => {
    const el = $(id);
    if (el) el.onclick = handler;
  };
  const bindChange = (id, handler) => {
    const el = $(id);
    if (el) el.onchange = handler;
  };

  bindClick('btnCreate', createJob);
  bindClick('btnUpload', upload);
  bindClick('btnSave', save);
  bindClick('btnRun', run);
  bindClick('btnCancel', cancel);
  bindClick('btnClear', clearJob);
  bindClick('btnResetSettings', resetSettings);
  bindClick('btnClearClicks', clearSeedClicks);
  bindClick('btnUndoClick', undoSeedClick);
  bindClick('btnEditPolygons', toggleMaskEditor);
  bindClick('btnDrawRink', () => startDrawingMode('rink'));
  bindClick('btnDrawBench', () => startDrawingMode('bench'));
  bindClick('btnStopDrawing', stopDrawingMode);
  bindClick('btnClearPolygon', clearSelectedPolygon);
  bindClick('btnSavePolygons', savePolygons);
  bindClick('btnApplyHockeyTune', applyHockeyTunePreset);

  bindChange('cameraMode', applyPreset);
  bindChange('videoType', () => {
    if (!confirm('This will overwrite Advanced settings.')) { return; }
    applyVideoTypePreset(readValue('videoType', DEFAULT_SETUP.video_type));
    loadSavedSetupOrDefaults();
    persistSetup();
  });
  bindChange('trackingMode', refreshHelp);
  bindChange('verifyMode', refreshHelp);
  bindChange('skipSeeding', updateRunButtonState);
  bindChange('maskTarget', drawMaskOverlay);
  bindChange('useRinkMask', () => { persistSetup(); updateMaskWarning(); });
  bindChange('useBenchMask', () => { persistSetup(); updateMaskWarning(); });
  bindChange('usePenaltyMask', () => { persistSetup(); updateMaskWarning(); });
  ['qtStability', 'qtClipAlive', 'qtReacquireStrength', 'qtReacquireWindow', 'qtLostTimeout', 'qtGapMerge', 'qtReidStrength', 'qtSwapGuard'].forEach((id) => {
    bindChange(id, () => { applyQuickTuneToAdvanced(); persistSetup(); });
    const el = $(id);
    if (el) el.addEventListener('input', () => { applyQuickTuneToAdvanced(); persistSetup(); });
  });

  const vid = $('vid');
  const maskCanvas = $('maskCanvas');
  if (vid) {
    vid.addEventListener('click', (evt) => {
      registerSeedClick(evt);
    });
    vid.addEventListener('loadedmetadata', () => { drawClickMarkers(); drawMaskOverlay(); });
    vid.addEventListener('seeked', drawClickMarkers);
  }
  if (maskCanvas) {
    maskCanvas.addEventListener('pointerdown', registerMaskPoint);
    maskCanvas.addEventListener('pointermove', updateMaskPointer);
    maskCanvas.addEventListener('pointerup', endMaskPointer);
    maskCanvas.addEventListener('contextmenu', (evt) => {
      if (!state.drawMode) return;
      evt.preventDefault();
      registerMaskPoint(evt);
    });
  }
  window.addEventListener('resize', () => { drawClickMarkers(); drawMaskOverlay(); });
  loadMaskPolygonsFromStorage();
  applyPreset();
  applyVideoTypePreset(getValue('videoType', DEFAULT_SETUP.video_type));
  loadSavedSetupOrDefaults();
  drawMaskOverlay();
  setMaskDrawButtons();
  updateMaskWarning();
  document.querySelectorAll('input, select').forEach((el) => {
    if(el.id === 'file') return;
    if (el) {
      el.addEventListener('change', persistSetup);
      el.addEventListener('input', persistSetup);
    }
  });

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
