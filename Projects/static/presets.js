(function initShiftClipperPresets(global) {
  const PRESET_VERSION = 'v1';
  const VIDEO_TYPE_ORDER = ['coach_cam', 'wide_single_cam', 'broadcast'];
  const VIDEO_TYPE_PRESETS = {
    coach_cam: {
      preset_name: 'Coach Cam (Single-cam, stable)',
      preset_version: PRESET_VERSION,
      values: { score_lock_threshold: 0.5, score_unlock_threshold: 0.38, lost_timeout: 6, reacquire_window_s: 14, reacquire_score_lock_threshold: 0.35, gap_merge_seconds: 0.5, extend_sec: 1.5, allow_bench_reacquire: false, allow_seed_clips: true, seed_lock_seconds: 4, seed_window_s: 60, seed_iou_min: 0.18, seed_dist_max: 0.16, seed_bonus: 0.8, reid_enabled: true, reid_model: 'osnet_x0_25', reid_every_n_frames: 3, reid_weight: 0.55, reid_min_similarity: 0.45, reid_crop_expand: 0.2, reid_batch: 16, reid_min_px: 12, reid_sharpness_threshold: 10, swap_guard_seconds: 8, swap_guard_bonus: 0.3, cold_lock_mode: 'require_seed', cold_lock_reid_min_similarity: 0.5, cold_lock_margin_min: 0.08, cold_lock_max_seconds: 3, lock_threshold_normal: 0.5, lock_threshold_reacquire: 0.35, lock_threshold_seed: 0.38, max_clip_len_sec: 0 },
    },
    wide_single_cam: {
      preset_name: 'Wide Single Cam (Live barn / fixed wide)',
      preset_version: PRESET_VERSION,
      values: { score_lock_threshold: 0.5, score_unlock_threshold: 0.38, lost_timeout: 6, reacquire_window_s: 14, reacquire_score_lock_threshold: 0.35, gap_merge_seconds: 0.5, extend_sec: 1.5, allow_bench_reacquire: false, allow_seed_clips: true, seed_lock_seconds: 4, seed_window_s: 90, seed_iou_min: 0.18, seed_dist_max: 0.16, seed_bonus: 0.8, reid_enabled: true, reid_model: 'osnet_x0_25', reid_every_n_frames: 3, reid_weight: 0.55, reid_min_similarity: 0.45, reid_crop_expand: 0.2, reid_batch: 16, reid_min_px: 12, reid_sharpness_threshold: 10, swap_guard_seconds: 10, swap_guard_bonus: 0.3, cold_lock_mode: 'require_seed', cold_lock_reid_min_similarity: 0.5, cold_lock_margin_min: 0.08, cold_lock_max_seconds: 3, lock_threshold_normal: 0.5, lock_threshold_reacquire: 0.35, lock_threshold_seed: 0.38, max_clip_len_sec: 0 },
    },
    broadcast: {
      preset_name: 'Broadcast (cuts/zooms)',
      preset_version: PRESET_VERSION,
      values: { score_lock_threshold: 0.5, score_unlock_threshold: 0.38, lost_timeout: 6, reacquire_window_s: 14, reacquire_score_lock_threshold: 0.35, gap_merge_seconds: 0.5, extend_sec: 1.5, allow_bench_reacquire: false, allow_seed_clips: true, seed_lock_seconds: 4, seed_window_s: 45, seed_iou_min: 0.18, seed_dist_max: 0.16, seed_bonus: 0.8, reid_enabled: true, reid_model: 'osnet_x0_25', reid_every_n_frames: 3, reid_weight: 0.55, reid_min_similarity: 0.45, reid_crop_expand: 0.2, reid_batch: 16, reid_min_px: 12, reid_sharpness_threshold: 10, swap_guard_seconds: 10, swap_guard_bonus: 0.3, cold_lock_mode: 'require_seed', cold_lock_reid_min_similarity: 0.5, cold_lock_margin_min: 0.08, cold_lock_max_seconds: 3, lock_threshold_normal: 0.5, lock_threshold_reacquire: 0.35, lock_threshold_seed: 0.38, max_clip_len_sec: 0 },
    },
  };

  const deepCopy = (value) => (typeof structuredClone === 'function' ? structuredClone(value) : JSON.parse(JSON.stringify(value)));
  const getVideoTypePreset = (name) => deepCopy(VIDEO_TYPE_PRESETS[name] || VIDEO_TYPE_PRESETS.coach_cam);
  const api = { PRESET_VERSION, VIDEO_TYPE_ORDER, VIDEO_TYPE_PRESETS, getVideoTypePreset, deepCopy };

  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  global.ShiftClipperPresets = api;
})(typeof window !== 'undefined' ? window : globalThis);
