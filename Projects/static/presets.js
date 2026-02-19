(function initShiftClipperPresets(global) {
  const REID_DEFAULTS = {
    reid_fail_policy: 'disable',
    reid_auto_download: true,
    reid_weights_path: '/workspace/shiftclipper/Projects/models/reid/osnet_x0_25_msmt17.pth',
    reid_weights_url: 'https://huggingface.co/kaiyangzhou/osnet/resolve/main/osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth',
  };

  const ADVANCED_PRESETS = {
    balanced: {
      score_lock_threshold: 0.55,
      score_unlock_threshold: 0.33,
      lost_timeout: 4.0,
      reacquire_window_seconds: 8.0,
      reacquire_score_lock_threshold: 0.4,
      gap_merge_seconds: 2.5,
      lock_seconds_after_confirm: 4,
      min_track_seconds: 0.75,
      min_clip_seconds: 1,
      allow_unconfirmed_clips: false,
      allow_seed_clips: true,
      seed_lock_seconds: 8,
      seed_iou_min: 0.12,
      seed_dist_max: 0.16,
      seed_bonus: 0.8,
      seed_window_s: 3,
      ocr_disable: false,
      ocr_every_n_frames: 12,
      ocr_min_conf: 0.2,
      ocr_veto_conf: 0.92,
      ocr_veto_seconds: 1.0,
      detect_stride: 2,
      yolo_imgsz: 512,
      yolo_batch: 4,
      tracker_type: 'bytetrack',
      reid_enable: true,
      reid_model: 'osnet_x0_25',
      reid_every_n_frames: 5,
      reid_weight: 0.45,
      reid_min_sim: 0.5,
      reid_crop_expand: 0.15,
      reid_batch: 16,
      reid_device: 'cuda:0',
      ...REID_DEFAULTS,
      swap_guard_seconds: 2.5,
      swap_guard_bonus: 0.1,
    },
    more_clips: {
      score_lock_threshold: 0.45,
      score_unlock_threshold: 0.20,
      lost_timeout: 4.0,
      reacquire_window_seconds: 10,
      reacquire_score_lock_threshold: 0.30,
      gap_merge_seconds: 4.0,
      min_track_seconds: 0.35,
      min_clip_seconds: 0.75,
      allow_unconfirmed_clips: true,
      allow_seed_clips: true,
      seed_lock_seconds: 12,
      seed_iou_min: 0.08,
      seed_dist_max: 0.22,
      ocr_veto_conf: 0.92,
      ocr_veto_seconds: 1.0,
      reid_enable: true,
      reid_model: 'osnet_x0_25',
      reid_every_n_frames: 5,
      reid_weight: 0.4,
      reid_min_sim: 0.48,
      reid_crop_expand: 0.15,
      reid_batch: 16,
      reid_device: 'cuda:0',
      ...REID_DEFAULTS,
      swap_guard_seconds: 2.0,
      swap_guard_bonus: 0.08,
    },
    track_quality: {
      score_lock_threshold: 0.60,
      score_unlock_threshold: 0.40,
      lost_timeout: 1.5,
      reacquire_window_seconds: 4,
      reacquire_score_lock_threshold: 0.45,
      gap_merge_seconds: 2.0,
      min_track_seconds: 0.90,
      min_clip_seconds: 1.25,
      allow_unconfirmed_clips: false,
      seed_iou_min: 0.14,
      seed_dist_max: 0.14,
      ocr_veto_conf: 0.85,
      ocr_veto_seconds: 2.0,
      reid_enable: true,
      reid_model: 'osnet_x0_25',
      reid_every_n_frames: 5,
      reid_weight: 0.5,
      reid_min_sim: 0.55,
      reid_crop_expand: 0.15,
      reid_batch: 16,
      reid_device: 'cuda:0',
      ...REID_DEFAULTS,
      swap_guard_seconds: 3.0,
      swap_guard_bonus: 0.12,
    },
  };

  function deepCopy(value) {
    if (typeof structuredClone === 'function') return structuredClone(value);
    return JSON.parse(JSON.stringify(value));
  }

  function getAdvancedPreset(name) {
    const chosen = ADVANCED_PRESETS[name] || ADVANCED_PRESETS.balanced;
    return { ...deepCopy(ADVANCED_PRESETS.balanced), ...deepCopy(chosen) };
  }

  const api = { ADVANCED_PRESETS, getAdvancedPreset, deepCopy };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
  global.ShiftClipperPresets = api;
})(typeof window !== 'undefined' ? window : globalThis);
