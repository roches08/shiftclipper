const test = require('node:test');
const assert = require('node:assert/strict');

const { VIDEO_TYPE_PRESETS, getVideoTypePreset, VIDEO_TYPE_ORDER } = require('../static/presets.js');

test('video type order is deterministic and default first', () => {
  assert.deepEqual(VIDEO_TYPE_ORDER, ['wide_single_cam_working_v1', 'hockey_wide_single_cam_stable', 'coach_cam', 'wide_single_cam', 'broadcast']);
});

test('new working preset includes expected profile values', () => {
  const preset = getVideoTypePreset('wide_single_cam_working_v1');
  assert.equal(preset.preset_name, 'Wide Single Cam — Working (Test 2 profile)');
  assert.equal(preset.values.allow_seed_clips, false);
  assert.equal(preset.values.score_unlock_threshold, 0.25);
  assert.equal(preset.values.reacquire_window_seconds, 90);
});

test('editing applied preset does not mutate source preset', () => {
  const applied = getVideoTypePreset('coach_cam');
  applied.values.reid_weight = 0.01;
  applied.values.swap_guard_seconds = 9.9;

  assert.equal(VIDEO_TYPE_PRESETS.coach_cam.values.reid_weight, 0.55);
  assert.equal(VIDEO_TYPE_PRESETS.coach_cam.values.swap_guard_seconds, 8);
});

test('broadcast preset keeps cold lock gating enabled', () => {
  const preset = getVideoTypePreset('broadcast');
  assert.equal(preset.values.seed_window_s, 45);
  assert.equal(preset.values.cold_lock_mode, 'require_seed');
  assert.equal(preset.values.swap_guard_seconds, 10);
});

test('hockey stable preset exposes expected stable defaults', () => {
  const preset = getVideoTypePreset('hockey_wide_single_cam_stable');
  assert.equal(preset.values.score_lock_threshold, 0.47);
  assert.equal(preset.values.clip_continue_threshold, 0.23);
  assert.equal(preset.values.reacquire_window_s, 110);
  assert.equal(preset.values.swap_guard_bonus, 0.3);
});
