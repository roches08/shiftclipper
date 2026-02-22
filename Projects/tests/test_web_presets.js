const test = require('node:test');
const assert = require('node:assert/strict');

const { VIDEO_TYPE_PRESETS, getVideoTypePreset, VIDEO_TYPE_ORDER } = require('../static/presets.js');

test('video type order is deterministic and default first', () => {
  assert.deepEqual(VIDEO_TYPE_ORDER, ['coach_cam', 'wide_single_cam', 'broadcast']);
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
