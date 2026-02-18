const test = require('node:test');
const assert = require('node:assert/strict');

const { ADVANCED_PRESETS, getAdvancedPreset } = require('../web/presets.js');

test('editing applied preset does not mutate source preset', () => {
  const applied = getAdvancedPreset('balanced');
  applied.reid_weight = 0.01;
  applied.swap_guard_seconds = 9.9;

  assert.equal(ADVANCED_PRESETS.balanced.reid_weight, 0.45);
  assert.equal(ADVANCED_PRESETS.balanced.swap_guard_seconds, 2.5);
});

test('switching presets restores original values', () => {
  const moreClips = getAdvancedPreset('more_clips');
  moreClips.reid_min_sim = 0.01;

  const restored = getAdvancedPreset('more_clips');

  assert.equal(restored.reid_min_sim, 0.48);
  assert.equal(restored.reid_weight, 0.4);
  assert.equal(restored.swap_guard_seconds, 2.0);
  assert.equal(restored.swap_guard_bonus, 0.08);
});
