# ShiftClipper Repository Audit (2026-02-17)

## Files Examined
- `Projects/common/config.py`
- `Projects/worker/tasks.py`
- `Projects/api/main.py`
- `Projects/worker/main.py`
- `Projects/runpod_start.sh`
- `Projects/web/index.html`
- `Projects/web/app.js`
- `Projects/scripts/eval_tracker.py`
- `Projects/scripts/smoke_test.py`

## 1) Tracker stability improvements

### ✅ Verified
- Camera presets include `broadcast_wide` with tuned defaults in both backend normalization and frontend preset mapping.
- Multi-signal identity logic is present in scoring:
  - OCR match signal (`ocr_match`) with confidence thresholding.
  - Motion signal via IoU with prior box (`motion`).
  - Color match (`_color_score`).
  - Lock persistence (`identity_bonus` when current track matches locked track, plus `locked_until`).
- Relevant parameters are exposed in setup normalization (`color_weight`, `motion_weight`, `ocr_weight`, `identity_weight`, lock/merge/lost/stride/min_track, OCR window params).

### ⚠️ Inconsistency
- Frontend advanced defaults for `ocr_min_conf` and some lock/timeout values are generic until preset apply; this is okay because preset apply runs on DOM load, but if custom code bypasses it backend defaults still differ from raw HTML defaults.

## 2) Shift Mode

### ✅ Verified
- `tracking_mode` exists in setup normalization and UI payload/options.
- Shift start/end state logic exists with `OFF_ICE` → `ON_ICE` → `EXITING` transitions and bench-zone gating.
- Bench detection parameter `bench_zone_ratio` exists in config and is used in shift exit logic.

### ❌ Missing / Partial
- `tracking_mode` is not used to alter worker output behavior. The worker always creates clips and combined output paths regardless of `tracking_mode`, then fails if no clips/combined are created.
- In shift workflows, expected behavior is typically to return shift summaries even when no clips are generated; currently this can fail runs unnecessarily.

## 3) Progress & Status

### ✅ Verified
- Upload progress handling exists via XHR `upload.onprogress` in UI.
- Worker status updates include `status/stage/progress/message` and stage transitions (`queued` → `tracking` → `clips` → `combined` → `done`).
- API status endpoint merges RQ state and exposes a stable status payload.

### ⚠️ Inconsistency
- `upload` writes `bytes_received`/`bytes_total` to `job.json`, but `/status` response does not include these fields. If server-side upload progress was expected through `/status`, this is currently omitted.

## 4) Results & Artifacts

### ✅ Verified
- Results include `artifacts` with clips and optional combined/debug paths.
- Validation for missing expected outputs raises failure when verify mode is off.
- Verify mode returns explicit `status: verified` with no generated clips.

### ❌ Bug
- `artifacts["list"]` appends `debug_timeline` using `debug_json_path/url` instead of `debug_timeline_path/url`, causing a mislabeled pointer.

## 5) Debug Tools

### ✅ Verified
- Debug overlay generation exists (`debug_overlay.mp4`) when enabled.
- Debug timeline JSON exists and is a coherent event list (`state`, `shift_state`, `merge`, timestamps).

## 6) GPU Default & Installation

### ✅ Verified
- Device resolver defaults to `cuda:0` when available.
- `runpod_start.sh` defaults `SHIFTCLIPPER_DEVICE` to `cuda:0`.
- Worker startup log includes device, torch version, cuda availability, and GPU name.

## 7) Startup Script Improvements

### ✅ Verified
- Startup checks `/api/health`, `/`, and `/static/app.js`.
- Worker startup logging is explicit via `worker.main` startup log line.
- Script fails if API checks fail (due to `set -e` + curl failures) or if worker process exits early.

### ⚠️ Inconsistency
- API failure path relies on shell exit status but does not print a custom diagnostic message before exiting.

## 8) UI Controls

### ✅ Verified
- Camera mode includes `broadcast_wide`.
- UI includes `tracking_mode` selector.
- Verify warning banner + run confirmation are present.
- Advanced tracking settings and debug toggles are present.

## 9) Smoke / Evaluation Tools

### ✅ Verified
- `Projects/scripts/eval_tracker.py` exists and exercises setup/run/status/results.
- `Projects/scripts/smoke_test.py` exists and validates basic output expectations when sample video is provided.

---

## Suggested Minimal Patches (for missing/buggy areas)

### Patch A: Respect `tracking_mode=shift` when deciding no-output failure

```diff
diff --git a/Projects/worker/tasks.py b/Projects/worker/tasks.py
@@
-        if not clips and not combined_path:
-            raise RuntimeError("No clips created; see debug overlay/timeline (checked clips and combined outputs)")
+        tracking_mode = str(setup.get("tracking_mode", "clip")).lower()
+        if tracking_mode == "clip" and not clips and not combined_path:
+            raise RuntimeError("No clips created; see debug overlay/timeline (checked clips and combined outputs)")
+        if tracking_mode == "shift" and not shifts_json:
+            raise RuntimeError("No shifts detected; see debug overlay/timeline")
```

### Patch B: Fix debug timeline artifact pointer in `artifacts.list`

```diff
diff --git a/Projects/worker/tasks.py b/Projects/worker/tasks.py
@@
-        if artifacts.get("debug_json_path"):
-            artifacts["list"].append({"type": "debug_timeline", "path": artifacts["debug_json_path"], "url": artifacts["debug_json_url"]})
+        if artifacts.get("debug_timeline_path"):
+            artifacts["list"].append({"type": "debug_timeline", "path": artifacts["debug_timeline_path"], "url": artifacts["debug_timeline_url"]})
```

### Patch C (optional): Return server-side upload byte counters in `/status`

```diff
diff --git a/Projects/api/main.py b/Projects/api/main.py
@@
     status_payload = {
         "job_id": job_id,
@@
         "proxy_ready": meta.get("proxy_ready"),
         "proxy_url": meta.get("proxy_url"),
+        "bytes_received": meta.get("bytes_received"),
+        "bytes_total": meta.get("bytes_total"),
     }
```

## Additional Logic Risks / Notes
- `track_presence` computes both `segments` and `shifts`, but only clip-cutting drives primary success criteria today; this can make shift-only workflows brittle.
- `verify_mode` returns `stage: done` and `status: verified`, which is coherent but differs from the stage enum used elsewhere (`done` as stage, `verified` as status).
