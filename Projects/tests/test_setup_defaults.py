import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from api import main as api_main
from worker import tasks as worker_tasks


def test_setup_json_persists_tracker_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(api_main, "JOBS_DIR", tmp_path)

    job_id = api_main.create_job({"name": "defaults-check"})["job_id"]
    api_main.setup_job(job_id, {})

    setup_path = tmp_path / job_id / "setup.json"
    setup = json.loads(setup_path.read_text())

    assert setup["tracker_type"] == "bytetrack"
    assert setup["allow_unconfirmed_clips"] is False
    assert setup["opponent_color"] == "#ffffff"
    assert setup["ocr_veto_conf"] == 0.92
    assert setup["ocr_veto_seconds"] == 1.0
    assert setup["reacquire_window_seconds"] == 8.0
    assert setup["lost_timeout"] == 4.0
    assert setup["reacquire_score_lock_threshold"] == 0.30
    assert setup["score_lock_threshold"] == 0.55
    assert setup["score_unlock_threshold"] == 0.33
    assert setup["swap_guard_seconds"] == 2.0
    assert setup["swap_guard_bonus"] == 0.1
    assert setup["reid_enable"] is True
    assert setup["reid_model"] == "osnet_x0_25"
    assert setup["reid_every_n_frames"] == 5
    assert setup["reid_weight"] == 0.4
    assert setup["reid_min_sim"] == 0.45
    assert setup["reid_crop_expand"] == 0.10
    assert setup["reid_batch"] == 16
    assert setup["reid_device"] == "cuda:0"
    assert setup["gap_merge_seconds"] == 1.5
    assert setup["transcode_enabled"] is False
    assert setup["transcode_scale_max"] == 1080
    assert setup["transcode_fps"] is None
    assert setup["transcode_deinterlace"] is True
    assert setup["transcode_denoise"] is False


def test_setup_overrides_persist_and_worker_consumes_and_debug_reflects_values(tmp_path, monkeypatch):
    monkeypatch.setattr(api_main, "JOBS_DIR", tmp_path)
    monkeypatch.setattr(worker_tasks, "JOBS_DIR", tmp_path)

    job_id = api_main.create_job({"name": "override-check"})["job_id"]
    job_dir = tmp_path / job_id
    (job_dir / "in.mp4").write_bytes(b"0")

    api_main.setup_job(job_id, {"tracking_mode": "shift", "score_lock_threshold": 0.45, "lost_timeout": 10.0, "tracker_type": "bytetrack"})

    meta = json.loads((job_dir / "job.json").read_text())
    meta["video_path"] = str(job_dir / "in.mp4")
    (job_dir / "job.json").write_text(json.dumps(meta))

    captured = {}

    def fake_track_presence(video_path, setup, heartbeat=None, cancel_check=None):
        captured["score_lock_threshold"] = setup.get("score_lock_threshold")
        captured["lost_timeout"] = setup.get("lost_timeout")
        captured["tracker_type"] = setup.get("tracker_type")
        return {"segments": [], "shifts": [], "timeline": [], "debug_overlay_path": None, "perf": {}, "target_embed_history": []}

    monkeypatch.setattr(worker_tasks, "track_presence", fake_track_presence)
    monkeypatch.setattr(worker_tasks, "resolve_device", lambda: ("cuda:0", 0))
    monkeypatch.setattr(worker_tasks, "analyze_video_quality", lambda _: {
        "width": 1920,
        "height": 1080,
        "fps": 30.0,
        "codec": "h264",
        "bit_rate": 1000000,
        "duration": 1.0,
        "is_vfr": False,
        "avg_frame_rate": "30/1",
        "r_frame_rate": "30/1",
    })

    result = worker_tasks.process_job(job_id)

    job_json = json.loads((job_dir / "job.json").read_text())
    debug_json = json.loads((job_dir / "debug.json").read_text())

    assert result["status"] == "done_no_shifts"
    assert job_json["setup"]["score_lock_threshold"] == 0.45
    assert job_json["setup"]["lost_timeout"] == 10.0
    assert captured["score_lock_threshold"] == 0.45
    assert captured["lost_timeout"] == 10.0
    assert captured["tracker_type"] == "bytetrack"
    assert debug_json["setup"]["score_lock_threshold"] == 0.45
    assert debug_json["setup"]["lost_timeout"] == 10.0


def test_non_bytetrack_tracker_is_normalized_to_bytetrack(tmp_path, monkeypatch):
    monkeypatch.setattr(api_main, "JOBS_DIR", tmp_path)

    job_id = api_main.create_job({"name": "tracker-normalize"})["job_id"]
    api_main.setup_job(job_id, {"tracker_type": "deepsort"})

    setup = json.loads((tmp_path / job_id / "setup.json").read_text())
    assert setup["tracker_type"] == "bytetrack"


def test_process_job_fails_fast_without_cuda(tmp_path, monkeypatch):
    monkeypatch.setattr(api_main, "JOBS_DIR", tmp_path)
    monkeypatch.setattr(worker_tasks, "JOBS_DIR", tmp_path)

    job_id = api_main.create_job({"name": "cuda-required"})["job_id"]
    job_dir = tmp_path / job_id
    (job_dir / "in.mp4").write_bytes(b"0")

    api_main.setup_job(job_id, {"tracker_type": "bytetrack"})

    meta = json.loads((job_dir / "job.json").read_text())
    meta["video_path"] = str(job_dir / "in.mp4")
    (job_dir / "job.json").write_text(json.dumps(meta))

    def _raise_cuda_required():
        raise RuntimeError("CUDA is required for ShiftClipper jobs, but no CUDA device is available")

    monkeypatch.setattr(worker_tasks, "resolve_device", _raise_cuda_required)

    try:
        worker_tasks.process_job(job_id)
        assert False, "process_job should raise when CUDA is unavailable"
    except RuntimeError as exc:
        assert "CUDA is required" in str(exc)

    result = json.loads((job_dir / "job.json").read_text())
    assert result["status"] == "error"
    assert "CUDA is required" in result["message"]


def test_process_job_transcode_writes_preflight_and_uses_transcoded_path(tmp_path, monkeypatch):
    monkeypatch.setattr(api_main, "JOBS_DIR", tmp_path)
    monkeypatch.setattr(worker_tasks, "JOBS_DIR", tmp_path)

    job_id = api_main.create_job({"name": "transcode-check"})["job_id"]
    job_dir = tmp_path / job_id
    src_path = job_dir / "in.mp4"
    src_path.write_bytes(b"0")

    api_main.setup_job(job_id, {"tracking_mode": "shift", "transcode_enabled": True, "transcode_scale_max": 1080, "transcode_fps": 30})

    meta = json.loads((job_dir / "job.json").read_text())
    meta["video_path"] = str(src_path)
    (job_dir / "job.json").write_text(json.dumps(meta))

    calls = {}

    def fake_analyze(path):
        if path.endswith("transcoded.mp4"):
            return {"codec": "h264", "pix_fmt": "yuv420p", "width": 1280, "height": 720, "avg_frame_rate": "30/1", "r_frame_rate": "30/1", "fps": 30.0, "duration": 1.0, "bit_rate": 1000000, "field_order": "progressive", "scan_type": "progressive", "interlaced_frame": 0, "is_interlaced": False, "is_vfr_likely": False}
        return {"codec": "mpeg4", "pix_fmt": "yuvj422p", "width": 1920, "height": 1080, "avg_frame_rate": "30000/1001", "r_frame_rate": "30/1", "fps": 29.97, "duration": 1.0, "bit_rate": 500000, "field_order": "tt", "scan_type": "interlaced", "interlaced_frame": 1, "is_interlaced": True, "is_vfr_likely": True}

    def fake_transcode(in_path, out_path, probe, setup):
        Path(out_path).write_bytes(b"transcoded")

    def fake_track_presence(video_path, setup, heartbeat=None, cancel_check=None):
        calls["video_path"] = video_path
        return {"segments": [], "shifts": [], "timeline": [{"t": 1.0, "event": "lock"}], "debug_overlay_path": None, "perf": {}, "target_embed_history": []}

    monkeypatch.setattr(worker_tasks, "resolve_device", lambda: ("cuda:0", 0))
    monkeypatch.setattr(worker_tasks, "analyze_video_quality", fake_analyze)
    monkeypatch.setattr(worker_tasks, "transcode_for_tracking", fake_transcode)
    monkeypatch.setattr(worker_tasks, "track_presence", fake_track_presence)

    result = worker_tasks.process_job(job_id)

    assert result["status"] == "done_no_shifts"
    assert calls["video_path"].endswith("transcoded.mp4")
    assert (job_dir / "transcoded.mp4").exists()
    assert (job_dir / "video_preflight.json").exists()

    timeline = json.loads((job_dir / "debug_timeline.json").read_text())
    events = [ev.get("event") for ev in timeline]
    assert "video_preflight" in events
    assert "transcode_started" in events
    assert "transcode_completed" in events
