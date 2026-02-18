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
    assert setup["score_lock_threshold"] == 0.50
    assert setup["score_unlock_threshold"] == 0.33
    assert setup["gap_merge_seconds"] == 1.5


def test_setup_overrides_persist_and_worker_consumes(tmp_path, monkeypatch):
    monkeypatch.setattr(api_main, "JOBS_DIR", tmp_path)
    monkeypatch.setattr(worker_tasks, "JOBS_DIR", tmp_path)

    job_id = api_main.create_job({"name": "override-check"})["job_id"]
    job_dir = tmp_path / job_id
    (job_dir / "in.mp4").write_bytes(b"0")

    api_main.setup_job(job_id, {"tracking_mode": "shift", "score_lock_threshold": 0.73, "lost_timeout": 2.7, "tracker_type": "bytetrack"})

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

    result = worker_tasks.process_job(job_id)

    assert result["status"] == "done_no_shifts"
    assert captured["score_lock_threshold"] == 0.73
    assert captured["lost_timeout"] == 2.7
    assert captured["tracker_type"] == "bytetrack"
