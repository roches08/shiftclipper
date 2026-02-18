import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from api import main as api_main
from worker import tasks as worker_tasks


def test_setup_json_persists_tracker_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(api_main, "JOBS_DIR", tmp_path)

    job_id = api_main.create_job({"name": "defaults-check"})["job_id"]
    api_main.setup_job(job_id, api_main.SetupRequest())

    setup_path = tmp_path / job_id / "setup.json"
    setup = json.loads(setup_path.read_text())

    assert setup["allow_unconfirmed_clips"] is False
    assert setup["allow_seed_clips"] is True
    assert setup["ocr_veto_conf"] == 0.85
    assert setup["ocr_veto_seconds"] == 2.0
    assert setup["reacquire_window_seconds"] == 4.0
    assert setup["lost_timeout"] == 1.5
    assert setup["reacquire_score_lock_threshold"] == 0.40
    assert setup["score_lock_threshold"] == 0.55
    assert setup["score_unlock_threshold"] == 0.35
    assert setup["gap_merge_seconds"] == 2.5


def test_worker_reads_non_default_setup_values(tmp_path, monkeypatch):
    job_id = "jobabc123"
    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True)

    fake_video = job_dir / "in.mp4"
    fake_video.write_bytes(b"stub")

    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "video_path": str(fake_video),
            }
        )
    )
    (job_dir / "setup.json").write_text(
        json.dumps(
            {
                "score_lock_threshold": 0.91,
                "lost_timeout": 3.25,
                "verify_mode": True,
            }
        )
    )

    monkeypatch.setattr(worker_tasks, "JOBS_DIR", tmp_path)
    monkeypatch.setattr(worker_tasks, "get_current_job", lambda: None)

    result = worker_tasks.process_job(job_id)

    persisted_setup = json.loads((job_dir / "setup.json").read_text())
    results_json = json.loads((job_dir / "results.json").read_text())

    assert persisted_setup["score_lock_threshold"] == 0.91
    assert persisted_setup["lost_timeout"] == 3.25
    assert result["status"] == "verified"
    assert results_json["status"] == "verified"
