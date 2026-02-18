import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from api import main as api_main


def test_setup_json_persists_tracker_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(api_main, "JOBS_DIR", tmp_path)

    job_id = api_main.create_job({"name": "defaults-check"})["job_id"]
    api_main.setup_job(job_id, {})

    setup_path = tmp_path / job_id / "setup.json"
    setup = json.loads(setup_path.read_text())

    assert setup["allow_unconfirmed_clips"] is True
    assert setup["ocr_veto_conf"] == 0.995
    assert setup["ocr_veto_seconds"] == 0.5
    assert setup["reacquire_window_seconds"] == 10.0
    assert setup["lost_timeout"] == 3.0
    assert setup["reacquire_score_lock_threshold"] == 0.30
    assert setup["score_lock_threshold"] == 0.45
    assert setup["score_unlock_threshold"] == 0.25
    assert setup["gap_merge_seconds"] == 4.0
