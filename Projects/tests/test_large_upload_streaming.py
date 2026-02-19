import json
from pathlib import Path

from fastapi.testclient import TestClient

from api import main


def test_upload_writes_part_then_renames(monkeypatch, tmp_path):
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(parents=True)
    monkeypatch.setattr(main, "JOBS_DIR", jobs_dir)

    def fake_proxy(in_path: str, out_path: str, max_h: int = 360, fps: int = 30) -> bool:
        Path(out_path).write_bytes(b"proxy")
        return True

    monkeypatch.setattr(main, "make_proxy", fake_proxy)

    client = TestClient(main.app)
    create = client.post("/jobs", json={"name": "upload-test"})
    assert create.status_code == 200
    job_id = create.json()["job_id"]

    payload = b"a" * (3 * 1024 * 1024)
    resp = client.post(
        f"/jobs/{job_id}/upload",
        files={"file": ("input.mp4", payload, "video/mp4")},
    )
    assert resp.status_code == 200

    job_path = jobs_dir / job_id
    assert (job_path / "in.mp4").read_bytes() == payload
    assert not (job_path / "in.mp4.part").exists()

    meta = json.loads((job_path / "job.json").read_text())
    assert meta["video_path"] == str(job_path / "in.mp4")
    assert meta["bytes_received"] == len(payload)
    assert meta["bytes_total"] is not None
