# ShiftClipper Tracker (RunPod)

## Quick start (clean pod)

1) Download + unzip into `/workspace` (or any folder). Do **not** move individual subfolders afterwards.

2) Start services:

```bash
cd /workspace
chmod +x runpod_start.sh
bash runpod_start.sh
```

3) Verify:

```bash
redis-cli ping
ps aux | egrep "redis-server|rq worker|uvicorn" | grep -v egrep
curl -s http://127.0.0.1:8000/ | head
```

4) Open the RunPod HTTP endpoint for **port 8000**.

## Notes

- The backend auto-detects `APP_ROOT` from the location of `api/main.py` and `worker/tasks.py`, and `runpod_start.sh` exports `APP_ROOT` to the folder containing the project.  
- If you extract somewhere else, just `cd` there and run `bash runpod_start.sh`.
