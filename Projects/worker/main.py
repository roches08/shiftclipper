import os
import time
import argparse

import redis
from rq import Queue, Worker
from rq.job import Job
from worker.device import get_runtime_device_info


REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
QUEUE_NAMES = [q.strip() for q in os.getenv("RQ_QUEUES", "jobs").split(",") if q.strip()]
JOBS_DIR = os.getenv("JOBS_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "jobs")))


def run_self_test(conn: redis.Redis) -> int:
    from worker.tasks import self_test_task

    queue = Queue("selftest", connection=conn)
    worker = Worker([queue], connection=conn)
    token = str(int(time.time()))
    job = queue.enqueue(self_test_task, {"token": token}, job_timeout=60)
    print(f"Self-test enqueue | redis={REDIS_URL} | queue=selftest | job_id={job.id}")
    worker.work(burst=True)
    refreshed = Job.fetch(job.id, connection=conn)
    print(f"Self-test complete | status={refreshed.get_status()} | result={refreshed.result}")
    return 0 if refreshed.get_status() == "finished" else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="ShiftClipper RQ worker")
    parser.add_argument("--self-test", action="store_true", help="Run a queue wiring self-test and exit")
    args = parser.parse_args()

    os.makedirs(JOBS_DIR, exist_ok=True)
    probe_path = os.path.join(JOBS_DIR, ".write_probe")
    with open(probe_path, "w", encoding="utf-8") as f:
        f.write("ok\n")
    os.remove(probe_path)

    conn = redis.from_url(REDIS_URL)
    dev = get_runtime_device_info()
    print(
        "Worker starting"
        f" | redis={REDIS_URL}"
        f" | queues={QUEUE_NAMES}"
        f" | jobs_dir={JOBS_DIR}"
        f" | selected_device={dev['selected_device']}"
        f" | torch={dev['torch_version']}"
        f" | cuda_available={dev['cuda_available']}"
        f" | gpu_name={dev['gpu_name']}"
    )

    if args.self_test:
        raise SystemExit(run_self_test(conn))

    worker = Worker([Queue(name, connection=conn) for name in QUEUE_NAMES], connection=conn)
    worker.work()


if __name__ == "__main__":
    main()
