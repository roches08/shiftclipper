import os
import time
import argparse
import logging

import redis
from rq import Queue, Worker
from rq.job import Job
from common.config import resolve_device


REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
QUEUE_NAMES = [q.strip() for q in os.getenv("RQ_QUEUES", "jobs").split(",") if q.strip()]
JOBS_DIR = os.getenv("JOBS_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "jobs")))


def configure_logging() -> None:
    log_path = os.getenv("WORKER_LOG_PATH", "/workspace/worker.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s job_id=%(job_id)s stage=%(stage)s %(name)s %(message)s")

    class JobContextFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            if not hasattr(record, "job_id"):
                record.job_id = "-"
            if not hasattr(record, "stage"):
                record.stage = "-"
            return True

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = []
    f = JobContextFilter()

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.addFilter(f)

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    fh.addFilter(f)

    root.addHandler(sh)
    root.addHandler(fh)




def gpu_info() -> str:
    try:
        import torch
        dev = resolve_device()
        cuda_ok = bool(torch.cuda.is_available())
        name = torch.cuda.get_device_name(0) if cuda_ok else "-"
        return f"device={dev} torch={torch.__version__} cuda_available={cuda_ok} gpu={name}"
    except Exception as e:
        return f"device={resolve_device()} torch=unavailable cuda_available=False gpu=- err={e}"


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

    configure_logging()
    conn = redis.from_url(REDIS_URL)
    logging.getLogger("worker").info("Worker starting | redis=%s | queues=%s | jobs_dir=%s | %s", REDIS_URL, QUEUE_NAMES, JOBS_DIR, gpu_info(), extra={"job_id":"-"})

    if args.self_test:
        raise SystemExit(run_self_test(conn))

    worker = Worker([Queue(name, connection=conn) for name in QUEUE_NAMES], connection=conn)
    worker.work()


if __name__ == "__main__":
    main()
