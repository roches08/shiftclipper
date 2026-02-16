import os
import time
import argparse
import logging

import redis
from rq import Connection, Queue, Worker
from rq.job import Job


REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
QUEUE_NAMES = [q.strip() for q in os.getenv("RQ_QUEUES", "jobs").split(",") if q.strip()]
JOBS_DIR = os.getenv("JOBS_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "jobs")))
REDIS_MAX_RETRIES = int(os.getenv("REDIS_MAX_RETRIES", "3"))
REDIS_RETRY_DELAY_S = float(os.getenv("REDIS_RETRY_DELAY_S", "0.5"))

log = logging.getLogger("worker.main")
log.setLevel(logging.INFO)


def redis_conn_with_retry() -> redis.Redis:
    last_exc = None
    for attempt in range(1, REDIS_MAX_RETRIES + 1):
        try:
            conn = redis.from_url(
                REDIS_URL,
                socket_connect_timeout=3,
                socket_timeout=3,
                retry_on_timeout=True,
            )
            conn.ping()
            return conn
        except Exception as exc:
            last_exc = exc
            log.warning(f"worker_redis_connect_retry attempt={attempt} error={exc}")
            time.sleep(REDIS_RETRY_DELAY_S)
    raise RuntimeError(f"Could not connect to Redis at {REDIS_URL}: {last_exc}")


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

    conn = redis_conn_with_retry()
    assert QUEUE_NAMES, "RQ_QUEUES must define at least one queue"
    log.info(f"worker_ready redis={REDIS_URL} queues={QUEUE_NAMES} jobs_dir={JOBS_DIR}")
    print(f"Worker starting | redis={REDIS_URL} | queues={QUEUE_NAMES} | jobs_dir={JOBS_DIR}")

    if args.self_test:
        raise SystemExit(run_self_test(conn))

    with Connection(conn):
        worker = Worker([Queue(name, connection=conn) for name in QUEUE_NAMES])
        worker.work()


if __name__ == "__main__":
    main()
