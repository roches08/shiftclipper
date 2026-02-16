import os

import redis
from rq import Connection, Queue, Worker


REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
QUEUE_NAMES = [q.strip() for q in os.getenv("RQ_QUEUES", "jobs").split(",") if q.strip()]
JOBS_DIR = os.getenv("JOBS_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "jobs")))


def main() -> None:
    os.makedirs(JOBS_DIR, exist_ok=True)
    probe_path = os.path.join(JOBS_DIR, ".write_probe")
    with open(probe_path, "w", encoding="utf-8") as f:
        f.write("ok\n")
    os.remove(probe_path)

    conn = redis.from_url(REDIS_URL)
    print(f"Worker starting | redis={REDIS_URL} | queues={QUEUE_NAMES} | jobs_dir={JOBS_DIR}")

    with Connection(conn):
        worker = Worker([Queue(name, connection=conn) for name in QUEUE_NAMES])
        worker.work()


if __name__ == "__main__":
    main()
