import os

import redis
from rq import Connection, Queue, Worker


REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379")


def main() -> None:
    conn = redis.from_url(REDIS_URL)
    print(f"Connected to Redis at: {REDIS_URL}")

    with Connection(conn):
        worker = Worker([Queue("jobs", connection=conn)])
        worker.work()


if __name__ == "__main__":
    main()
