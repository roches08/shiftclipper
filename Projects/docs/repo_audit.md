# Repository Audit (Projects app)

## Intended UI strategy

This repository follows **Strategy A**: committed static UI assets under `Projects/static` (`index.html`, `app.js`, `presets.js`) served directly by the API at `/` and `/static/*`. There is no frontend build project (`package.json`) in this repo.

## Top-level inventory and classification

- `.git/` — Git metadata (local only).
- `Projects/` — **Source code (keep, tracked)**.
- `.gitattributes` — **Source code/config (keep, tracked)**.
- `docker-compose.yml` — **Source code/config (keep, tracked)**.
- `.gitignore` — **Source code/config (keep, tracked)**.

## `Projects/` inventory and classification

- `api/` — **Source code (keep, tracked)**.
- `worker/` — **Source code (keep, tracked)**.
- `common/` — **Source code (keep, tracked)**.
- `static/` — **Source code (keep, tracked)** committed UI assets.
- `tests/` — **Source code (keep, tracked)**.
- `scripts/` — **Source code/tooling (keep, tracked)**.
- `docs/` — **Documentation (keep, tracked)**.
- `data/jobs/` — **Runtime data (do not track; gitignored)**.
- `models/reid/` — **Runtime model cache/weights (do not track; gitignored)**.
- `.venv/` — **Environment cache (do not track; gitignored)**.

## Cleanup decisions in this PR

- Removed tracked Finder junk files (`.DS_Store`).
- Removed duplicated UI source folder `Projects/web/` to prevent UI drift and random parallel frontend trees.
- Updated tests to import committed UI presets from `Projects/static/presets.js`.
- Expanded `.gitignore` to block runtime/build/cache artifacts and model weights.
- Made ReID transitive dependency requirement explicit (`tensorboard`, `yacs`, `gdown`) in non-pro requirements files.
