# MLCanvas Backend

FastAPI backend that accepts graph JSON from the frontend, validates and compiles it into PyTorch, and runs async NN training jobs with websocket metric streaming.

## Remote Training Worker (Modal)

You can offload NN training to a Modal GPU worker.

1. Deploy the worker from the backend directory:

```bash
cd /Users/yax/programming/burn/backend
modal deploy modal_worker.py
```

2. In the backend environment, configure:

```bash
TRAINING_BACKEND=modal
MODAL_APP_NAME=burn-training
MODAL_FUNCTION_NAME=train_job_remote
# Optional: MODAL_ENVIRONMENT_NAME=<your-modal-environment>
```

3. Start backend normally (`uv run python main.py` or uvicorn).

When `TRAINING_BACKEND=modal`, `/api/model/train` runs the full training loop on Modal and then persists the returned `.pt` weights into the normal local artifacts directory.

Optional Modal speed knobs (set in backend env before starting API):

```bash
# Try torch.compile in remote worker (default: 1)
MODAL_USE_TORCH_COMPILE=1

# Multiply training batch size from graph payload (default: 1)
MODAL_BATCH_SIZE_MULTIPLIER=2

# Evaluate test split every N epochs instead of every epoch (default: 1)
MODAL_EVAL_EVERY=2

# Cap train/test batches per epoch for very fast iteration (default: 0 = full)
MODAL_MAX_TRAIN_BATCHES=0
MODAL_MAX_TEST_BATCHES=0
```

## Run

```bash
cd /Users/yax/programming/burn/backend
uv run uvicorn main:app --reload
# or
uv run python main.py
```

For long-running training jobs, prefer:

```bash
uv run python main.py
```

`python main.py` runs with `reload=False`, which avoids in-memory job loss during dataset downloads/training.

## API Client Script

```bash
cd /Users/yax/programming/burn/backend
uv run python scripts/use_api.py
```

`scripts/use_api.py` is hardcoded (non-CLI). Edit the constants at the top of the file to change:
- backend URL
- graph file path
- training hyperparameters
- websocket behavior
- inference input path
- output directory

## WebSocket Watcher Script

Starts (or watches) a training job and prints live websocket updates:

```bash
cd /Users/yax/programming/burn/backend
uv run python scripts/watch_training_ws.py
```

Behavior:
- default mode is always-on daemon watcher with colorized logs.
- `START_NEW_JOB = False`: watches jobs started elsewhere and auto-attaches to new latest jobs.
- `START_NEW_JOB = True`: watcher starts jobs itself and can auto-restart indefinitely.
- `ALWAYS_WATCH = True`: never exits on job completion; it waits for next job.

## API

- `POST /api/model/validate`
- `POST /api/model/compile`
- `POST /api/model/train`
- `POST /api/model/stop`
- `GET /api/model/status?job_id=<id>`
- `POST /api/model/infer`
- `GET /api/model/export?job_id=<id>&format=py|pt`
- `POST /api/deploy` (create local deployment from a trained NN job)
- `GET /api/deploy/list`
- `GET /api/deploy/status?deployment_id=<id>`
- `GET /api/deploy/logs?deployment_id=<id>&limit=<n>`
- `POST /api/deploy/{deployment_id}/infer`
- `POST /api/deploy/{deployment_id}/start`
- `DELETE /api/deploy/{deployment_id}`
- `GET /api/datasets`
- `WS /ws/training/{job_id}`

### Random Forest API

- `POST /api/rf/validate`
- `POST /api/rf/compile`
- `POST /api/rf/train`
- `POST /api/rf/stop`
- `GET /api/rf/status?job_id=<id>`
- `GET /api/rf/latest`
- `POST /api/rf/infer`
- `GET /api/rf/export?job_id=<id>&format=py|pkl`
- `GET /api/rf/datasets`
- `WS /ws/rf/training/{job_id}`

## Notes

- v1 supports sequential graphs only.
- Supported layers: `Input`, `Dense`, `Dropout`, `Flatten`, `Output`.
- NN datasets:
  - `mnist` (Kaggle: `oddrationale/mnist-in-csv`, requires Kaggle credentials)
  - `digits` (scikit-learn built-in 8x8 digits dataset, no Kaggle auth required)
- Training uses the train split and evaluates on the test split each epoch.
- WebSocket `epoch_update` includes both train and test metrics (`train_*`, `test_*`).
- RF datasets are Kaggle-backed (`iris`, `wine`, `breast_cancer`) and fail fast if Kaggle auth/download is unavailable.
- Every train job is persisted to disk for later reuse:
  - NN jobs: `/Users/yax/programming/burn/backend/artifacts/jobs/<job_id>/`
  - RF jobs: `/Users/yax/programming/burn/backend/artifacts/rf/jobs/<job_id>/`
  - Bundle files include `model.py`, `graph.json`, `training.json`, `summary.json`, `metadata.json`, and trained artifact (`model.pt` or `model.pkl`) when available.
- Deployment Manager metadata/logs are persisted to:
  - `/Users/yax/programming/burn/backend/artifacts/deployments/registry.json`
  - After backend restart, deployments restore as `stopped` (runtime model objects are in-memory only) and can be restarted from `/deployments`.

Kaggle setup quick check:

```bash
kaggle datasets list -s mnist
```
