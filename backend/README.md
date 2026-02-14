# MLCanvas Backend

FastAPI backend that accepts graph JSON from the frontend, validates and compiles it into PyTorch, and runs async MNIST training jobs with websocket metric streaming.

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
- `GET /api/datasets`
- `WS /ws/training/{job_id}`

## Notes

- v1 supports sequential graphs only.
- Supported layers: `Input`, `Dense`, `Dropout`, `Flatten`, `Output`.
- Dataset support is MNIST only in v1.
- MNIST is downloaded from Kaggle dataset `oddrationale/mnist-in-csv`.
- You must have the Kaggle CLI configured (`kaggle.json` credentials).
- Training uses the train split and evaluates on the test split each epoch.
- WebSocket `epoch_update` includes both train and test metrics (`train_*`, `test_*`).

Kaggle setup quick check:

```bash
kaggle datasets list -s mnist
```
