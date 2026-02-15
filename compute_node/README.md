# VLM Compute Node

Small FastAPI service that runs VLM inference on a remote GPU machine (for example, your ASUS GX10) and exposes both HTTP + websocket endpoints.

## Run

```bash
cd /Users/rajpandya/workspace/burn/compute_node
uv sync
uv run uvicorn main:app --host 0.0.0.0 --port 8100
```

## API

- `GET /health`
- `POST /api/v1/vlm/infer`
- `WS /ws/v1/vlm/infer`

### Example infer payload

```json
{
  "image_base64": "data:image/jpeg;base64,...",
  "model_id": "hustvl/yolos-tiny",
  "score_threshold": 0.45,
  "max_detections": 25
}
```

## Connect from backend

Set this in `backend/.env` (or environment):

```bash
VLM_COMPUTE_NODE_URL=http://<gx10-ip-or-hostname>:8100
VLM_COMPUTE_NODE_TIMEOUT_SECONDS=30
```

Then restart the backend. `POST /api/vlm/infer` will automatically use the compute node for inference when no local trained artifact is selected.

## Performance knobs (GPU)

These are read by `compute_node` on startup:

```bash
# Default 1 on CUDA: enables AMP autocast for faster inference
COMPUTE_NODE_USE_AUTOCAST=1

# Optional: convert model weights to fp16 (more aggressive, may affect stability)
COMPUTE_NODE_USE_FP16_MODEL=0
```

Current acceleration settings are visible at:

- `GET /health` (returns `device`, `autocast`, `fp16_model`)

## Cosmos engine mode

Switch compute node to Cosmos-compatible vision chat inference:

```bash
VLM_ENGINE=cosmos
NVIDIA_API_KEY=nvapi-...
# Optional overrides:
# COSMOS_API_KEY=...
# COSMOS_BASE_URL=https://integrate.api.nvidia.com/v1
# COSMOS_MODEL=nvidia/cosmos-nemotron-34b
# COSMOS_TIMEOUT_SECONDS=45
```

In Cosmos mode, the node asks for structured findings JSON and maps returned findings into:

- `findings_summary`
- `findings[]`
- `detections[]` (only when Cosmos returns bounding boxes)
