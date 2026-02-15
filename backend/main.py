from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent
load_dotenv(BACKEND_DIR / ".env", override=False)
load_dotenv(REPO_ROOT / ".env", override=False)

from routers.deploy import router as deploy_router
from routers.datasets import router as datasets_router
from routers.model import router as model_router
from routers.rf_datasets import router as rf_datasets_router
from routers.rf_model import router as rf_model_router
from routers.rf_websocket import router as rf_websocket_router
from routers.vlm_model import router as vlm_model_router
from routers.vlm_websocket import router as vlm_websocket_router
from routers.websocket import router as websocket_router
from routers.ml_model import router as ml_model_router
from routers.ml_websocket import router as ml_ws_router
from routers.ai_coach import router as ai_coach_router, warn_if_ai_provider_keys_missing

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    artifacts_dir = Path(__file__).resolve().parent / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    warn_if_ai_provider_keys_missing()
    compute_node_url = os.getenv("VLM_COMPUTE_NODE_URL", "").strip()
    if compute_node_url:
        logger.info("VLM compute node configured at %s", compute_node_url)
    else:
        logger.info("VLM compute node is not configured; local VLM runtime will be used")
    yield


app = FastAPI(title="MLCanvas Backend", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(model_router)
app.include_router(deploy_router)
app.include_router(datasets_router)
app.include_router(websocket_router)
app.include_router(ml_model_router)
app.include_router(ml_ws_router)
app.include_router(ai_coach_router)
app.include_router(rf_model_router)
app.include_router(rf_datasets_router)
app.include_router(rf_websocket_router)
app.include_router(vlm_model_router)
app.include_router(vlm_websocket_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
