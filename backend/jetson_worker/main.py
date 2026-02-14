from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from jetson_worker.routers.training import router as training_router


@asynccontextmanager
async def lifespan(_: FastAPI):
    artifacts_dir = Path(__file__).resolve().parent / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="MLCanvas Jetson Worker", version="0.1.0", lifespan=lifespan)
app.include_router(training_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("jetson_worker.main:app", host="0.0.0.0", port=8001, reload=False)
