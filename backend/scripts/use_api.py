from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any
from urllib import error, request

import websockets

# ---------------------------------------------------------------------------
# Hardcoded configuration
# ---------------------------------------------------------------------------
BASE_URL = "http://127.0.0.1:8000"
GRAPH_FILE: Path | None = None
EPOCHS = 3
BATCH_SIZE = 64
LEARNING_RATE = 0.001
STREAM_WS = True
STOP_AFTER_NO_WS = False
RUN_INFERENCE = True
INFERENCE_INPUT_FILE: Path | None = None
WAIT_TIMEOUT_SECONDS = 300
POLL_INTERVAL_SECONDS = 1.0
OUTPUT_DIR = Path("client_outputs")


def default_graph_payload() -> dict[str, Any]:
    return {
        "nodes": [
            {"id": "node_1", "type": "Input", "config": {"shape": [1, 28, 28]}},
            {"id": "node_2", "type": "Flatten", "config": {}},
            {"id": "node_3", "type": "Dense", "config": {"units": 128, "activation": "relu"}},
            {"id": "node_4", "type": "Output", "config": {"num_classes": 10, "activation": "softmax"}},
        ],
        "edges": [
            {"id": "e1", "source": "node_1", "target": "node_2"},
            {"id": "e2", "source": "node_2", "target": "node_3"},
            {"id": "e3", "source": "node_3", "target": "node_4"},
        ],
        "training": {
            "dataset": "mnist",
            "epochs": EPOCHS,
            "batchSize": BATCH_SIZE,
            "optimizer": "adam",
            "learningRate": LEARNING_RATE,
            "loss": "cross_entropy",
        },
    }


def http_json(method: str, url: str, payload: dict[str, Any] | None = None) -> Any:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = request.Request(url=url, method=method, data=data, headers=headers)
    try:
        with request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")
            if not body:
                return None
            return json.loads(body)
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        message = body if body else str(exc)
        raise RuntimeError(f"{method} {url} failed ({exc.code}): {message}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Cannot reach API at {url}: {exc.reason}") from exc


def http_text(method: str, url: str) -> str:
    req = request.Request(url=url, method=method, headers={"Accept": "*/*"})
    try:
        with request.urlopen(req, timeout=120) as resp:
            return resp.read().decode("utf-8")
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        message = body if body else str(exc)
        raise RuntimeError(f"{method} {url} failed ({exc.code}): {message}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Cannot reach API at {url}: {exc.reason}") from exc


def http_bytes(method: str, url: str) -> bytes:
    req = request.Request(url=url, method=method, headers={"Accept": "*/*"})
    try:
        with request.urlopen(req, timeout=120) as resp:
            return resp.read()
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        message = body if body else str(exc)
        raise RuntimeError(f"{method} {url} failed ({exc.code}): {message}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Cannot reach API at {url}: {exc.reason}") from exc


def to_ws_base(http_base: str) -> str:
    if http_base.startswith("https://"):
        return "wss://" + http_base.removeprefix("https://")
    if http_base.startswith("http://"):
        return "ws://" + http_base.removeprefix("http://")
    raise ValueError("BASE_URL must start with http:// or https://")


async def stream_training(ws_url: str) -> dict[str, Any] | None:
    try:
        print(f"Streaming training updates from {ws_url}")
        async with websockets.connect(ws_url, open_timeout=60) as websocket:  # type: ignore[attr-defined]
            while True:
                raw = await websocket.recv()
                msg = json.loads(raw)
                mtype = msg.get("type")

                if mtype == "epoch_update":
                    train_loss = float(msg.get("train_loss", msg.get("loss", 0.0)))
                    train_acc = float(msg.get("train_accuracy", msg.get("accuracy", 0.0)))
                    test_loss = float(msg.get("test_loss", msg.get("loss", 0.0)))
                    test_acc = float(msg.get("test_accuracy", msg.get("accuracy", 0.0)))
                    print(
                        f"epoch={msg.get('epoch')} "
                        f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                        f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
                    )
                    continue

                if mtype == "training_done":
                    final_train_loss = float(msg.get("final_train_loss", msg.get("final_loss", 0.0)))
                    final_train_acc = float(
                        msg.get("final_train_accuracy", msg.get("final_accuracy", 0.0))
                    )
                    final_test_loss = float(msg.get("final_test_loss", msg.get("final_loss", 0.0)))
                    final_test_acc = float(
                        msg.get("final_test_accuracy", msg.get("final_accuracy", 0.0))
                    )
                    print(
                        "training_done "
                        f"final_train_loss={final_train_loss:.4f} "
                        f"final_train_acc={final_train_acc:.4f} "
                        f"final_test_loss={final_test_loss:.4f} "
                        f"final_test_acc={final_test_acc:.4f}"
                    )
                    return msg

                if mtype == "error":
                    print(f"training_error: {msg.get('message')}")
                    return msg

                print(f"ws_message: {msg}")
    except Exception as exc:
        print(f"websocket stream interrupted: {exc}")
        return None


def load_payload() -> dict[str, Any]:
    if GRAPH_FILE is None:
        payload = default_graph_payload()
    else:
        payload = json.loads(GRAPH_FILE.read_text(encoding="utf-8"))
        if "training" not in payload:
            payload["training"] = default_graph_payload()["training"]

    payload["training"]["epochs"] = EPOCHS
    payload["training"]["batchSize"] = BATCH_SIZE
    payload["training"]["learningRate"] = LEARNING_RATE
    return payload


def load_inference_input() -> list:
    if INFERENCE_INPUT_FILE is None:
        return [[[0.0 for _ in range(28)] for _ in range(28)]]
    return json.loads(INFERENCE_INPUT_FILE.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def wait_for_terminal_state(base_url: str, job_id: str, timeout_s: int, poll_interval_s: float) -> dict[str, Any]:
    start = time.time()
    while True:
        status = http_json("GET", f"{base_url}/api/model/status?job_id={job_id}")
        if status.get("terminal", False):
            return status
        if time.time() - start >= timeout_s:
            raise RuntimeError(f"Timed out waiting for job {job_id} to finish after {timeout_s}s")
        time.sleep(poll_interval_s)


def run() -> int:
    payload = load_payload()

    print("1) Validating graph...")
    validation = http_json("POST", f"{BASE_URL}/api/model/validate", payload)
    print(json.dumps(validation, indent=2))
    if not validation.get("valid", False):
        print("Validation failed. Aborting.")
        return 1

    print("2) Compiling graph...")
    compiled = http_json("POST", f"{BASE_URL}/api/model/compile", payload)
    print(f"param_count={compiled['summary']['param_count']}")
    warnings = compiled.get("warnings", [])
    if warnings:
        print("warnings:")
        for warning in warnings:
            print(f"- {warning}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_text(OUTPUT_DIR / "generated_model.py", compiled["python_source"])
    print(f"saved: {OUTPUT_DIR / 'generated_model.py'}")

    print("3) Starting training job...")
    train_resp = http_json("POST", f"{BASE_URL}/api/model/train", payload)
    job_id = train_resp["job_id"]
    print(f"job_id={job_id}, status={train_resp.get('status')}")

    ws_result: dict[str, Any] | None = None
    if STREAM_WS:
        ws_base = to_ws_base(BASE_URL)
        ws_result = asyncio.run(stream_training(f"{ws_base}/ws/training/{job_id}"))

    final_status: dict[str, Any]
    if ws_result is None and STOP_AFTER_NO_WS:
        print("No websocket stream result. Stopping the job explicitly...")
        stop_resp = http_json("POST", f"{BASE_URL}/api/model/stop", {"job_id": job_id})
        print(f"stop_response={stop_resp}")
        final_status = wait_for_terminal_state(
            base_url=BASE_URL,
            job_id=job_id,
            timeout_s=WAIT_TIMEOUT_SECONDS,
            poll_interval_s=POLL_INTERVAL_SECONDS,
        )
    elif ws_result is None:
        print("No websocket stream result. Polling /api/model/status until terminal...")
        final_status = wait_for_terminal_state(
            base_url=BASE_URL,
            job_id=job_id,
            timeout_s=WAIT_TIMEOUT_SECONDS,
            poll_interval_s=POLL_INTERVAL_SECONDS,
        )
    else:
        # WebSocket returned a terminal message, but we still poll status to get
        # canonical terminal state from the backend job registry.
        final_status = wait_for_terminal_state(
            base_url=BASE_URL,
            job_id=job_id,
            timeout_s=WAIT_TIMEOUT_SECONDS,
            poll_interval_s=POLL_INTERVAL_SECONDS,
        )

    print(f"job_status={final_status}")

    print("4) Exporting generated Python...")
    try:
        py_text = http_text("GET", f"{BASE_URL}/api/model/export?job_id={job_id}&format=py")
    except RuntimeError as exc:
        print(f"export py failed, falling back to compiled source: {exc}")
        py_text = compiled["python_source"]
    write_text(OUTPUT_DIR / f"{job_id}.py", py_text)
    print(f"saved: {OUTPUT_DIR / f'{job_id}.py'}")

    if final_status.get("status") == "failed":
        print("Training failed; skipping .pt export and inference.")
        return 1

    print("5) Exporting trained .pt artifact (if available)...")
    try:
        pt_bytes = http_bytes("GET", f"{BASE_URL}/api/model/export?job_id={job_id}&format=pt")
        pt_path = OUTPUT_DIR / f"{job_id}.pt"
        pt_path.write_bytes(pt_bytes)
        print(f"saved: {pt_path}")
    except RuntimeError as exc:
        print(f".pt export not available yet: {exc}")

    if RUN_INFERENCE:
        print("6) Running inference...")
        infer_inputs = load_inference_input()
        infer_resp = http_json(
            "POST",
            f"{BASE_URL}/api/model/infer",
            {"job_id": job_id, "inputs": infer_inputs, "return_probabilities": True},
        )
        print(json.dumps(infer_resp, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
