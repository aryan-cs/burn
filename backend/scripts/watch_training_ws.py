from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib import error, request

import websockets

# ---------------------------------------------------------------------------
# Hardcoded configuration
# ---------------------------------------------------------------------------
BASE_URL = "http://127.0.0.1:8000"

# If True, this script starts jobs itself. If False, it watches jobs started
# by other scripts/tools and auto-attaches to the latest unseen job.
START_NEW_JOB = False

# Always run and keep watching for new jobs.
ALWAYS_WATCH = True

# If START_NEW_JOB is True, this delay is used between finished jobs.
AUTO_RESTART_DELAY_SECONDS = 2.0

# Only used when START_NEW_JOB is True.
GRAPH_FILE: Path | None = None
TRAIN_EPOCHS = 20
TRAIN_BATCH_SIZE = 64
TRAIN_LR = 0.001

# Polling behavior.
POLL_INTERVAL_SECONDS = 1.0
MAX_STATUS_WAIT_SECONDS = 600

# Websocket behavior.
WS_OPEN_TIMEOUT_SECONDS = 60


class Color:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"


USE_COLOR = os.getenv("NO_COLOR") is None


def c(text: str, color: str) -> str:
    if not USE_COLOR:
        return text
    return f"{color}{text}{Color.RESET}"


def now_ts() -> str:
    return time.strftime("%H:%M:%S")


def log_info(msg: str) -> None:
    print(f"{c('[INFO]', Color.BLUE)} {c(now_ts(), Color.CYAN)} {msg}")


def log_epoch(msg: str) -> None:
    print(f"{c('[EPOCH]', Color.MAGENTA)} {c(now_ts(), Color.CYAN)} {msg}")


def log_success(msg: str) -> None:
    print(f"{c('[DONE]', Color.GREEN)} {c(now_ts(), Color.CYAN)} {msg}")


def log_warn(msg: str) -> None:
    print(f"{c('[WARN]', Color.YELLOW)} {c(now_ts(), Color.CYAN)} {msg}")


def log_error(msg: str) -> None:
    print(f"{c('[ERROR]', Color.RED)} {c(now_ts(), Color.CYAN)} {msg}")


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
            "epochs": TRAIN_EPOCHS,
            "batchSize": TRAIN_BATCH_SIZE,
            "optimizer": "adam",
            "learningRate": TRAIN_LR,
            "loss": "cross_entropy",
        },
    }


def load_payload() -> dict[str, Any]:
    if GRAPH_FILE is None:
        return default_graph_payload()
    return json.loads(GRAPH_FILE.read_text(encoding="utf-8"))


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
        details = body if body else str(exc)
        raise RuntimeError(f"{method} {url} failed ({exc.code}): {details}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Cannot reach backend at {url}: {exc.reason}") from exc


def get_job_status(job_id: str) -> dict[str, Any]:
    return http_json("GET", f"{BASE_URL}/api/model/status?job_id={job_id}")


def get_latest_job() -> dict[str, Any]:
    return http_json("GET", f"{BASE_URL}/api/model/latest")


def wait_for_terminal_status(job_id: str) -> dict[str, Any]:
    start = time.time()
    while True:
        status = get_job_status(job_id)
        if status.get("terminal", False):
            return status
        if time.time() - start > MAX_STATUS_WAIT_SECONDS:
            raise RuntimeError(f"Timed out waiting for terminal status on job {job_id}")
        time.sleep(POLL_INTERVAL_SECONDS)


def ws_base_url(http_base: str) -> str:
    if http_base.startswith("https://"):
        return "wss://" + http_base.removeprefix("https://")
    if http_base.startswith("http://"):
        return "ws://" + http_base.removeprefix("http://")
    raise ValueError("BASE_URL must start with http:// or https://")


def print_ws_message(message: dict[str, Any]) -> None:
    msg_type = message.get("type")
    if msg_type == "epoch_update":
        epoch = message.get("epoch")
        train_loss = float(message.get("train_loss", message.get("loss", 0.0)))
        train_acc = float(message.get("train_accuracy", message.get("accuracy", 0.0)))
        test_loss = float(message.get("test_loss", message.get("loss", 0.0)))
        test_acc = float(message.get("test_accuracy", message.get("accuracy", 0.0)))
        log_epoch(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )
        return

    if msg_type == "training_done":
        final_train_loss = float(message.get("final_train_loss", message.get("final_loss", 0.0)))
        final_train_acc = float(
            message.get("final_train_accuracy", message.get("final_accuracy", 0.0))
        )
        final_test_loss = float(message.get("final_test_loss", message.get("final_loss", 0.0)))
        final_test_acc = float(message.get("final_test_accuracy", message.get("final_accuracy", 0.0)))
        log_success(
            f"training_done final_train_loss={final_train_loss:.4f} "
            f"final_train_acc={final_train_acc:.4f} "
            f"final_test_loss={final_test_loss:.4f} "
            f"final_test_acc={final_test_acc:.4f}"
        )
        return

    if msg_type == "error":
        log_error(f"training_error: {message.get('message')}")
        return

    log_info(f"ws_message: {json.dumps(message)}")


async def stream_job_updates(job_id: str) -> dict[str, Any] | None:
    ws_url = f"{ws_base_url(BASE_URL)}/ws/training/{job_id}"
    log_info(f"Connecting websocket: {ws_url}")

    try:
        async with websockets.connect(ws_url, open_timeout=WS_OPEN_TIMEOUT_SECONDS) as websocket:
            while True:
                raw = await websocket.recv()
                message = json.loads(raw)
                print_ws_message(message)

                if message.get("type") in {"training_done", "error"}:
                    return message

    except Exception as exc:
        log_warn(f"WebSocket interrupted: {exc}")
        return None


def start_training_job() -> str:
    payload = load_payload()
    train_resp = http_json("POST", f"{BASE_URL}/api/model/train", payload)
    job_id = train_resp["job_id"]
    log_info(f"Started job_id={job_id} status={train_resp.get('status')}")
    return job_id


def wait_for_next_unseen_job(seen_job_ids: set[str]) -> str:
    while True:
        latest = get_latest_job()
        latest_job_id = latest.get("job_id")
        if latest_job_id and latest_job_id not in seen_job_ids:
            log_info(
                f"Discovered new job_id={latest_job_id} "
                f"status={latest.get('status')} terminal={latest.get('terminal')}"
            )
            return str(latest_job_id)

        log_info("No new jobs yet; watcher is still running...")
        time.sleep(POLL_INTERVAL_SECONDS)


def run_single_job(job_id: str) -> int:
    _ = asyncio.run(stream_job_updates(job_id))

    final_status = wait_for_terminal_status(job_id)
    status = final_status.get("status")
    log_info(f"Final status: {json.dumps(final_status)}")

    if status == "failed":
        err = final_status.get("error")
        log_error("Job failed and watcher reached terminal state for this job.")
        if err:
            log_error(f"Failure reason: {err}")
        return 1

    log_success("Job completed/stopped and watcher reached terminal state for this job.")
    return 0


def main() -> int:
    seen_job_ids: set[str] = set()

    while True:
        try:
            if START_NEW_JOB:
                job_id = start_training_job()
            else:
                job_id = wait_for_next_unseen_job(seen_job_ids)

            seen_job_ids.add(job_id)
            result = run_single_job(job_id)

            if not ALWAYS_WATCH:
                return result

            if START_NEW_JOB:
                log_info(f"Restarting watcher with a new training job in {AUTO_RESTART_DELAY_SECONDS}s")
                time.sleep(AUTO_RESTART_DELAY_SECONDS)
            else:
                # Keep daemon alive and wait for future jobs.
                log_info("Continuing to watch for the next job...")

        except KeyboardInterrupt:
            log_warn("Watcher interrupted by user. Exiting.")
            return 130
        except Exception as exc:
            log_error(f"Watcher loop error: {exc}")
            if not ALWAYS_WATCH:
                return 1
            time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
