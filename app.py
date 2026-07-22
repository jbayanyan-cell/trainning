"""
AgriShield Training Service — Railway
Triggered by admin Start Training. Uses PHP APIs (no direct MySQL from Railway).
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path

import requests
from flask import Flask, jsonify, request

app = Flask(__name__)

PHP_API_BASE = os.getenv(
    "PHP_API_BASE",
    "https://agrishield.bccbsis.com/api/training",
).rstrip("/")
TRAINING_SCRIPT = os.getenv("TRAINING_SCRIPT", "train.py")
DEFAULT_EPOCHS = int(os.getenv("DEFAULT_EPOCHS", "3"))
DEFAULT_BATCH_SIZE = int(os.getenv("DEFAULT_BATCH_SIZE", "8"))
TRAINING_API_KEY = os.getenv("TRAINING_API_KEY", "")
# Railway has no GPUs; CPU jobs need a long wall clock. Override with TRAINING_TIMEOUT_SEC.
TRAINING_TIMEOUT_SEC = int(os.getenv("TRAINING_TIMEOUT_SEC", str(6 * 3600)))
# Soft caps so admin typos (e.g. epochs=16) don't OOM / never finish on CPU.
MAX_EPOCHS_CPU = int(os.getenv("MAX_EPOCHS_CPU", "5"))
MAX_BATCH_CPU = int(os.getenv("MAX_BATCH_CPU", "8"))


def _headers():
    h = {"Content-Type": "application/json", "Accept": "application/json"}
    if TRAINING_API_KEY:
        h["X-API-Key"] = TRAINING_API_KEY
    return h


def php_get(path: str, params: dict | None = None, timeout: int = 20):
    url = f"{PHP_API_BASE}/{path.lstrip('/')}"
    return requests.get(url, params=params or {}, headers=_headers(), timeout=timeout)


def php_post(path: str, payload: dict, timeout: int = 30):
    url = f"{PHP_API_BASE}/{path.lstrip('/')}"
    return requests.post(url, json=payload, headers=_headers(), timeout=timeout)


def get_training_job(job_id: int):
    try:
        r = php_get("get_job.php", {"job_id": job_id})
        if r.status_code != 200:
            return None
        data = r.json()
        if data.get("success") and data.get("job"):
            return data["job"]
        return None
    except Exception as exc:
        print(f"[ERROR] get_training_job: {exc}", flush=True)
        return None


def update_job_status(job_id: int, status: str, message: str | None = None):
    payload = {"job_id": job_id, "status": status}
    if message:
        payload["message"] = str(message)[:500]
    try:
        php_post("update_status.php", payload)
    except Exception as exc:
        print(f"[WARN] update_job_status failed: {exc}", flush=True)


def log_to_php(job_id: int, level: str, message: str):
    try:
        php_post(
            "add_log.php",
            {
                "job_id": job_id,
                "level": level,
                "message": str(message)[:1000],
            },
        )
    except Exception as exc:
        print(f"[WARN] log_to_php failed: {exc}", flush=True)


def parse_epochs_batch(job: dict) -> tuple[int, int]:
    epochs = job.get("epochs")
    batch_size = job.get("batch_size")
    config = job.get("training_config") or {}
    if isinstance(config, str):
        try:
            config = json.loads(config) if config.strip() else {}
        except json.JSONDecodeError:
            config = {}
    if not isinstance(config, dict):
        config = {}

    if epochs is None:
        epochs = config.get("epochs")
    if batch_size is None:
        batch_size = config.get("batch_size")

    try:
        epochs = int(epochs) if epochs not in (None, "") else DEFAULT_EPOCHS
    except (TypeError, ValueError):
        epochs = DEFAULT_EPOCHS
    try:
        batch_size = int(batch_size) if batch_size not in (None, "") else DEFAULT_BATCH_SIZE
    except (TypeError, ValueError):
        batch_size = DEFAULT_BATCH_SIZE

    # Detect swapped values from older admin bugs
    if batch_size > 32 and epochs < 20:
        epochs, batch_size = batch_size, epochs

    return epochs, batch_size


def _tail_useful_error(text: str, limit: int = 800) -> str:
    """Prefer the real traceback; ignore gdown progress noise at the start of stderr."""
    if not text:
        return "unknown error (no output; process may have been OOM-killed)"
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    useful = [
        ln
        for ln in lines
        if not ln.startswith("Downloading")
        and not ln.startswith("From:")
        and not ln.startswith("To:")
        and "MB/s" not in ln
        and "█" not in ln
        and "▉" not in ln
    ]
    joined = "\n".join(useful if useful else lines)
    if "Traceback" in joined:
        idx = joined.rfind("Traceback")
        joined = joined[idx:]
    else:
        joined = "\n".join(joined.splitlines()[-40:])
    joined = joined.strip() or (
        "Process exited non-zero (often OOM on Railway). Try batch_size=4-8 and epochs=3."
    )
    return joined[:limit]


def run_training(job_id: int):
    try:
        job = get_training_job(job_id)
        if not job:
            print(f"[ERROR] Job {job_id} not found", flush=True)
            return

        epochs, batch_size = parse_epochs_batch(job)

        if epochs > MAX_EPOCHS_CPU:
            log_to_php(
                job_id,
                "WARNING",
                f"epochs={epochs} capped to {MAX_EPOCHS_CPU} for Railway CPU (set MAX_EPOCHS_CPU to override)",
            )
            epochs = MAX_EPOCHS_CPU
        if batch_size > MAX_BATCH_CPU:
            log_to_php(
                job_id,
                "WARNING",
                f"batch_size={batch_size} capped to {MAX_BATCH_CPU} for Railway CPU memory",
            )
            batch_size = MAX_BATCH_CPU

        update_job_status(job_id, "running")
        log_to_php(
            job_id,
            "INFO",
            f"Training started on Railway CPU (epochs={epochs}, batch={batch_size}, timeout={TRAINING_TIMEOUT_SEC}s). Railway has no GPU.",
        )

        script_dir = Path(__file__).resolve().parent
        script_path = script_dir / TRAINING_SCRIPT
        if not script_path.exists():
            msg = f"Training script not found: {script_path}"
            update_job_status(job_id, "failed", msg)
            log_to_php(job_id, "ERROR", msg)
            return

        env = os.environ.copy()
        env["PHP_API_BASE"] = PHP_API_BASE
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("OMP_NUM_THREADS", "2")
        env.setdefault("MKL_NUM_THREADS", "2")
        env.setdefault("OMP_WAIT_POLICY", "PASSIVE")

        cmd = [
            "python",
            str(script_path),
            "--job_id",
            str(job_id),
            "--epochs",
            str(epochs),
            "--batch_size",
            str(batch_size),
        ]
        print(f"[INFO] Running: {' '.join(cmd)}", flush=True)
        log_to_php(job_id, "INFO", f"Running train.py epochs={epochs} batch={batch_size}")

        log_path = script_dir / f"training_job_{job_id}.log"
        with open(log_path, "w", encoding="utf-8", errors="replace") as logf:
            result = subprocess.run(
                cmd,
                stdout=logf,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=TRAINING_TIMEOUT_SEC,
                cwd=str(script_dir),
                env=env,
            )

        log_text = ""
        try:
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass

        if result.returncode == 0:
            update_job_status(job_id, "completed")
            log_to_php(job_id, "INFO", "Training completed successfully")
        else:
            err = _tail_useful_error(log_text)
            if result.returncode < 0:
                err = (
                    f"Process killed (signal {-result.returncode}). "
                    f"Likely out of memory. Use batch_size=4 and epochs=3. Detail: {err}"
                )
            update_job_status(job_id, "failed", err)
            log_to_php(job_id, "ERROR", f"Training failed (exit={result.returncode}): {err}")
    except subprocess.TimeoutExpired:
        hours = max(1, TRAINING_TIMEOUT_SEC // 3600)
        msg = f"Training timeout (exceeded {hours} hours on Railway CPU)"
        update_job_status(job_id, "failed", msg)
        log_to_php(job_id, "ERROR", msg)
    except Exception as exc:
        msg = str(exc)[:500]
        update_job_status(job_id, "failed", msg)
        log_to_php(job_id, "ERROR", f"Training error: {msg}")


@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-API-Key"
    return response


@app.route("/", methods=["GET"])
def root():
    return jsonify(
        {
            "service": "AgriShield Training Service",
            "platform": "railway",
            "version": "2.0.0",
            "php_api_base": PHP_API_BASE,
            "endpoints": {
                "health": "/health",
                "train": "/train (POST)",
                "status": "/status/<job_id> (GET)",
            },
        }
    )


@app.route("/health", methods=["GET", "OPTIONS"])
def health():
    if request.method == "OPTIONS":
        return "", 204
    try:
        r = php_get("get_job.php", {"job_id": 0}, timeout=12)
        php_ok = r.status_code in (200, 400, 404)
        from dataset_source import get_dataset_zip_url

        ds_url = get_dataset_zip_url()
        env_raw = {
            "DATASET_ZIP_URL": bool((os.getenv("DATASET_ZIP_URL") or "").strip()),
            "DATASET_URL": bool((os.getenv("DATASET_URL") or "").strip()),
            "DATASET_DRIVE_URL": bool((os.getenv("DATASET_DRIVE_URL") or "").strip()),
        }
        payload = {
            "status": "ok" if php_ok else "error",
            "service": "training-service",
            "platform": "railway",
            "php_api": "connected" if php_ok else "unreachable",
            "php_api_base": PHP_API_BASE,
            "php_http_code": r.status_code,
            "dataset_zip_configured": bool(ds_url),
            "dataset_from_env": any(env_raw.values()),
            "dataset_env_flags": env_raw,
            "dataset_using_builtin_default": bool(ds_url) and not any(env_raw.values()),
            "dataset_zip_url_preview": (ds_url[:70] + "...") if ds_url else None,
            "code_version": "cpu-safe-v10",
            "cuda_available": False,
            "gpu_note": "Railway does not offer GPU instances; training runs on CPU.",
            "training_timeout_sec": TRAINING_TIMEOUT_SEC,
            "default_epochs": DEFAULT_EPOCHS,
            "default_batch_size": DEFAULT_BATCH_SIZE,
            "max_epochs_cpu": MAX_EPOCHS_CPU,
            "max_batch_cpu": MAX_BATCH_CPU,
        }
        return jsonify(payload), (200 if php_ok else 503)
    except Exception as exc:
        return (
            jsonify(
                {
                    "status": "error",
                    "service": "training-service",
                    "platform": "railway",
                    "php_api": "unreachable",
                    "message": str(exc)[:200],
                }
            ),
            503,
        )


@app.route("/train", methods=["POST", "OPTIONS"])
def start_training():
    if request.method == "OPTIONS":
        return "", 204
    try:
        data = request.get_json(silent=True) or {}
        job_id = data.get("job_id")
        if not job_id:
            return jsonify({"success": False, "message": "job_id required"}), 400

        job_id = int(job_id)
        job = get_training_job(job_id)
        if not job:
            return jsonify({"success": False, "message": "Job not found"}), 404

        status = (job.get("status") or "").lower()
        if status not in ("pending", "running"):
            return (
                jsonify(
                    {
                        "success": False,
                        "message": f"Job status is {status}, expected pending",
                    }
                ),
                400,
            )

        thread = threading.Thread(target=run_training, args=(job_id,), daemon=True)
        thread.start()

        return jsonify(
            {
                "success": True,
                "message": "Training started on Railway",
                "job_id": job_id,
                "platform": "railway",
            }
        )
    except Exception as exc:
        return jsonify({"success": False, "message": str(exc)}), 500


@app.route("/status/<int:job_id>", methods=["GET"])
def get_status(job_id: int):
    try:
        job = get_training_job(job_id)
        if not job:
            return jsonify({"success": False, "message": "Job not found"}), 404
        return jsonify(
            {
                "success": True,
                "job_id": job_id,
                "status": job.get("status"),
                "completed_at": job.get("completed_at"),
                "error_message": job.get("error_message"),
            }
        )
    except Exception as exc:
        return jsonify({"success": False, "message": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
