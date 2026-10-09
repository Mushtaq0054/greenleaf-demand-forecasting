"""
GreenLeaf Grocery - Daily Service Latency & Health Monitoring Script
Task 5: Production monitoring script for scheduled GitHub Actions.

Pings the live demand forecasting service endpoints (/health and /predict),
measures request latency in milliseconds, checks HTTP response codes,
and appends execution logs to reports/monitoring_logs.csv.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timezone
import urllib.request
import urllib.error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("GreenLeafMonitor")

# Configuration
DEFAULT_API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
LOG_FILE_PATH = os.environ.get(
    "MONITORING_LOG_FILE",
    os.path.join(os.path.dirname(__file__), "reports", "monitoring_logs.csv"),
)

SAMPLE_PREDICT_PAYLOAD = {
    "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    "sku_id": "SKU_001",
    "product_name": "Organic Honeycrisp Apples",
    "category": "Fruit",
    "unit_price": 2.99,
    "inventory_level": 85,
    "promotion": "No",
}


def ensure_log_file_header(filepath: str) -> None:
    """Ensures reports directory exists and CSV header is present."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    if not os.path.exists(filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("timestamp,endpoint,status_code,latency_ms,result,notes\n")
        logger.info("Initialized monitoring log CSV at: %s", filepath)


def log_metric(
    filepath: str,
    endpoint: str,
    status_code: int,
    latency_ms: float,
    result: str,
    notes: str = "",
) -> None:
    """Appends an individual monitoring record to the CSV file."""
    timestamp_iso = datetime.now(timezone.utc).isoformat()
    record_line = f"{timestamp_iso},{endpoint},{status_code},{latency_ms:.2f},{result},{notes}\n"
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(record_line)


def ping_endpoint(
    url: str,
    method: str = "GET",
    payload: dict = None,
    timeout: float = 30.0,
) -> tuple:
    """
    Sends HTTP request using Python standard library, measuring round-trip latency.
    Returns (status_code, latency_ms, response_data, error_message).
    """
    data_bytes = None
    headers = {"Accept": "application/json"}

    if payload is not None:
        data_bytes = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data_bytes, headers=headers, method=method)

    start_time = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            body = response.read().decode("utf-8")
            status_code = response.getcode()
            parsed_json = json.loads(body) if body else {}
            return status_code, latency_ms, parsed_json, None
    except urllib.error.HTTPError as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return exc.code, latency_ms, None, f"HTTPError: {exc.reason}"
    except urllib.error.URLError as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return 0, latency_ms, None, f"URLError: {exc.reason}"
    except Exception as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return 0, latency_ms, None, f"Exception: {str(exc)}"


def run_daily_monitoring(api_base_url: str = DEFAULT_API_URL) -> bool:
    """
    Orchestrates daily monitoring checks:
    1. Checks /health endpoint
    2. Sends sample inference to /predict endpoint
    3. Logs metrics to CSV
    """
    logger.info("=== Starting Daily Service Health & Latency Monitoring ===")
    logger.info("Target Service URL: %s", api_base_url)

    ensure_log_file_header(LOG_FILE_PATH)
    all_passed = True

    # Check 1: Health Endpoint
    health_url = f"{api_base_url}/health"
    status_code, latency_ms, data, err = ping_endpoint(health_url, method="GET")

    if status_code == 200 and data and data.get("status") == "healthy":
        logger.info(
            "Health check PASSED: Status %d, Latency %.2f ms",
            status_code,
            latency_ms,
        )
        log_metric(LOG_FILE_PATH, "/health", status_code, latency_ms, "healthy", "model_loaded=True")
    else:
        all_passed = False
        notes = err or f"unexpected_status_{status_code}"
        logger.error(
            "Health check FAILED: Status %d, Latency %.2f ms, Error: %s",
            status_code,
            latency_ms,
            notes,
        )
        log_metric(LOG_FILE_PATH, "/health", status_code, latency_ms, "failed", notes)

    # Check 2: Prediction Endpoint
    predict_url = f"{api_base_url}/predict"
    status_code, latency_ms, data, err = ping_endpoint(
        predict_url, method="POST", payload=SAMPLE_PREDICT_PAYLOAD
    )

    if status_code == 200 and data and "predicted_demand" in data:
        pred_val = data["predicted_demand"]
        logger.info(
            "Prediction check PASSED: Status %d, Latency %.2f ms, Predicted: %.2f units",
            status_code,
            latency_ms,
            pred_val,
        )
        log_metric(
            LOG_FILE_PATH,
            "/predict",
            status_code,
            latency_ms,
            "success",
            f"predicted_demand={pred_val}",
        )
    else:
        all_passed = False
        notes = err or f"unexpected_status_{status_code}"
        logger.error(
            "Prediction check FAILED: Status %d, Latency %.2f ms, Error: %s",
            status_code,
            latency_ms,
            notes,
        )
        log_metric(LOG_FILE_PATH, "/predict", status_code, latency_ms, "failed", notes)

    logger.info("Monitoring completed. Log file updated at: %s", LOG_FILE_PATH)
    return all_passed


if __name__ == "__main__":
    success = run_daily_monitoring()
    sys.exit(0 if success else 1)
