"""Verify a running release image serves its bundled model and useful outputs."""

import argparse
import http.client
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def wait_for_ready(call, attempts=60, delay=1):
    """Docker may accept/reset connections before the application is listening."""
    for attempt in range(attempts):
        try:
            return call("/health/ready")
        except urllib.error.HTTPError as exc:
            if exc.code != 503 or attempt == attempts - 1:
                raise
        except (urllib.error.URLError, OSError, http.client.HTTPException):
            if attempt == attempts - 1:
                raise
        time.sleep(delay)
    raise ValueError("attempts must be positive")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]

    def call(path, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            args.url + path, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=20) as response:
            return json.load(response)

    ready = wait_for_ready(call)
    model = call("/model")
    assert model["name"] == "hist_gradient_boosting" and model["device"] == "cpu"
    request = json.loads((root / "deployment/example_request.json").read_text())
    response = call("/predict", request)
    assert len(response["predictions"]) == 2
    probabilities = [r["rain_probability"] for r in response["predictions"]]
    assert probabilities[0] < 0.1 and probabilities[1] > 0.8
    assert [r["rain_prediction"] for r in response["predictions"]] == ["No", "Yes"]
    assert response["model_sha256"] == model["model_sha256"] == ready["model_sha256"]
    try:
        call("/predict", {"records": []})
        raise AssertionError("Empty requests must be rejected")
    except urllib.error.HTTPError as exc:
        assert exc.code == 422
    for path in ["/docs", "/openapi.json", "/metrics"]:
        with urllib.request.urlopen(args.url + path, timeout=10) as r:
            assert r.status == 200
    report = {
        "status": "passed",
        "model": model,
        "example_predictions": response["predictions"],
        "checks": [
            "readiness",
            "bundled CPU model",
            "two-record inference",
            "empty request rejected",
            "interactive docs",
            "OpenAPI",
            "metrics",
        ],
        "limitations": "Smoke test of the documented example; not a new predictive-accuracy evaluation or universal device guarantee.",
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
