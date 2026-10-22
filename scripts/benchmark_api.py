"""Benchmark a running HTTP API; save measured latency and optional server resources."""

import argparse
import json
import platform
import time
import urllib.request
from pathlib import Path
import numpy as np
import psutil


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--requests", type=int, default=100)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--pid", type=int, help="API server PID for RSS/CPU sampling")
    p.add_argument(
        "--hourly-cost",
        type=float,
        help="User-supplied host cost in EUR/hour; not a measured cloud price",
    )
    p.add_argument("--output", type=Path, default=Path("reports/api_benchmark.json"))
    args = p.parse_args()
    if args.requests < 1 or args.warmup < 0:
        p.error("Require positive requests and nonnegative warmup")
    root = Path(__file__).resolve().parents[1]
    payload = (root / "deployment/example_request.json").read_bytes()
    process = psutil.Process(args.pid) if args.pid else None
    samples, rss = [], []

    def call():
        request = urllib.request.Request(
            args.url + "/predict",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
            assert len(result["predictions"]) == 2
        return time.perf_counter() - started

    with urllib.request.urlopen(args.url + "/model") as response:
        model = json.load(response)
    for _ in range(args.warmup):
        call()
    cpu_before = sum(process.cpu_times()[:2]) if process else None
    started = time.perf_counter()
    for _ in range(args.requests):
        samples.append(call())
        if process:
            rss.append(process.memory_info().rss)
    elapsed = time.perf_counter() - started
    cpu_seconds = sum(process.cpu_times()[:2]) - cpu_before if process else None
    report = {
        "measured_at_utc": __import__("datetime")
        .datetime.now(__import__("datetime").timezone.utc)
        .isoformat(),
        "protocol": "sequential HTTP, two synthetic records/request, warmup excluded; no accuracy claim",
        "requests": args.requests,
        "warmup": args.warmup,
        "model": model,
        "platform": platform.platform(),
        "latency_ms": {
            k: float(np.percentile(samples, v) * 1000)
            for k, v in [("p50", 50), ("p95", 95), ("p99", 99)]
        },
        "requests_per_second": args.requests / elapsed,
        "sampled_server_max_rss_mib": max(rss) / 2**20 if rss else None,
        "server_cpu_percent_one_core_basis": cpu_seconds / elapsed * 100
        if process
        else None,
        "estimated_eur_per_1000_requests_at_measured_saturation": args.hourly_cost
        / 3600
        * elapsed
        / args.requests
        * 1000
        if args.hourly_cost is not None
        else None,
        "limitations": "RSS sampled between requests can miss peaks. CPU may exceed 100% with multiple cores. Cost excludes idle time, storage and network. No GPU memory measurement here.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
