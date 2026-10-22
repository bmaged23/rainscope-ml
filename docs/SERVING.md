# Run the prediction service

The research comparison selects TabM. The main serving route uses that trusted local artifact on CUDA. The ready-to-run public CPU image bundles this project’s fitted histogram-gradient-boosting pipeline. Git contains code and aggregate results, not model binaries or raw data. The image builds its CPU model from the pinned source data and deletes the downloaded raw observations in the same build step. No TabICL checkpoint or context rows are distributed.

## Local GPU API

Follow [GETTING_STARTED.md](GETTING_STARTED.md) to install the base and modern dependencies, obtain data and train the models first. Then, from the repository root:

```bash
python -m pip install -r requirements-api.txt
RAINSCOPE_DEVICE=cuda OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -m uvicorn rainscope.api:app --host 127.0.0.1 --port 8000 --workers 1
```

Open http://127.0.0.1:8000/docs for interactive API documentation. In another terminal:

```bash
curl -fsS http://127.0.0.1:8000/health/ready
curl -fsS http://127.0.0.1:8000/model
curl -fsS http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' --data-binary @deployment/example_request.json
curl -fsS http://127.0.0.1:8000/metrics
```

The checked-in request contains two **synthetic** observations, for demonstrating inference rather than evaluating accuracy. Responses preserve input row order and include the model file's SHA-256 and validation-selected threshold. Missing measurements must be explicit JSON `null`; required columns cannot be omitted. Target labels and unexpected columns are rejected. Requests contain 1–256 records. Schema errors return 422; a missing model returns 503; inference failures return a generic 500 with details only in server logs.

## GPU container — primary deployment

Docker Desktop must enable WSL integration and GPU access (or configure NVIDIA Container Toolkit on a Linux host). The host needs a compatible NVIDIA driver for the CUDA 12.6 PyTorch runtime. Build from the repository root:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up --build -d
curl -fsS http://127.0.0.1:8000/model
docker compose -f compose.yaml -f compose.gpu.yaml logs api
docker compose -f compose.yaml -f compose.gpu.yaml down
```

**Verification status:** native GPU serving and CPU Docker prediction are tested locally; the GPU container is configured but its runtime verification is pending.

The `/model` response must report `"device": "cuda"`. Explicit CUDA mode fails startup if CUDA is unavailable rather than silently switching to CPU. This image includes the CUDA runtime and is considerably larger than the CPU image. For the GPU source-build route, the local `artifacts` folder is mounted read-only. The public CPU image instead includes its fitted CPU model. The service runs as an unprivileged user, with a read-only root filesystem, dropped capabilities and a localhost-only port. The Docker health check tests model readiness, not just a running process.

## Ready-to-run Docker Hub image — CPU

Docker Desktop (Linux containers) or Docker Engine is required. The release targets **Linux amd64 / x86-64**, including compatible Windows machines using Docker Desktop/WSL2. Other architectures have not been verified. No GPU, Python installation, dataset download, model mount or training is needed at startup.

```bash
docker pull bmaged23/rainscope-ml:cpu
docker run -d --name rainscope -p 127.0.0.1:8000:8000 bmaged23/rainscope-ml:cpu
```

Open **http://localhost:8000/docs**. Expand **POST /predict**, choose **Try it out**, and paste the example JSON from this repository’s `deployment/example_request.json`. From a cloned repository, the same request is:

```bash
curl -fsS http://localhost:8000/health/ready
curl -fsS http://localhost:8000/model
curl -fsS http://localhost:8000/predict -H 'Content-Type: application/json' --data-binary @deployment/example_request.json
```

If you only pulled the image, it also contains the example. This command sends it from inside the running container:

```bash
docker exec rainscope python -c "import urllib.request; p=open('/app/deployment/example_request.json','rb').read(); r=urllib.request.Request('http://127.0.0.1:8000/predict',data=p,headers={'Content-Type':'application/json'}); print(urllib.request.urlopen(r).read().decode())"
```

Expect two predictions, **No** and **Yes**, for the two synthetic examples. This verifies inference behavior, not accuracy on real observations. The model is **histogram gradient boosting**, not the TabM GPU winner. It uses the frozen training/validation protocol and threshold 0.30.

```bash
docker logs rainscope
docker stop rainscope
docker rm rainscope
```

The public image is an API service, not the Streamlit dashboard or a live-weather downloader. Input observations must be supplied by the caller. `latest` refers to the same CPU release; use the `cpu` tag to make the intended runtime explicit.

[Docker Hub repository](https://hub.docker.com/r/bmaged23/rainscope-ml). Publication and pull verification are recorded in [serving results](../reports/SERVING_RESULTS.md); do not assume a pending tag is available until that record says it passed.

### Build the complete CPU image yourself

From the repository root:

```bash
docker compose up --build -d
docker compose down
```

The build downloads and hash-verifies the source dataset, trains only the CPU serving baseline, retains its fitted model, and removes the raw CSV before the build layer is saved. Internet access is required during the build and image pull, but not during ordinary inference. The model metadata is included at `/app/artifacts/serving_model.json`.

## Configuration and monitoring

| Setting | Default / main GPU configuration | Purpose |
| --- | --- | --- |
| `RAINSCOPE_MODEL` | `artifacts/selected_model.joblib` locally; bundled CPU model in the public image | Trusted artifact to load once at startup |
| `RAINSCOPE_DEVICE` | `auto` locally; `cuda` in GPU Compose | TabM serving device; auto selects CUDA if available |
| Uvicorn workers | 1 | Avoid loading duplicate model copies into VRAM |
| `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` | 2 in containers | Bound CPU preprocessing threads |
| Batch size | 1–256 records | Bound rows per prediction request |
| Threshold | Artifact's validation-selected value | No per-request threshold tuning |

TabM stays resident on the selected serving device, avoiding a CPU transfer after each request. The training/evaluation lifecycle remains unchanged. Inference calls are serialized within one process to protect the estimator's device state; this is not a high-concurrency throughput claim.

`/health/live` reports process liveness; `/health/ready` is 503 without a model. `/metrics` exports Prometheus-text counters for prediction requests entering the handler, failures, successfully predicted rows, and cumulative handler seconds (including lock waiting). Schema-rejected 422 requests are handled before these counters. Counters reset on process restart and are per-worker. HTTP access logs provide status codes; application exceptions go to stderr. There is no persistent metrics database or dashboard yet.

The API has no authentication or rate limiter. It is intended for local demonstration; public deployment requires an authenticated TLS gateway, request-body/concurrency limits and an operator-controlled artifact source. Loading joblib executes Python code: never load an untrusted artifact or expose an upload route for it.

## Reproduce service measurements

With the server running, use its PID printed by Uvicorn:

```bash
python scripts/benchmark_api.py --pid SERVER_PID --output reports/api_benchmark_local.json
# Optional explicit host-cost assumption, in EUR/hour:
python scripts/benchmark_api.py --pid SERVER_PID --hourly-cost 0.50
```

The benchmark makes five warmup calls, then 100 sequential real HTTP calls with two synthetic records per request. It fails on an unsuccessful response. It records p50/p95/p99 latency, throughput, sampled server RSS and CPU time. CPU percentage uses one core as 100%; RSS samples can miss peaks. Cost stays null unless supplied: a provided hourly rate yields an idealized saturated cost per 1,000 requests, excluding idle capacity, storage and networking. No paid cloud service has been deployed.

See [serving measurements](../reports/SERVING_RESULTS.md). Historical model-fit timings in the main README measure another workload and must not be compared directly with HTTP request latency.

Implementation references: [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/) and [API testing](https://fastapi.tiangolo.com/tutorial/testing/).
