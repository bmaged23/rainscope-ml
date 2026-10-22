# API measurements — 22 October 2026

These are local HTTP measurements on the NVIDIA GTX 1660 Ti (6 GB), WSL2, Python 3.13.16 and PyTorch 2.8.0+cu126. The service loaded the original TabM artifact with validation threshold 0.30. Each run used five warmup calls followed by 100 sequential requests (200 in the Docker-idle rerun) containing two synthetic weather rows. These inputs demonstrate serving and do not measure predictive accuracy.

| Run | p50 | p95 | p99 | Requests/s | Sampled server max RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial GPU API | 21.90 ms | 24.59 ms | 25.99 ms | 44.46 | 837.2 MiB |
| GPU-resident API, concurrent Docker build | 28.23 ms | 42.09 ms | 45.71 ms | 34.89 | 832.0 MiB |
| GPU-resident API, Docker idle (200 requests) | 20.55 ms | 23.42 ms | 28.47 ms | 47.47 | 802.9 MiB |

Raw records: [initial](api_benchmark_gpu.json), [resident](api_benchmark_gpu_resident.json), [Docker-idle rerun](api_benchmark_gpu_quiet.json). The resident run overlapped Docker image building; these are not controlled paired runs. **No latency improvement is established.** Keeping the model on its serving device avoids repeated transfers, but CPU preprocessing, HTTP overhead and host load still affect this small batch. These are not concurrency, production SLA or cold-start measurements. Model-file SHA-256 is included in each report.

The Docker-idle run confirms the current API latency, but does not establish a controlled before/after improvement. Server CPU was approximately 95–96% on a one-core basis during the runs. RSS was sampled between requests and may miss peaks; it includes interpreter and CUDA runtime overhead. GPU VRAM peaks were not measured by this benchmark. No cloud billing was incurred or measured; cost is deliberately null rather than presented as zero.

## Verification

- Pipeline and API tests cover chronological splitting, target leakage, malformed inputs, missing values, unseen categories, probability agreement with the fitted estimator, batch limits and model readiness.
- A neural-serving regression checks CPU/GPU probability agreement where CUDA is available and verifies device residency.
- CPU Docker image built locally; readiness and two-row prediction returned HTTP 200 using the separate histogram-gradient-boosting artifact. Output probabilities were approximately 0.02151 and 0.92988.
- GPU Docker runtime is **not yet verified**. Local CUDA inference is verified. A full CUDA image build and GPU container smoke test remain necessary on a Docker host with enough build storage.

Run commands and limitations: [serving guide](../docs/SERVING.md). CI validates CPU behavior without private training data or CUDA; it does not certify GPU deployment.

## Complete CPU container verification

The release image now builds and bundles the project-trained histogram-gradient-boosting pipeline. It was started with `--network none`, no model/data volumes and a read-only filesystem. Readiness and predictions passed using the bundled synthetic example. The raw training CSV is absent from the final image. A separate replay of the model extracted from the image over all 25,974 test observations reproduced the recorded CPU metrics within 1e-5: accuracy 83.29%, AP 0.7383. See [bundled-model verification](container_model_verification.json).

Docker Hub publication and pull validation are recorded in `deployment/dockerhub_release.json` once completed. This CPU container is distinct from the TabM GPU API and does not claim TabM’s accuracy.

### Published-image pull test

The public `bmaged23/rainscope-ml:cpu` tag was pulled with an empty Docker credential configuration and matched digest `sha256:76c198dd1567004218dd93b0a0299cd8b0039495fe585e2a2b913c58a48d0b36`. A new Linux amd64 container, running as `appuser` with no host mounts, passed readiness, bundled-model prediction, invalid-input rejection, interactive-docs HTTP, OpenAPI and metrics checks. The two examples returned probabilities 0.02151 and 0.92988 with labels No and Yes. [Test record](dockerhub_pull_test.json). This verifies the documented container path on the reference host, not every OS/architecture or machine configuration. Cached content-addressed layers were reused locally; the registry manifest and public anonymous tag access were checked.
