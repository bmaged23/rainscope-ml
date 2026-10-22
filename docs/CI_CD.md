# CI and container delivery

## Continuous integration

`.github/workflows/tests.yml` runs on pushes and pull requests:

1. Install the project in a clean Ubuntu runner with Python 3.13.
2. Run Ruff checks and pipeline/API tests. The optional neural regression skips if modern dependencies are absent; CI does not claim to verify a physical GPU.
3. Build the complete CPU image from source, including the model trained during the build from hash-verified data.
4. Start the image without a model/data volume and test readiness, predictions, validation errors, OpenAPI documentation and metrics.

The container test uses `scripts/smoke_test_api.py`. It checks actual predictions rather than declaring success just because the server process started.

**Verified:** [CI](https://github.com/bmaged23/rainscope-ml/actions/runs/39844918468) passed code tests and complete-container prediction checks on code commit `0549f6b`. [Docker Hub delivery](https://github.com/bmaged23/rainscope-ml/actions/runs/39844973128) passed build, tests, authentication, publication and post-pull inference on the same commit. See `deployment/ci_delivery_verification.json` for the image digest and scope. GPU container verification remains separate and unfinished.

## Continuous delivery to Docker Hub

`.github/workflows/docker-release.yml` runs manually via **Actions → Publish verified CPU image → Run workflow**, or when a version tag beginning with `v` is pushed.

The workflow:

1. Checks publication credentials are configured.
2. Runs lint and tests.
3. Builds and smoke-tests the complete CPU image.
4. Authenticates to Docker Hub.
5. Pushes `bmaged23/rainscope-ml:cpu` and `bmaged23/rainscope-ml:cpu-<full-git-commit>`.
6. Pulls the published commit tag and smoke-tests it again.

The stable `cpu` tag updates only after pre-publication tests pass. A unique commit tag links the image to source. Post-publication verification can still fail because of registry/network issues; inspect the run before treating a release as successful.

This is image delivery, not deployment of a publicly hosted inference server. It does not publish an unverified GPU image.

### One-time maintainer setup

The local Docker Desktop sign-in does not authenticate GitHub-hosted runners. For this personal Docker Hub repository:

1. Create a Docker Hub personal access token with the permissions needed to push images (read/write; delete is unnecessary).
2. Open this GitHub repository’s **Settings → Secrets and variables → Actions → New repository secret**.
3. Store the token under **`DOCKERHUB_TOKEN`**. Do not commit it or paste it into the README or an issue.
4. Push the workflow and run it manually once. Confirm all stages pass before relying on tag-triggered releases.

The `DOCKERHUB_TOKEN` repository secret is now configured with Read & Write access only, expiring on 20 April 2027. The token shared in chat was disabled. No token values are stored in this repository. The release workflow has passed end to end; its verified run is linked above. The initial public CPU image was published and tested from Docker Desktop.

User instructions require no credentials: [pull and run the public image](SERVING.md#ready-to-run-docker-hub-image--cpu).

References: [Docker GitHub Actions authentication and publishing](https://docs.docker.com/build/ci/github-actions/push-multi-registries/) and [Docker access tokens](https://docs.docker.com/security/access-tokens/).
