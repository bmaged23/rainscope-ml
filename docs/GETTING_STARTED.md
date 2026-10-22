# From clone to a running RainScope ML application

Use **Python 3.13**, Git and internet access for installation/data downloading. The reference platform is Linux/Ubuntu under WSL2. Start with the CPU route below; it requires no NVIDIA GPU or PyTorch. Use the GPU section to reproduce the six-model study shown in the README.

## 1. Clone the repository

```bash
git clone https://github.com/bmaged23/rainscope-ml.git
cd rainscope-ml
```

Run all following commands from this directory.

## 2. Create and activate a virtual environment

Linux / macOS / Windows with WSL:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

For native Windows PowerShell, use these environment commands instead. Native Windows has not been independently verified; WSL2 is the tested Windows route.

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Keep the environment activated for the remaining commands. If `python3.13` / `py -3.13` is missing, install Python 3.13 first. On Linux distributions that separate the venv package, install their matching Python 3.13 venv support if environment creation reports that ensurepip is unavailable.

## 3. Install the requirements and project

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

`requirements.txt` pins the direct CPU, dashboard and API dependencies. `requirements-lock.txt` records the full reference environment, including CUDA dependencies; it is not the portable CPU installation command.

## 4. Download the dataset

```bash
python scripts/download_data.py
```

This creates `data/raw/weatherAUS.csv` and verifies its SHA-256 against the recorded source snapshot. Raw data and trained models are not included in a Git clone.

## 5. Train the CPU models

```bash
python -m rainscope.train --root . --config configs/cpu.json
```

This runs climatology, logistic regression, random forest and histogram gradient boosting. It fits preprocessing on training data, chooses the model and threshold using validation data, then evaluates the final chronological test set. It creates:

- `artifacts/selected_model.joblib` — fitted prediction pipeline.
- `artifacts/all_models.joblib` — all models from this run.
- `reports/` — metrics, plots, selection and resource measurements.

Wait for training to finish successfully before trying predictions. A local CPU run creates its own selection/results; it does not reproduce the README's six-model TabM experiment. Retraining writes the local reports and selected artifact, replacing the corresponding checked-in report snapshots in your working directory.

## 6. Launch the application

```bash
python -m streamlit run app.py
```

Open **http://localhost:8501**. Explore the model comparison and methodology tabs. In **Batch prediction**, upload `data/example_weather.csv` to test the locally trained model. These are synthetic example inputs, not accuracy-evaluation samples.

Stop the application with **Ctrl+C**.

## 7. Optional: run the REST API

Keep the virtual environment activated:

```bash
python -m uvicorn rainscope.api:app --host 127.0.0.1 --port 8000 --workers 1
```

Open **http://localhost:8000/docs**, expand **POST /predict**, click **Try it out**, and use the JSON from `deployment/example_request.json`.

From another terminal, you can also run:

```bash
curl -fsS http://localhost:8000/health/ready
curl -fsS http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @deployment/example_request.json
```

The readiness endpoint should return HTTP 200, and prediction should return one probability and label per input row. A 503 means the model artifact is missing; finish step 5 first. In PowerShell, use `curl.exe` for the curl examples.

## Optional: reproduce all six models with an NVIDIA GPU

Complete steps 1–4, then install the pinned modern-model dependencies and run the original configuration:

```bash
python -m pip install -r requirements-modern.txt
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -m rainscope.train --root . --config config.json
python -m streamlit run app.py
```

The reference hardware is a **GTX 1660 Ti with 6 GB VRAM**, using **PyTorch 2.8.0+cu126** and a compatible NVIDIA driver. Confirm the CUDA check prints `True` before the six-model run. Do not install a CUDA build unsupported by your driver/device. TabICLv2 downloads its checkpoint on first use and its prediction stage is much slower than the other models.

For GPU API serving after training, set `RAINSCOPE_DEVICE=cuda` before the Uvicorn command (in PowerShell: `$env:RAINSCOPE_DEVICE="cuda"`). See [SERVING.md](SERVING.md) for device settings, Docker, metrics and benchmark commands.

## Optional: batch prediction without a web application

```bash
python -m rainscope.predict data/example_weather.csv --output predictions.csv
```

## Tests and model evaluation

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

After a completed **six-model** experiment, verify the saved models and preprocessing without retraining or changing thresholds:

```bash
python scripts/verify_models.py
python scripts/verify_preprocessing.py
python scripts/build_verification_report.py
python scripts/build_model_visualizations.py
```

The six-model verification scripts require modern dependencies and all six saved estimators; do not use them after the four-model CPU-only run. Results go to `reports/model_verification/`. This is a replay of the same historical holdout, not a new independent benchmark.

Other optional analysis commands:

```bash
python scripts/check_processing.py
python scripts/build_readme_figures.py
```

Figure generation requires a completed local experiment, including its private prediction CSV. The published six-model comparison charts require the six-model experiment and verification outputs.

## Common setup problems

| Problem | What to check |
| --- | --- |
| `No module named rainscope` | Activate `.venv` and run `python -m pip install -e . --no-deps` from the repository root. |
| Missing CSV | Run step 4 and confirm `data/raw/weatherAUS.csv` exists. |
| Missing model / API returns 503 | Finish training in step 5 or the GPU experiment first. |
| CUDA unavailable | Check the host driver and WSL GPU access; use the CPU configuration until CUDA is available. |
| Port already in use | Stop the existing service or choose `--server.port 8502` for Streamlit / `--port 8001` for Uvicorn. |
| Shell activation failed | Use the environment command for your shell, or invoke `.venv/bin/python` directly on Linux/WSL. |

Docker usage and its current verification status are documented separately in [SERVING.md](SERVING.md). The ready-to-run CPU image is documented in the serving guide; its publication/pull status is recorded in `reports/SERVING_RESULTS.md`.
