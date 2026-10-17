# Install and reproduce RainScope ML


Use an existing **Python 3.13** interpreter and a project-local virtual environment. GPU experiments need a compatible NVIDIA driver and a PyTorch CUDA build that supports the device; the reference workstation has a GTX 1660 Ti (6 GB).

```bash
git clone https://github.com/bmaged23/rainscope-ml.git
cd rainscope-ml
python3.13 -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the exact reference CPU/application dependencies from `requirements.txt`. For the modern GPU experiments, install the CUDA PyTorch build and modern dependencies documented in `requirements-modern.txt`. A complete reference-environment snapshot is in `requirements-lock.txt`; the smaller base/modern/dev files pin direct dependencies. Do not mix a CUDA 13-only wheel with an unsupported GPU/driver.

```bash
python -m pip install -r requirements.txt
# For the default six-model GPU experiment:
python -m pip install -r requirements-modern.txt
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
python scripts/download_data.py
python -m rainscope.train --root .
python -m streamlit run app.py
# Optional batch CLI (synthetic example inputs, not evaluation data):
python -m rainscope.predict data/example_weather.csv --output predictions.csv
```

Check `config.json` before running: models, thread count, temporal boundaries, threshold candidates, bootstrap repetitions, neural epochs and foundation-model context size are explicit. To run CPU baselines only, remove `tabm` and `tabicl_v2` from the model list. Modern checkpoint downloading occurs on first use and requires internet access. Never add validation/test labels to a foundation model's context.

```bash
python -m pytest
# Optional validation-only processing sensitivity report:
python scripts/check_processing.py
```


To rebuild the README heatmap and metric figures without retraining (after a completed local experiment, which creates the private prediction CSV):

```bash
python scripts/build_readme_figures.py
```
