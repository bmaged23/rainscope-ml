"""Verify saved imputation/scaling/categories against the frozen training recipe."""

import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from rainscope.data import CATEGORICAL, load_data, xy

root = Path(__file__).resolve().parents[1]
selection = json.loads((root / "reports/selection.json").read_text())
cfg = selection["config"]
parts, audit = load_data(root / "data/raw/weatherAUS.csv", cfg)
assert audit["sha256"] == selection["data_sha256"]
x, y = xy(parts["train"])
bundle = joblib.load(root / "artifacts/all_models.joblib")
results = {}
for name, model in bundle["models"].items():
    fitted = x
    if name == "tabicl_v2" and len(y) > cfg["tabicl_context_rows"]:
        ix, _ = train_test_split(
            np.arange(len(y)),
            train_size=cfg["tabicl_context_rows"],
            stratify=y,
            random_state=cfg["seed"],
        )
        fitted = x.iloc[ix]
    preprocessor = (
        model.preprocessor
        if hasattr(model, "preprocessor")
        else model.named_steps["preprocess"]
    )
    numeric = [c for c in x.columns if c not in CATEGORICAL]
    imputer = preprocessor.named_transformers_["numeric"].named_steps["impute"]
    np.testing.assert_allclose(
        imputer.statistics_,
        fitted[numeric].median().to_numpy(),
        equal_nan=True,
        atol=1e-10,
    )
    scaler = preprocessor.named_transformers_["numeric"].named_steps["scale"]
    assert int(scaler.n_samples_seen_) == len(fitted)
    encoder = preprocessor.named_transformers_["category"]
    for category, values in zip(CATEGORICAL, encoder.categories_):
        np.testing.assert_array_equal(values, np.sort(fitted[category].unique()))
    results[name] = {
        "verified": True,
        "fitted_rows": len(fitted),
        "checks": [
            "numeric imputer medians match training recipe",
            "scaler sample count matches training rows",
            "category vocabularies match training recipe",
        ],
    }
output = root / "reports/model_verification/preprocessing_audit.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
