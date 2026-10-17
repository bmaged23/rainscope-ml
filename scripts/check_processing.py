"""Validation-only sensitivity study; never changes final model selection."""

from pathlib import Path
import json
import pandas as pd
from rainscope.data import load_data, xy
from rainscope.models import make_model
from rainscope.evaluation import metrics
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
cfg = json.loads((ROOT / "config.json").read_text())
parts, _ = load_data(ROOT / "data/raw/weatherAUS.csv", cfg)
x, y = xy(parts["train"])
v, z = xy(parts["validation"])
derived = {"MonthSin", "MonthCos", "PressureChange", "HumidityChange"}
rows = []
with threadpool_limits(limits=cfg["workers"]):
    for name, drop, indicators in [
        ("configured_features", set(), True),
        ("without_missing_indicators", set(), False),
        ("without_derived_features", derived, True),
        (
            "without_high_missing_numeric_features",
            set(x.columns[x.isna().mean() > 0.4]),
            True,
        ),
    ]:
        model = make_model("logistic", cfg)
        pre = model.named_steps["preprocess"]
        pre.transformers = [
            (label, transformer, [c for c in cols if c not in drop])
            for label, transformer, cols in pre.transformers
        ]
        pre.transformers[0][1].set_params(impute__add_indicator=indicators)
        model.fit(x, y)
        p = model.predict_proba(v)[:, 1]
        m = metrics(z, p, 0.5)
        rows.append(
            {
                "variant": name,
                "dropped_features": ", ".join(sorted(drop)),
                "validation_average_precision": m["average_precision"],
                "validation_brier": m["brier"],
                "validation_f1_at_0_5": m["f1"],
            }
        )
        print(name, rows[-1], flush=True)
pd.DataFrame(rows).to_csv(ROOT / "reports/preprocessing_sensitivity.csv", index=False)
print(
    "Descriptive logistic-regression ablation on validation only; final selection unchanged."
)
