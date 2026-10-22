"""Train an explicit CPU serving baseline; select threshold on validation only."""

from pathlib import Path
import json
import joblib
from threadpoolctl import threadpool_limits
from rainscope.data import load_data, xy
from rainscope.models import make_model
from rainscope.evaluation import choose_threshold

root = Path(__file__).resolve().parents[1]
cfg = json.loads((root / "config.json").read_text())
parts, audit = load_data(root / "data/raw/weatherAUS.csv", cfg)
x, y = xy(parts["train"])
xv, yv = xy(parts["validation"])
name = "hist_gradient_boosting"
with threadpool_limits(limits=cfg["workers"]):
    model = make_model(name, cfg).fit(x, y)
    threshold = choose_threshold(
        yv, model.predict_proba(xv)[:, 1], cfg["threshold_grid"]
    )
selection = {
    "selected_model": name,
    "threshold": float(threshold),
    "data_sha256": audit["sha256"],
    "purpose": "CPU serving baseline; not the TabM study winner",
}
(root / "artifacts").mkdir(exist_ok=True)
joblib.dump(
    {"model": model, "selection": selection}, root / "artifacts/serving_model.joblib"
)
(root / "artifacts/serving_model.json").write_text(
    json.dumps(selection, indent=2) + "\n"
)
print(json.dumps(selection))
