"""Read recorded reports without loading a model; predict only after local training."""

from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
from rainscope.data import features

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="RainScope ML", page_icon="🌦️", layout="wide")
st.markdown(
    """<style>.stApp{background:#f5f8fc}h1,h2,h3{color:#15394b}.block-container{padding-top:2rem}</style>""",
    unsafe_allow_html=True,
)
st.caption("BOLA MAGED RAMSES NESSIM · TABULAR MACHINE LEARNING")
st.title("RainScope ML")
st.write(
    "Australian rainfall classification · chronological evaluation · modern tabular models"
)
st.info(
    "Historical research study, not an operational weather forecast. All displayed performance comes from saved experiment reports."
)
report = ROOT / "reports"
selection_path = report / "selection.json"
if not selection_path.exists():
    st.warning(
        "No experiment report yet. Download the data and run: python -m rainscope.train"
    )
    st.stop()
selection = json.loads(selection_path.read_text())
metrics = (
    json.loads((report / "test_metrics.json").read_text())
    if (report / "test_metrics.json").exists()
    else None
)
if metrics is None:
    st.warning("Experiment is still running; final test report is not ready.")
    st.stop()
a, b, c, d = st.columns(4)
a.metric("Selected by validation", selection["selected_model"])
b.metric("Test average precision", f"{metrics['average_precision']:.3f}")
c.metric("Test rain-class F1", f"{metrics['f1']:.3f}")
d.metric("Decision threshold", f"{selection['threshold']:.2f}")
comparison, quality, inference = st.tabs(
    ["Model comparison", "Data & methodology", "Batch prediction"]
)
with comparison:
    st.subheader("Validation-first model selection")
    st.dataframe(
        pd.read_csv(report / "validation_leaderboard.csv"),
        width="stretch",
        hide_index=True,
    )
    st.subheader("Final holdout diagnostics")
    st.dataframe(
        pd.read_csv(report / "test_leaderboard.csv"), width="stretch", hide_index=True
    )
    for name in ["model_comparison.png", "feature_importance.png"]:
        if (report / "figures" / name).exists():
            st.image(str(report / "figures" / name), width="stretch")
    st.caption(
        "TabICLv2 uses a limited training context; check training_rows before comparing. Results are a single-seed, bounded-budget study."
    )
    resources = json.loads((report / "resources.json").read_text())
    table = []
    for model, phases in resources.items():
        for phase, m in phases.items():
            table.append({"model": model, "phase": phase, **m})
    st.subheader("Measured runtime and memory")
    st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True)
with quality:
    audit = json.loads((report / "data_audit.json").read_text())
    st.json(audit["splits"])
    st.image(str(report / "figures/data_quality.png"), width="stretch")
    st.write(
        "Preprocessing is fitted on training only. Two-day embargoes separate 2015 validation and 2016+ test from preceding records. Station overlap is allowed across time; this is not a spatial holdout."
    )
    st.write(
        "Missing targets are excluded. Numeric features use median imputation and missingness indicators. Raw Date and RainTomorrow are never predictor columns. Model and threshold are selected on validation only."
    )
    st.subheader("Station-level holdout analysis")
    st.dataframe(
        pd.read_csv(report / "station_metrics.csv"), width="stretch", hide_index=True
    )
    st.caption(
        "Station slices describe errors; they are not permission to retune against the final test set."
    )
with inference:
    st.write(
        "After local training, upload a CSV with the original weather feature columns and Date. RainTomorrow is optional and ignored as an input. Unknown categories and missing numeric values are supported."
    )
    artifact = ROOT / "artifacts/selected_model.joblib"
    upload = st.file_uploader("Weather observations", type=["csv"])
    if not artifact.exists():
        st.warning(
            "Train locally to create artifacts/selected_model.joblib. Model artifacts and raw observations are not committed."
        )
    if upload is not None and artifact.exists():
        try:
            frame = pd.read_csv(upload)
            with st.spinner("Generating probabilities…"):
                bundle = joblib.load(artifact)
                p = bundle["model"].predict_proba(features(frame))[:, 1]
            out = frame[[c for c in ["Date", "Location"] if c in frame]].copy()
            out["rain_probability"] = p
            out["rain_prediction"] = [
                "Yes" if v >= bundle["selection"]["threshold"] else "No" for v in p
            ]
            st.dataframe(out, width="stretch")
            st.download_button(
                "Download predictions",
                out.to_csv(index=False),
                "rain_predictions.csv",
                "text/csv",
            )
        except (ValueError, KeyError, TypeError) as exc:
            st.error(str(exc))
st.caption(
    "Source dataset lineage and original author credit: docs/ATTRIBUTION.md. Results measure a historical holdout, not future weather reliability."
)
