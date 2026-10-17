"""Batch scoring with a locally trained, trusted pipeline."""

import argparse
from pathlib import Path
import joblib
import pandas as pd
from .data import features


def predict(frame, artifact):
    bundle = joblib.load(artifact)
    p = bundle["model"].predict_proba(features(frame))[:, 1]
    out = frame[[c for c in ["Date", "Location"] if c in frame]].copy()
    out["rain_probability"] = p
    out["rain_prediction"] = [
        "Yes" if v >= bundle["selection"]["threshold"] else "No" for v in p
    ]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--model", type=Path, default=Path("artifacts/selected_model.joblib")
    )
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    args = parser.parse_args()
    predict(pd.read_csv(args.input), args.model).to_csv(args.output, index=False)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
