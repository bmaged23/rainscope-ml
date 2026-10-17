"""Schema checks and chronological partitions, before learned preprocessing."""

from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

CATEGORICAL = ["Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday"]
NUMERIC = [
    "MinTemp",
    "MaxTemp",
    "Rainfall",
    "Evaporation",
    "Sunshine",
    "WindGustSpeed",
    "WindSpeed9am",
    "WindSpeed3pm",
    "Humidity9am",
    "Humidity3pm",
    "Pressure9am",
    "Pressure3pm",
    "Cloud9am",
    "Cloud3pm",
    "Temp9am",
    "Temp3pm",
]
FEATURES = (
    NUMERIC + ["MonthSin", "MonthCos", "PressureChange", "HumidityChange"] + CATEGORICAL
)


def features(frame):
    required = ["Date"] + NUMERIC + CATEGORICAL
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    x = frame.copy()
    dates = pd.to_datetime(x["Date"], format="ISO8601", errors="coerce")
    if dates.isna().any():
        raise ValueError("Date contains missing or invalid values")
    for c in NUMERIC:
        converted = pd.to_numeric(x[c], errors="coerce")
        if (x[c].notna() & converted.isna()).any():
            raise ValueError(f"{c} contains non-numeric values")
        x[c] = converted.replace([np.inf, -np.inf], np.nan)
    for c in CATEGORICAL:
        x[c] = x[c].fillna("__MISSING__").astype(str)
    month = dates.dt.month
    x["MonthSin"] = np.sin(2 * np.pi * month / 12)
    x["MonthCos"] = np.cos(2 * np.pi * month / 12)
    x["PressureChange"] = x["Pressure3pm"] - x["Pressure9am"]
    x["HumidityChange"] = x["Humidity3pm"] - x["Humidity9am"]
    return x[FEATURES]


def load_data(path, config):
    path = Path(path)
    raw = pd.read_csv(path)
    audit = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "raw_rows": len(raw),
        "missing_percent": (raw.isna().mean() * 100).round(3).to_dict(),
    }
    if "RainTomorrow" not in raw:
        raise ValueError("Training data requires RainTomorrow")
    valid = raw["RainTomorrow"].isin(["Yes", "No"])
    audit["excluded_missing_or_invalid_targets"] = int((~valid).sum())
    df = raw.loc[valid].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    df = df.drop_duplicates()
    if df.duplicated(["Date", "Location"]).any():
        raise ValueError(
            "Conflicting duplicate Date/Location records require manual resolution"
        )
    audit["exact_duplicates_removed"] = int(valid.sum() - len(df))
    df = df.sort_values(["Date", "Location"]).reset_index(drop=True)
    features(df)  # validate schema before fitting
    v, t = pd.Timestamp(config["validation_start"]), pd.Timestamp(config["test_start"])
    gap = pd.Timedelta(days=config["embargo_days"])
    if gap < pd.Timedelta(days=1) or v >= t:
        raise ValueError("Require at least one embargo day and ordered split dates")
    masks = {
        "train": df.Date < v - gap,
        "validation": (df.Date >= v) & (df.Date < t - gap),
        "test": df.Date >= t,
    }
    parts = {name: df.loc[mask].copy() for name, mask in masks.items()}
    audit["embargo_rows_excluded"] = len(df) - sum(map(len, parts.values()))
    audit["splits"] = {}
    for name, part in parts.items():
        if len(part) == 0 or part.RainTomorrow.nunique() != 2:
            raise ValueError(f"{name} must contain both target classes")
        audit["splits"][name] = {
            "rows": len(part),
            "start": str(part.Date.min().date()),
            "end": str(part.Date.max().date()),
            "rain_rate": float((part.RainTomorrow == "Yes").mean()),
            "stations": int(part.Location.nunique()),
        }
    audit["unseen_validation_stations"] = sorted(
        set(parts["validation"].Location) - set(parts["train"].Location)
    )
    audit["unseen_test_stations"] = sorted(
        set(parts["test"].Location) - set(parts["train"].Location)
    )
    return parts, audit


def xy(frame):
    return features(frame), (frame.RainTomorrow == "Yes").astype(int).to_numpy()
