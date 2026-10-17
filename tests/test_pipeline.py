import numpy as np
import pandas as pd
import pytest
from rainscope.data import NUMERIC, CATEGORICAL, features, load_data, xy
from rainscope.models import make_model
from rainscope.evaluation import choose_threshold, metrics


def observations():
    rows = []
    for date in [
        "2014-12-27",
        "2014-12-28",
        "2014-12-29",
        "2014-12-30",
        "2014-12-31",
        "2015-01-01",
        "2015-01-02",
        "2015-12-28",
        "2015-12-29",
        "2015-12-30",
        "2015-12-31",
        "2016-01-01",
        "2016-01-02",
    ]:
        for i, station in enumerate(["A", "B"]):
            rows.append(
                {
                    "Date": date,
                    **dict.fromkeys(NUMERIC, 1.0),
                    **dict.fromkeys(CATEGORICAL, "N"),
                    "Location": station,
                    "RainToday": "No",
                    "RainTomorrow": "Yes" if i else "No",
                }
            )
    return pd.DataFrame(rows)


def test_split_embargo_and_no_station_date_overlap(tmp_path):
    path = tmp_path / "weather.csv"
    observations().to_csv(path, index=False)
    parts, audit = load_data(
        path,
        {
            "validation_start": "2015-01-01",
            "test_start": "2016-01-01",
            "embargo_days": 2,
        },
    )
    assert (
        parts["train"].Date.max() + pd.Timedelta(days=2)
        < parts["validation"].Date.min()
    )
    assert (
        parts["validation"].Date.max() + pd.Timedelta(days=2) < parts["test"].Date.min()
    )
    assert audit["embargo_rows_excluded"] == 8


def test_target_and_column_order_do_not_leak():
    frame = observations()
    a = features(frame)
    frame.RainTomorrow = "Something else"
    frame["future_target"] = 999
    pd.testing.assert_frame_equal(a, features(frame[list(reversed(frame.columns))]))
    assert "Date" not in a and "RainTomorrow" not in a


def test_unknown_categories_and_missing_values_supported():
    frame = observations()
    x, y = xy(frame)
    cfg = {"seed": 42, "workers": 1, "random_forest_trees": 3, "boosting_iterations": 3}
    for name in ["climatology", "logistic", "random_forest", "hist_gradient_boosting"]:
        model = make_model(name, cfg).fit(x, y)
        q = x.iloc[:2].copy()
        q["Location"] = "UNSEEN"
        q["Humidity3pm"] = np.nan
        p = model.predict_proba(q)[:, 1]
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()


def test_malformed_inputs_rejected():
    frame = observations()
    frame.loc[0, "Date"] = "invalid"
    with pytest.raises(ValueError, match="Date"):
        features(frame)
    frame = observations()
    frame["Humidity9am"] = frame["Humidity9am"].astype(object)
    frame.loc[0, "Humidity9am"] = "nonsense"
    with pytest.raises(ValueError, match="non-numeric"):
        features(frame)


def test_conflicting_duplicates_rejected(tmp_path):
    a = observations()
    b = a.iloc[[0]].copy()
    b["RainTomorrow"] = "Yes"
    path = tmp_path / "data.csv"
    pd.concat([a, b]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Conflicting"):
        load_data(
            path,
            {
                "validation_start": "2015-01-01",
                "test_start": "2016-01-01",
                "embargo_days": 2,
            },
        )


def test_threshold_and_metrics_positive_class():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.4])
    assert choose_threshold(y, p, [0.25, 0.5]) == 0.25
    assert metrics(y, p, 0.25)["confusion_matrix"] == [[2, 0], [0, 2]]
