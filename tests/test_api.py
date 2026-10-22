import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from rainscope.api import create_app
from rainscope.data import features
from rainscope.models import make_model
from test_pipeline import observations


@pytest.fixture
def fitted(tmp_path):
    frame = observations()
    model = make_model("logistic", {"seed": 42}).fit(
        features(frame), (frame.RainTomorrow == "Yes").astype(int)
    )
    path = tmp_path / "model.joblib"
    joblib.dump(
        {"model": model, "selection": {"threshold": 0.3, "selected_model": "logistic"}},
        path,
    )
    records = frame.drop(columns="RainTomorrow").iloc[:2].to_dict("records")
    return path, records, model


def test_predictions_match_pipeline_and_metrics(fitted):
    path, records, model = fitted
    with TestClient(create_app(path)) as client:
        assert client.get("/health/ready").status_code == 200
        response = client.post("/predict", json={"records": records})
        assert response.status_code == 200
        result = response.json()
        expected = model.predict_proba(features(pd.DataFrame(records)))[:, 1]
        assert [p["rain_probability"] for p in result["predictions"]] == pytest.approx(
            expected
        )
        assert len(result["model_sha256"]) == 64
        assert client.get("/model").json()["name"] == "logistic"
        assert "rainscope_predicted_rows_total 2" in client.get("/metrics").text


def test_invalid_requests_and_batch_limit(fitted):
    path, records, _ = fitted
    with TestClient(create_app(path)) as client:
        for body in [
            {"records": []},
            {"records": records * 129},
            {"records": [{"Date": "bad"}]},
        ]:
            assert client.post("/predict", json=body).status_code == 422
        records[0]["RainTomorrow"] = "Yes"
        assert client.post("/predict", json={"records": records}).status_code == 422


def test_missing_model_liveness_is_not_readiness(tmp_path, fitted):
    with TestClient(create_app(tmp_path / "absent.joblib")) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503
        assert client.post("/predict", json={"records": fitted[1]}).status_code == 503


def test_nullable_measurements_and_unseen_station(fitted):
    path, records, _ = fitted
    records[0]["Humidity3pm"] = None
    records[0]["Location"] = "Unknown station"
    with TestClient(create_app(path)) as client:
        assert client.post("/predict", json={"records": records}).status_code == 200


def test_model_failure_is_counted_and_service_recovers(fitted, monkeypatch):
    path, records, _ = fitted
    app = create_app(path)
    with TestClient(app) as client:
        model = app.state.bundle["model"]
        original = model.predict_proba

        def broken(_):
            raise RuntimeError("private implementation detail")

        monkeypatch.setattr(model, "predict_proba", broken)
        response = client.post("/predict", json={"records": records})
        assert response.status_code == 500
        assert response.json() == {"detail": "Prediction failed"}
        monkeypatch.setattr(model, "predict_proba", original)
        assert client.post("/predict", json={"records": records}).status_code == 200
        metrics = client.get("/metrics").text
        assert "rainscope_prediction_requests_total 2" in metrics
        assert "rainscope_prediction_failures_total 1" in metrics
        assert "rainscope_predicted_rows_total 2" in metrics
