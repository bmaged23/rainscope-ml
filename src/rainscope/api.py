"""Local serving of a trusted fitted pipeline; no training or artifact uploads."""

from contextlib import asynccontextmanager
from datetime import date
import hashlib
import logging
import os
from pathlib import Path
from threading import Lock
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field, create_model

from .data import CATEGORICAL, NUMERIC, features

logger = logging.getLogger(__name__)


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


WeatherRecord = create_model(
    "WeatherRecord",
    __base__=StrictRecord,
    Date=(date, ...),
    **{name: (float | None, ...) for name in NUMERIC},
    **{name: (str | None, ...) for name in CATEGORICAL},
)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[WeatherRecord] = Field(min_length=1, max_length=256)


def create_app(artifact=None):
    path = Path(
        artifact or os.getenv("RAINSCOPE_MODEL", "artifacts/selected_model.joblib")
    )
    inference_lock, metrics_lock = Lock(), Lock()
    counters = {"requests": 0, "failures": 0, "rows": 0, "seconds": 0.0}

    @asynccontextmanager
    async def lifespan(app):
        app.state.bundle = None
        app.state.model_sha256 = None
        if path.is_file():
            # joblib can execute code: only operator-provisioned, trusted artifacts.
            bundle = joblib.load(path)
            threshold = float(bundle["selection"]["threshold"])
            if not np.isfinite(threshold) or not 0 <= threshold <= 1:
                raise ValueError("Invalid artifact threshold")
            if hasattr(bundle["model"], "set_serving_device"):
                import torch

                device = os.getenv("RAINSCOPE_DEVICE", "auto")
                if device == "auto":
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                bundle["model"].set_serving_device(device)
            app.state.bundle = bundle
            app.state.model_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            logger.warning("Model unavailable; readiness and prediction return 503")
        yield
        app.state.bundle = None

    app = FastAPI(title="RainScope ML API", version="0.1.0", lifespan=lifespan)

    @app.get("/health/live")
    def live():
        return {"status": "alive"}

    @app.get("/health/ready")
    def ready():
        if app.state.bundle is None:
            raise HTTPException(503, "Model not loaded")
        return {"status": "ready", "model_sha256": app.state.model_sha256}

    @app.get("/model")
    def model_info():
        ready()
        selection = app.state.bundle["selection"]
        return {
            "name": selection["selected_model"],
            "threshold": selection["threshold"],
            "model_sha256": app.state.model_sha256,
            "max_batch_rows": 256,
            "device": getattr(app.state.bundle["model"], "serving_device", "cpu"),
        }

    @app.post("/predict")
    def predict(request: PredictRequest):
        started = perf_counter()
        with metrics_lock:
            counters["requests"] += 1
        try:
            ready()
            frame = pd.DataFrame([r.model_dump(mode="json") for r in request.records])
            with inference_lock:
                probabilities = np.asarray(
                    app.state.bundle["model"].predict_proba(features(frame))
                )[:, 1]
            if (
                not np.isfinite(probabilities).all()
                or ((probabilities < 0) | (probabilities > 1)).any()
            ):
                raise RuntimeError("Model returned invalid probabilities")
            threshold = app.state.bundle["selection"]["threshold"]
            with metrics_lock:
                counters["rows"] += len(frame)
            return {
                "model_sha256": app.state.model_sha256,
                "threshold": threshold,
                "predictions": [
                    {
                        "rain_probability": float(p),
                        "rain_prediction": "Yes" if p >= threshold else "No",
                    }
                    for p in probabilities
                ],
            }
        except HTTPException:
            with metrics_lock:
                counters["failures"] += 1
            raise
        except Exception:
            with metrics_lock:
                counters["failures"] += 1
            logger.exception("Prediction failed")
            raise HTTPException(500, "Prediction failed") from None
        finally:
            with metrics_lock:
                counters["seconds"] += perf_counter() - started

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics():
        with metrics_lock:
            values = dict(counters)
        names = {
            "requests": "rainscope_prediction_requests_total",
            "failures": "rainscope_prediction_failures_total",
            "rows": "rainscope_predicted_rows_total",
            "seconds": "rainscope_prediction_duration_seconds_total",
        }
        return "".join(
            f"# TYPE {names[k]} counter\n{names[k]} {v}\n" for k, v in values.items()
        )

    return app


app = create_app()
