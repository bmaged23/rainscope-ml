"""Optional neural-serving regression; CPU CI skips without modern dependencies."""

import numpy as np
import pytest
from rainscope.modern import TabMModel
from rainscope.models import preprocessing
from rainscope.data import features
from test_pipeline import observations


def test_resident_serving_preserves_predictions():
    torch = pytest.importorskip("torch")
    TabM = pytest.importorskip("tabm").TabM
    frame = features(observations().iloc[:2])
    model = TabMModel.__new__(TabMModel)
    model.preprocessor = preprocessing().fit(frame)
    width = model.preprocessor.transform(frame).shape[1]
    model.model = TabM.make(
        n_num_features=width, d_out=1, n_blocks=1, d_block=16, dropout=0.0, k=2
    )
    # Compare CPU lifecycle paths, including machines with no CUDA runtime.
    model.set_serving_device("cpu")
    expected = model.predict_proba(frame)
    np.testing.assert_allclose(model.predict_proba(frame), expected, atol=1e-7)
    assert model.serving_device == "cpu"
    if torch.cuda.is_available():
        model.set_serving_device("cuda")
        np.testing.assert_allclose(model.predict_proba(frame), expected, atol=1e-6)
        assert next(model.model.parameters()).is_cuda
    with pytest.raises(ValueError):
        model.set_serving_device("typo")
