from pathlib import Path
import pickle
import numpy as np
import pandas as pd

MODELS_DIR = Path("models")

def load_pickle(filename: str):
    with (MODELS_DIR / filename).open("rb") as file:
        return pickle.load(file)

def test_saved_feature_order_and_prediction_output():
    model = load_pickle("xgb_model.pkl")
    scaler = load_pickle("scaler.pkl")
    feature_names = list(load_pickle("feature_names.pkl"))

    sample = pd.DataFrame(
        [np.zeros(len(feature_names), dtype=float)],
        columns=feature_names,
    )

    assert sample.shape == (1, len(feature_names))
    assert sample.columns.tolist() == feature_names
    assert not sample.isna().any().any()
    assert np.isfinite(sample.to_numpy()).all()

    scaled = scaler.transform(sample)
    assert scaled.shape == sample.shape
    assert np.isfinite(scaled).all()

    probabilities = model.predict_proba(sample)
    assert probabilities.shape == (1, 2)
    assert np.isfinite(probabilities).all()
    assert np.isclose(probabilities.sum(axis=1)[0], 1.0, atol=1e-6)
    assert 0 <= probabilities[0][1] <= 1
