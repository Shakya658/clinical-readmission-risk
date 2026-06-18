from pathlib import Path
import pickle

MODELS_DIR = Path("models")
ARTEFACT_PATHS = {
    "model": MODELS_DIR / "xgb_model.pkl",
    "scaler": MODELS_DIR / "scaler.pkl",
    "feature_names": MODELS_DIR / "feature_names.pkl",
    "threshold": MODELS_DIR / "threshold.pkl",
}

def load_pickle(path: Path):
    with path.open("rb") as file:
        return pickle.load(file)

def test_required_model_artefacts_exist_and_load():
    loaded = {}
    for name, path in ARTEFACT_PATHS.items():
        assert path.exists(), f"Missing {name} artefact: {path}"
        loaded[name] = load_pickle(path)

    assert hasattr(loaded["model"], "predict_proba")
    assert hasattr(loaded["scaler"], "transform")

    feature_names = list(loaded["feature_names"])
    threshold = float(loaded["threshold"])

    assert feature_names
    assert len(feature_names) == len(set(feature_names))
    assert len(feature_names) == 95
    assert 0 <= threshold <= 1
