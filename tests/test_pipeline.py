"""Preprocessing, leakage, evaluation and inference tests.   Run: pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.data_prep import clean, split_masks, to_dt
from src.evaluate import capacity_flags, regression_metrics
from src.features import FORBIDDEN_FEATURES, FeatureBuilder, add_lag_features
from src.generate_data import generate_dataset
from src.train import run


@pytest.fixture(scope="module")
def small_raw():
    return generate_dataset("2023-10-01", "2024-03-31", seed=C.SEED)


@pytest.fixture(scope="module")
def small_clean(small_raw):
    df = small_raw.copy()
    df["date"] = to_dt(df["date"])
    return clean(df)[0]


def test_generator_is_reproducible():
    a = generate_dataset("2024-01-01", "2024-01-20", seed=7)
    b = generate_dataset("2024-01-01", "2024-01-20", seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_cleaning_fixes_injected_problems(small_raw):
    raw = small_raw.copy()
    raw["date"] = to_dt(raw["date"])
    df, rep = clean(raw)
    assert df["trip_id"].is_unique
    assert (df["passengers"] >= 0).all()
    assert (df["passengers"] <= df["capacity"]).all()
    assert rep["duplicate_trips_removed"] > 0


def test_chronological_split_has_no_overlap(small_clean):
    m = split_masks(small_clean, "2024-01-31", "2024-02-29")
    d = small_clean["date"]
    assert d[m["train"]].max() < d[m["val"]].min()
    assert d[m["val"]].max() < d[m["test"]].min()
    assert not (m["train"] & m["val"]).any() and not (m["val"] & m["test"]).any()


def test_lag_features_use_only_the_past_and_ignore_own_target(small_clean):
    df = small_clean
    row = df[df["date"] >= "2024-01-15"].iloc[[0]]
    base = add_lag_features(row, df)
    expected = df[(df.route_id == row.route_id.iloc[0]) & (df.departure_time == row.departure_time.iloc[0])
                  & (df.date == row.date.iloc[0] - pd.Timedelta(days=7))]["passengers"]
    assert base["lag_w1"].iloc[0] == expected.iloc[0]
    # changing the row's OWN target and any FUTURE target must not change its features
    df2 = df.copy()
    df2.loc[(df2.date >= row.date.iloc[0]), "passengers"] = 9999
    pd.testing.assert_frame_equal(base, add_lag_features(row, df2))


def test_no_forbidden_columns_in_features(small_clean):
    fb = FeatureBuilder().fit(small_clean)
    X = fb.transform(small_clean, history=small_clean)
    assert not set(X.columns) & set(FORBIDDEN_FEATURES)
    assert X.shape[0] == len(small_clean)


def test_metrics_known_values():
    m = regression_metrics([10, 20], [12, 18])
    assert m["MAE"] == 2 and m["RMSE"] == 2 and abs(m["WAPE_%"] - 100 * 4 / 30) < 1e-9


def test_capacity_flags():
    f = capacity_flags([48, 40, 25, 10], [50, 44, 30, 15], [50, 50, 50, 50])
    assert list(f) == ["ADD_CAPACITY", "WATCH", "NORMAL", "UNDER_USED"]


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("proj")
    run(root=root, fast=True, verbose=False, data_start="2023-10-01", data_end="2024-04-30",
        train_end="2024-02-15", val_end="2024-03-15")
    return root


def test_training_saves_artifact(trained):
    assert (trained / "models" / "forecaster.joblib").exists()
    assert (trained / "reports" / "metrics.json").exists()


def test_inference_loads_saved_model_and_is_sane(trained):
    from src.predict import Forecaster
    f = Forecaster(trained)
    r = f.predict("R03", "2024-04-10", "18:30")
    lo, hi = r["interval_80"]
    assert 0 <= lo <= r["predicted_passengers"] <= hi <= r["capacity"]
    assert r["flag"] in {"ADD_CAPACITY", "WATCH", "NORMAL", "UNDER_USED"}
    assert f.predict("R03", "2024-04-10", "18:30") == r  # deterministic
    # different inputs must give different outputs (i.e. not hard-coded)
    assert f.predict("R07", "2024-04-12", "19:00")["predicted_passengers"] != r["predicted_passengers"]


def test_unknown_route_is_refused(trained):
    from src.predict import Forecaster
    with pytest.raises(ValueError):
        Forecaster(trained).predict("R99", "2024-04-10", "18:30")
