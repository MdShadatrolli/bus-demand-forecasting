"""Robustness experiments: rolling-origin backtest and new-route (cold-start) stress test."""
import numpy as np
import pandas as pd

from . import config as C
from .evaluate import regression_metrics
from .features import FeatureBuilder
from .train import HistoricalBaseline, make_lgbm


def rolling_backtest(df, params, n_estimators, cutoffs=("2024-04-01", "2024-07-01", "2024-10-01"), window_days=92):
    """Expanding-window backtest: train on everything before the cutoff, test on the next ~3 months."""
    rows = []
    for c in cutoffs:
        c = pd.Timestamp(c)
        usable = df["date"] >= df["date"].min() + pd.Timedelta(days=C.WARMUP_DAYS)
        tr = usable & (df["date"] < c)
        te = (df["date"] >= c) & (df["date"] < c + pd.Timedelta(days=window_days))
        fb = FeatureBuilder().fit(df[tr])
        X = fb.transform(df, history=df)
        m = make_lgbm(params, n_estimators).fit(X[tr], df.loc[tr, C.TARGET])
        base = HistoricalBaseline().fit(df[tr]).predict(df[te])
        y = df.loc[te, C.TARGET]
        rows.append({"test_window": f"{c.date()} .. {(c + pd.Timedelta(days=window_days - 1)).date()}",
                     "baseline_WAPE_%": regression_metrics(y, base)["WAPE_%"],
                     "lightgbm_WAPE_%": regression_metrics(y, m.predict(X[te]))["WAPE_%"]})
    return pd.DataFrame(rows)


def cold_start_experiment(df, params, n_estimators, normal_test_out):
    """Hold out each route entirely. The model never saw the route and has no demand history for it
    (route one-hot and lag features removed) -> how bad is a NEW route?"""
    usable = df["date"] >= df["date"].min() + pd.Timedelta(days=C.WARMUP_DAYS)
    fit_mask = usable & (df["date"] <= pd.Timestamp(C.VAL_END))
    test_mask = df["date"] > pd.Timestamp(C.VAL_END)
    rows = []
    for r in sorted(df["route_id"].unique()):
        tr = fit_mask & (df["route_id"] != r)
        te = test_mask & (df["route_id"] == r)
        fb = FeatureBuilder().fit(df[tr | te])           # route table only; no target information used
        X = fb.transform(df, history=df[df["route_id"] != r])  # no history exists for the new route
        cols = [c for c in X.columns if not (c.startswith("route_") or c.startswith("lag_"))]
        m = make_lgbm(params, n_estimators).fit(X.loc[tr, cols], df.loc[tr, C.TARGET])
        cold = regression_metrics(df.loc[te, C.TARGET], np.clip(m.predict(X.loc[te, cols]), 0, df.loc[te, "capacity"]))
        g = normal_test_out[normal_test_out["route_id"] == r]
        norm = regression_metrics(g[C.TARGET], g["pred"])
        rows.append({"route_id": r, "WAPE_%_known_route": norm["WAPE_%"], "WAPE_%_new_route": cold["WAPE_%"],
                     "MAE_known": norm["MAE"], "MAE_new": cold["MAE"]})
    return pd.DataFrame(rows)
