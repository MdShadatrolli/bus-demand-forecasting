"""Repeatable training pipeline:  python -m src.train   (add --fast for a quick smoke run)

Steps: generate/load data -> clean -> chronological split -> baselines -> Ridge / RandomForest / LightGBM
-> tune LightGBM on VALIDATION only -> refit on train+val -> evaluate ONCE on the held-out TEST set -> save artifact.
"""
import argparse
import json
import random
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .data_prep import clean, load_raw, split_masks
from .evaluate import capacity_flags, group_metrics, interval_coverage, regression_metrics
from .features import FeatureBuilder
from .generate_data import generate_dataset

warnings.filterwarnings("ignore", message=".*eval_set.*")
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")


# ----------------------------------------------------------------------------- baselines
class HistoricalBaseline:
    """Mean demand by (route, departure time, weekday); falls back to (route, time) then global mean."""

    def fit(self, df):
        df = df.assign(dow=df["date"].dt.dayofweek)
        self.full_ = df.groupby(["route_id", "departure_time", "dow"])[C.TARGET].mean()
        self.slot_ = df.groupby(["route_id", "departure_time"])[C.TARGET].mean()
        self.global_ = float(df[C.TARGET].mean())
        return self

    def predict(self, df):
        idx = pd.MultiIndex.from_arrays([df["route_id"], df["departure_time"], df["date"].dt.dayofweek])
        p = pd.Series(self.full_.reindex(idx).to_numpy(), index=df.index)
        idx2 = pd.MultiIndex.from_arrays([df["route_id"], df["departure_time"]])
        p = p.fillna(pd.Series(self.slot_.reindex(idx2).to_numpy(), index=df.index)).fillna(self.global_)
        return p.to_numpy()


def seasonal_naive(X, fallback):
    """Same route+slot, same weekday last week; falls back to the historical baseline if missing."""
    return X["lag_w1"].fillna(pd.Series(fallback, index=X.index)).to_numpy()


# ----------------------------------------------------------------------------- models
def make_ridge(alpha=10.0):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("m", Ridge(alpha=alpha))])


def make_rf(fast=False):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("m", RandomForestRegressor(n_estimators=40 if fast else 300, min_samples_leaf=3, max_features=0.5,
                                                 n_jobs=-1, random_state=C.SEED))])


LGB_SPACE = dict(num_leaves=[15, 31, 63], learning_rate=[0.03, 0.05, 0.1], min_child_samples=[10, 20, 50, 100],
                 colsample_bytree=[0.6, 0.8, 1.0], subsample=[0.7, 0.9, 1.0], reg_lambda=[0, 1, 5, 10])


def make_lgbm(params, n_estimators, objective="regression", alpha=None):
    kw = dict(objective=objective, n_estimators=n_estimators, subsample_freq=1, random_state=C.SEED,
              n_jobs=-1, verbose=-1, deterministic=True, force_row_wise=True, **params)
    if alpha is not None:
        kw["alpha"] = alpha
    return lgb.LGBMRegressor(**kw)


def tune_lgbm(Xtr, ytr, Xva, yva, n_iter=20, seed=C.SEED):
    """Random search; early stopping + selection on the VALIDATION set only."""
    rng, rows = random.Random(seed), []
    for _ in range(n_iter):
        p = {k: rng.choice(v) for k, v in LGB_SPACE.items()}
        m = lgb.LGBMRegressor(objective="regression", metric="l1", n_estimators=2000, subsample_freq=1,
                              random_state=seed, n_jobs=-1, verbose=-1, deterministic=True, force_row_wise=True, **p)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(100, verbose=False)])
        rows.append({**p, "best_iter": int(m.best_iteration_),
                     "val_MAE": float(np.mean(np.abs(yva - m.predict(Xva))))})
    res = pd.DataFrame(rows).sort_values("val_MAE").reset_index(drop=True)
    best = {k: res.loc[0, k] for k in LGB_SPACE}
    best = {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v)
            for k, v in best.items()}
    return best, int(res.loc[0, "best_iter"]), res


# ----------------------------------------------------------------------------- pipeline
def run(root=C.ROOT, fast=False, verbose=True, data_start=C.START_DATE, data_end=C.END_DATE,
        train_end=C.TRAIN_END, val_end=C.VAL_END, save=True):
    paths = C.get_paths(root)
    log = print if verbose else (lambda *a, **k: None)

    if not paths["raw"].exists():
        paths["raw"].parent.mkdir(parents=True, exist_ok=True)
        generate_dataset(data_start, data_end).to_csv(paths["raw"], index=False)
        log(f"generated synthetic data -> {paths['raw']}")
    raw = load_raw(paths["raw"])
    df, dq = clean(raw)
    log("data-quality report:", dq)

    m = split_masks(df, train_end, val_end)
    tr, va, te = df[m["train"]], df[m["val"]], df[m["test"]]
    log(f"rows  train={len(tr):,}  val={len(va):,}  test={len(te):,}")
    y = df[C.TARGET]

    # --- stage 1: tuning on train -> validation
    fb = FeatureBuilder().fit(tr)
    X = fb.transform(df, history=df)
    base_val = HistoricalBaseline().fit(tr).predict(va)
    cand = {"Ridge": make_ridge().fit(X[m["train"]], y[m["train"]]),
            "RandomForest": make_rf(fast).fit(X[m["train"]], y[m["train"]])}
    val_scores = {"Baseline: route/time/weekday mean": regression_metrics(va[C.TARGET], base_val)}
    for name, mdl in cand.items():
        val_scores[name] = regression_metrics(va[C.TARGET], mdl.predict(X[m["val"]]))
    best, best_iter, tuning = tune_lgbm(X[m["train"]], y[m["train"]], X[m["val"]], y[m["val"]],
                                        n_iter=4 if fast else 20)
    lgb_val = make_lgbm(best, best_iter).fit(X[m["train"]], y[m["train"]])
    val_scores["LightGBM (tuned)"] = regression_metrics(va[C.TARGET], lgb_val.predict(X[m["val"]]))
    log("best LightGBM params:", best, "iters:", best_iter)

    # --- stage 2: refit on train+val, evaluate once on test
    trva_mask = m["train"] | m["val"]
    trva = df[trva_mask]
    fb2 = FeatureBuilder().fit(trva)
    X2 = fb2.transform(df, history=df)
    n_final = int(best_iter * 1.15)
    models = {
        "Ridge": make_ridge().fit(X2[trva_mask], y[trva_mask]),
        "RandomForest": make_rf(fast).fit(X2[trva_mask], y[trva_mask]),
        "LightGBM (tuned)": make_lgbm(best, n_final).fit(X2[trva_mask], y[trva_mask]),
    }
    q10 = make_lgbm(best, n_final, "quantile", 0.10).fit(X2[trva_mask], y[trva_mask])
    q90 = make_lgbm(best, n_final, "quantile", 0.90).fit(X2[trva_mask], y[trva_mask])
    base = HistoricalBaseline().fit(trva)

    Xte, yte = X2[m["test"]], y[m["test"]]
    preds = {"Baseline: route/time/weekday mean": base.predict(te)}
    preds["Baseline: same slot last week"] = seasonal_naive(Xte, preds["Baseline: route/time/weekday mean"])
    for name, mdl in models.items():
        preds[name] = mdl.predict(Xte)
    test_scores = {k: regression_metrics(yte, np.clip(v, 0, te["capacity"].to_numpy())) for k, v in preds.items()}

    lo, hi = np.minimum(q10.predict(Xte), preds["LightGBM (tuned)"]), np.maximum(q90.predict(Xte), preds["LightGBM (tuned)"])
    lo, hi = np.clip(lo, 0, te["capacity"]), np.clip(hi, 0, te["capacity"])
    cov = interval_coverage(yte, lo, hi)

    out = te[["trip_id", "route_id", "date", "departure_time", "capacity", C.TARGET, "disruption"]].copy()
    out["pred"] = np.clip(preds["LightGBM (tuned)"], 0, te["capacity"].to_numpy())
    out["pred_baseline"] = preds["Baseline: route/time/weekday mean"]
    out["p10"], out["p90"] = lo, hi
    out["is_peak"] = Xte["is_peak"].to_numpy()
    out["flag"] = capacity_flags(out["pred"], out["p90"], out["capacity"])

    results = {"data_quality": dq, "val_scores": val_scores, "test_scores": test_scores,
               "interval_coverage_80": cov, "mean_interval_width": float((hi - lo).mean()),
               "best_params": best, "n_estimators_final": n_final,
               "split": {"train_end": train_end, "val_end": val_end, "data_end": data_end,
                         "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te))}}
    if save:
        paths["artifact"].parent.mkdir(parents=True, exist_ok=True)
        paths["history"].parent.mkdir(parents=True, exist_ok=True)
        paths["reports"].mkdir(parents=True, exist_ok=True)
        joblib.dump({"builder": fb2, "point": models["LightGBM (tuned)"], "q10": q10, "q90": q90,
                     "feature_names": fb2.feature_names, "meta": results}, paths["artifact"])
        df.to_csv(paths["history"], index=False)
        out.to_csv(paths["reports"] / "test_predictions.csv", index=False)
        tuning.to_csv(paths["reports"] / "lgbm_tuning.csv", index=False)
        (paths["reports"] / "metrics.json").write_text(json.dumps(results, indent=2, default=str))
        log(f"saved artifact -> {paths['artifact']}")
    log(pd.DataFrame(test_scores).T.round(2).to_string())
    return {"df": df, "X": X2, "masks": m, "models": models, "q10": q10, "q90": q90, "fb": fb2, "baseline": base,
            "test_out": out, "preds": preds, "results": results, "tuning": tuning, "val_X": X, "paths": paths}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="small/quick run for smoke tests")
    a = ap.parse_args()
    run(fast=a.fast)


if __name__ == "__main__":
    main()
