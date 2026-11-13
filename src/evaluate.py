"""Metrics, breakdowns and capacity-aware flags."""
import numpy as np
import pandas as pd

from .config import FULL_OCC, LOW_OCC, WATCH_OCC


def regression_metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = y - p
    return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "WAPE_%": float(100 * np.abs(e).sum() / y.sum()), "Bias": float(np.mean(p - y))}


def group_metrics(frame: pd.DataFrame, by, y="passengers", p="pred") -> pd.DataFrame:
    rows = []
    for key, g in frame.groupby(by, observed=True):
        rows.append({**(dict(zip(by, key)) if isinstance(by, list) else {by: key}),
                     "n_trips": len(g), **regression_metrics(g[y], g[p])})
    return pd.DataFrame(rows)


def interval_coverage(y, lo, hi) -> float:
    y = np.asarray(y)
    return float(np.mean((y >= np.asarray(lo)) & (y <= np.asarray(hi))))


def capacity_flags(pred, p90, capacity) -> np.ndarray:
    """ADD_CAPACITY / WATCH / NORMAL / UNDER_USED - advisory only, a human approves every action."""
    pred, p90, capacity = (np.asarray(a, float) for a in (pred, p90, capacity))
    occ, hi = pred / capacity, p90 / capacity
    return np.select([occ >= FULL_OCC, (occ >= WATCH_OCC) | (hi >= 1.0), occ < LOW_OCC],
                     ["ADD_CAPACITY", "WATCH", "UNDER_USED"], default="NORMAL")
