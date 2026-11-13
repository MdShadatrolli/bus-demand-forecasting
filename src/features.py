"""Leakage-safe feature engineering.

Forecast origin: 48 hours (HORIZON_DAYS) before departure. Allowed inputs:
  * route / schedule attributes (route, capacity, distance, departure time, fare set by the pricing rule)
  * calendar (weekday, month, season, holiday calendar - all known far in advance)
  * demand history of the SAME route+slot at least 7 days earlier (>= horizon, so always observed)
Forbidden: passengers (target), revenue_inr (computed from target), disruption (known only afterwards).
"""
import numpy as np
import pandas as pd

from .config import FORBIDDEN_FEATURES, HOLIDAYS, PEAK_HOURS
from .data_prep import to_dt

HOL_CAP = 15
_HOL = sorted((pd.Timestamp(d), w) for d, _, w in HOLIDAYS)
HOL_DATES = np.array([d for d, _ in _HOL], dtype="datetime64[D]")
HOL_W = np.array([w for _, w in _HOL])
SEASONS = ["winter", "summer", "monsoon", "festive"]
SEASON_OF_MONTH = {12: "winter", 1: "winter", 2: "winter", 3: "summer", 4: "summer", 5: "summer",
                   6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon", 10: "festive", 11: "festive"}
ROUTE_TYPES = ["commuter", "intercity", "tourist"]
LAG_COLS = ["lag_w1", "lag_w2", "lag_w3", "lag_w4"]
KEYS = ["route_id", "departure_time", "date"]

NUMERIC_FEATURES = [
    "capacity", "distance_km", "fare_inr", "dep_hour", "is_peak",
    "dow", "is_weekend", "month", "doy", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
    "is_holiday", "days_to_next_holiday", "next_holiday_weight", "days_since_prev_holiday", "prev_holiday_weight",
    "lag_w1", "lag_w2", "lag_w3", "lag_w4", "lag_mean_4",
]


def holiday_features(dates: pd.Series) -> pd.DataFrame:
    d = dates.to_numpy().astype("datetime64[D]")
    n = len(HOL_DATES)
    i_next = np.searchsorted(HOL_DATES, d, side="left")       # first holiday >= d
    i_prev = np.searchsorted(HOL_DATES, d, side="left") - 1   # last holiday  <  d
    nxt = np.minimum(i_next, n - 1)
    prv = np.maximum(i_prev, 0)
    days_next = np.where(i_next < n, (HOL_DATES[nxt] - d).astype(int), HOL_CAP)
    days_prev = np.where(i_prev >= 0, (d - HOL_DATES[prv]).astype(int), HOL_CAP)
    return pd.DataFrame({
        "is_holiday": (days_next == 0).astype(int),
        "days_to_next_holiday": np.minimum(days_next, HOL_CAP),
        "next_holiday_weight": np.where(days_next < HOL_CAP, HOL_W[nxt], 0.0),
        "days_since_prev_holiday": np.minimum(days_prev, HOL_CAP),
        "prev_holiday_weight": np.where(days_prev < HOL_CAP, HOL_W[prv], 0.0),
    }, index=dates.index)


def add_lag_features(df: pd.DataFrame, history: pd.DataFrame, start_week: int = 1, n_weeks: int = 4) -> pd.DataFrame:
    """Same route+slot demand 7*start_week, 7*(start_week+1)... days BEFORE each row's date.
    start_week=1 -> exactly last week (always known at a 48h horizon). A larger start_week is used at inference
    when the history is stale (forecast date more than a week beyond the last observed day)."""
    out = df[KEYS].copy()
    out["_row"] = np.arange(len(out))
    h = history[KEYS + ["passengers"]].drop_duplicates(KEYS)
    for w in range(1, n_weeks + 1):
        shifted = h.assign(date=h["date"] + pd.Timedelta(days=7 * (start_week + w - 1)))
        shifted = shifted.rename(columns={"passengers": f"lag_w{w}"})
        out = out.merge(shifted, on=KEYS, how="left")
    out = out.sort_values("_row")
    res = out[LAG_COLS].copy()
    res.index = df.index
    res["lag_mean_4"] = res[LAG_COLS].mean(axis=1)
    return res


class FeatureBuilder:
    """Fit on training data only (learns route table + median fares); transform anything."""

    def fit(self, df: pd.DataFrame):
        r = df.drop_duplicates("route_id")[["route_id", "route_name", "route_type", "distance_km", "capacity"]]
        self.routes_ = r.set_index("route_id").sort_index()
        self.route_ids_ = list(self.routes_.index)
        self.slots_ = {k: sorted(v) for k, v in df.groupby("route_id")["departure_time"].unique().items()}
        self.fare_median_ = df.groupby("route_id")["fare_inr"].median()
        return self

    @property
    def feature_names(self):
        return (NUMERIC_FEATURES + [f"route_{r}" for r in self.route_ids_]
                + [f"type_{t}" for t in ROUTE_TYPES] + [f"season_{s}" for s in SEASONS])

    def transform(self, df: pd.DataFrame, history: pd.DataFrame, start_week: int = 1) -> pd.DataFrame:
        d = to_dt(df["date"])
        X = pd.DataFrame(index=df.index)
        rid = df["route_id"]
        X["capacity"] = rid.map(self.routes_["capacity"]).astype(float)
        X["distance_km"] = rid.map(self.routes_["distance_km"]).astype(float)
        fare = df["fare_inr"] if "fare_inr" in df.columns else pd.Series(np.nan, index=df.index)
        X["fare_inr"] = fare.fillna(rid.map(self.fare_median_)).astype(float)
        hh = df["departure_time"].str.slice(0, 2).astype(int)
        mm = df["departure_time"].str.slice(3, 5).astype(int)
        X["dep_hour"] = hh + mm / 60.0
        X["is_peak"] = sum(((hh >= a) & (hh < b)).astype(int) for a, b in PEAK_HOURS).clip(upper=1)
        X["dow"] = d.dt.dayofweek
        X["is_weekend"] = (X["dow"] >= 5).astype(int)
        X["month"] = d.dt.month
        X["doy"] = d.dt.dayofyear
        X["dow_sin"], X["dow_cos"] = np.sin(2 * np.pi * X["dow"] / 7), np.cos(2 * np.pi * X["dow"] / 7)
        X["doy_sin"], X["doy_cos"] = np.sin(2 * np.pi * X["doy"] / 365.25), np.cos(2 * np.pi * X["doy"] / 365.25)
        X = pd.concat([X, holiday_features(d)], axis=1)
        tmp = df[["route_id", "departure_time"]].assign(date=d)
        X = pd.concat([X, add_lag_features(tmp, history, start_week)], axis=1)
        for r in self.route_ids_:
            X[f"route_{r}"] = (rid == r).astype(int)
        rtype = rid.map(self.routes_["route_type"])
        for t in ROUTE_TYPES:
            X[f"type_{t}"] = (rtype == t).astype(int)
        season = d.dt.month.map(SEASON_OF_MONTH)
        for s in SEASONS:
            X[f"season_{s}"] = (season == s).astype(int)
        X = X[self.feature_names]
        assert not set(X.columns) & set(FORBIDDEN_FEATURES), "forbidden column leaked into features"
        return X
