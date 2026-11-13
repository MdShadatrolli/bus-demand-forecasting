"""Inference: loads the SAVED artifact + history table and returns real model forecasts.

Forecast = point estimate + 80% interval (P10-P90) + capacity-aware flag + top drivers + human-review reasons.
"""
import math

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .data_prep import load_raw, to_dt
from .evaluate import capacity_flags

DISPLAY = {
    "lag_w1": "Demand same slot last week", "lag_w2": "Demand same slot 2 weeks ago", "lag_w3": "Demand 3 weeks ago",
    "lag_w4": "Demand 4 weeks ago", "lag_mean_4": "Avg demand same slot, last 4 weeks", "dow": "Day of week",
    "dow_sin": "Day of week (cyclic)", "dow_cos": "Day of week (cyclic)", "doy_sin": "Time of year (cyclic)",
    "doy_cos": "Time of year (cyclic)", "doy": "Day of year", "month": "Month", "dep_hour": "Departure hour",
    "is_peak": "Peak-hour slot", "is_weekend": "Weekend", "is_holiday": "Public holiday / festival",
    "days_to_next_holiday": "Days until next holiday", "next_holiday_weight": "Size of next holiday",
    "days_since_prev_holiday": "Days since last holiday", "prev_holiday_weight": "Size of last holiday",
    "capacity": "Bus capacity", "distance_km": "Route distance", "fare_inr": "Ticket fare",
}


class Forecaster:
    def __init__(self, root=C.ROOT):
        paths = C.get_paths(root)
        art = joblib.load(paths["artifact"])
        self.builder, self.point, self.q10, self.q90 = art["builder"], art["point"], art["q10"], art["q90"]
        self.feature_names, self.meta = art["feature_names"], art["meta"]
        self.history = load_raw(paths["history"])
        self.last_date = self.history["date"].max()
        self.split = self.meta["split"]

    # ---- helpers for UIs
    def routes(self):
        r = self.builder.routes_
        return {rid: r.loc[rid, "route_name"] for rid in r.index}

    def slots(self, route_id):
        return self.builder.slots_[route_id]

    # ---- main entry point
    def predict(self, route_id, date, departure_time, fare_inr=None, top_k=5) -> dict:
        if route_id not in self.builder.routes_.index:
            raise ValueError(f"Route '{route_id}' was not in the training data. The model cannot forecast new routes "
                             f"(see limitations) - use planner judgement or a similar-route proxy.")
        d = pd.Timestamp(date).normalize()
        if not (isinstance(departure_time, str) and len(departure_time) == 5 and departure_time[2] == ":"):
            raise ValueError("departure_time must be 'HH:MM'")
        row = pd.DataFrame({"route_id": [route_id], "date": [d], "departure_time": [departure_time],
                            "fare_inr": [np.nan if fare_inr in (None, "") else float(fare_inr)]})
        row["date"] = to_dt(row["date"])

        gap_days = (d - self.last_date).days
        start_week = max(1, math.ceil(gap_days / 7)) if gap_days > 0 else 1
        X = self.builder.transform(row, history=self.history, start_week=start_week)
        cap = float(X["capacity"].iloc[0])
        pred = float(np.clip(self.point.predict(X)[0], 0, cap))
        lo = float(np.clip(min(self.q10.predict(X)[0], pred), 0, cap))
        hi = float(np.clip(max(self.q90.predict(X)[0], pred), 0, cap))
        flag = str(capacity_flags([pred], [hi], [cap])[0])

        contrib = self.point.booster_.predict(X, pred_contrib=True)[0]
        drivers = sorted(({"feature": DISPLAY.get(f, f), "raw_feature": f, "value": (None if pd.isna(X[f].iloc[0])
                           else float(X[f].iloc[0])), "contribution_passengers": float(c)}
                          for f, c in zip(self.feature_names, contrib[:-1])),
                         key=lambda x: -abs(x["contribution_passengers"]))[:top_k]

        reasons = ["Decision-support only: a planner must approve any capacity/schedule change."]
        if departure_time not in self.builder.slots_[route_id]:
            reasons.append("Departure time is not a scheduled slot for this route - no slot history exists.")
        if pd.isna(X["lag_w1"].iloc[0]):
            reasons.append("No demand history found for last week's same slot (e.g. data outage) - lower reliability.")
        if start_week > 1:
            reasons.append(f"Forecast date is {gap_days} days past the last observed day; history shifted back "
                           f"{start_week - 1} extra week(s). Reliability falls the further ahead you go.")
        if X["days_to_next_holiday"].iloc[0] <= 2 or X["days_since_prev_holiday"].iloc[0] <= 2:
            reasons.append("Within 2 days of a holiday/festival - demand is volatile and the model has seen few such events.")
        if (hi - lo) / cap > 0.6:
            reasons.append("Wide prediction interval - high uncertainty.")
        reasons.append("Disruptions (strikes, floods, road closures) and special events are NOT predictable by this model.")

        hist = self.history[(self.history["route_id"] == route_id) & (self.history["date"] == d)
                            & (self.history["departure_time"] == departure_time)]
        actual = int(hist["passengers"].iloc[0]) if len(hist) else None
        if d <= pd.Timestamp(self.split["val_end"]):
            where = "inside training data (in-sample, NOT a fair test)"
        elif d <= self.last_date:
            where = "held-out test period"
        else:
            where = "future date (no actuals)"
        return {"route_id": route_id, "route_name": self.routes()[route_id], "date": str(d.date()),
                "departure_time": departure_time, "capacity": int(cap), "predicted_passengers": round(pred, 1),
                "interval_80": [round(lo, 1), round(hi, 1)], "expected_occupancy_pct": round(100 * pred / cap, 1),
                "flag": flag, "top_drivers": drivers, "review_notes": reasons, "history_shift_weeks": start_week - 1,
                "actual_passengers_if_known": actual, "date_location": where,
                "model": "LightGBM point + P10/P90 quantile models"}
