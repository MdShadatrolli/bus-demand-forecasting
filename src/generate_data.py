"""Synthetic bus-trip generator (fully reproducible with a seed).

Each row is one scheduled trip: (route, departure date, departure time).
Demand is built from interpretable factors (slot, weekday, month/season, holiday run-up, growth)
with multiplicative noise + Poisson sampling, then capped at bus capacity (boarded passengers can never
exceed seats -> demand is *censored* when a bus sells out). Random disruptions (strikes, floods) cut demand.
Deliberate data-quality problems are injected so the cleaning step has real work to do.
No personal data is generated: every row is an aggregate trip count.
"""
import argparse

import numpy as np
import pandas as pd

from .config import END_DATE, HOLIDAYS, SEED, START_DATE, get_paths

ROUTES = [
    dict(route_id="R01", route_name="Bengaluru - Chennai", route_type="intercity", distance_km=350, capacity=45,
         base=30, growth=0.10, fare_per_km=1.6,
         slots=[("06:00", 0.9), ("09:30", 0.7), ("14:00", 0.6), ("21:00", 1.3), ("22:30", 1.2)]),
    dict(route_id="R02", route_name="Bengaluru - Hyderabad", route_type="intercity", distance_km=570, capacity=45,
         base=32, growth=0.12, fare_per_km=1.5,
         slots=[("18:30", 1.0), ("20:00", 1.3), ("21:30", 1.2)]),
    dict(route_id="R03", route_name="Mumbai - Pune", route_type="commuter", distance_km=150, capacity=52,
         base=30, growth=0.06, fare_per_km=1.4,
         slots=[("06:00", 1.2), ("07:30", 1.3), ("09:00", 1.0), ("12:00", 0.6), ("17:00", 1.2), ("18:30", 1.3),
                ("20:00", 0.9)]),
    dict(route_id="R04", route_name="Delhi - Jaipur", route_type="intercity", distance_km=280, capacity=45,
         base=28, growth=0.08, fare_per_km=1.5,
         slots=[("06:00", 1.0), ("08:00", 0.9), ("14:00", 0.7), ("22:00", 1.2)]),
    dict(route_id="R05", route_name="Hyderabad - Vijayawada", route_type="intercity", distance_km=275, capacity=45,
         base=27, growth=0.07, fare_per_km=1.4,
         slots=[("07:00", 0.9), ("13:00", 0.7), ("19:00", 1.2), ("22:30", 1.3)]),
    dict(route_id="R06", route_name="Chennai - Coimbatore", route_type="intercity", distance_km=500, capacity=40,
         base=26, growth=0.11, fare_per_km=1.6,
         slots=[("20:00", 1.2), ("21:30", 1.3), ("23:00", 1.0)]),
    dict(route_id="R07", route_name="Pune - Goa", route_type="tourist", distance_km=450, capacity=40,
         base=30, growth=0.15, fare_per_km=1.8,
         slots=[("17:00", 1.0), ("19:00", 1.3), ("21:00", 1.1)]),
    dict(route_id="R08", route_name="Ahmedabad - Surat", route_type="commuter", distance_km=265, capacity=52,
         base=28, growth=0.05, fare_per_km=1.3,
         slots=[("06:30", 1.1), ("09:00", 0.9), ("12:30", 0.6), ("17:30", 1.2), ("19:30", 1.0)]),
]

DOW = {  # Mon..Sun multipliers
    "commuter": [1.15, 1.05, 1.00, 1.05, 1.20, 0.70, 0.80],
    "intercity": [0.95, 0.85, 0.85, 0.95, 1.25, 1.15, 1.20],
    "tourist": [0.55, 0.55, 0.60, 0.80, 1.55, 1.60, 1.05],
}
MONTH = {  # Jan..Dec multipliers
    "commuter": [1.00, 1.00, 0.95, 0.90, 0.90, 1.00, 1.00, 1.00, 1.05, 1.10, 1.10, 1.05],
    "intercity": [1.00, 0.95, 1.05, 1.15, 1.15, 0.95, 0.90, 0.90, 0.95, 1.10, 1.10, 1.05],
    "tourist": [1.10, 0.90, 1.10, 1.40, 1.50, 0.70, 0.60, 0.65, 0.80, 1.00, 1.10, 1.50],
}
DEMAND_SCALE = 0.78  # global calibration so that only true peaks/festivals sell out
HOL_TYPE_MULT = {"commuter": 0.3, "intercity": 1.0, "tourist": 1.4}
HOL_OFFSET_EFFECT = {-3: 0.10, -2: 0.25, -1: 0.40, 0: -0.10, 1: 0.25, 2: 0.10}  # days relative to holiday


def _holiday_effect(dates: pd.DatetimeIndex) -> dict:
    """Sum of weighted run-up/return effects per calendar date."""
    hol = [(pd.Timestamp(d), w) for d, _, w in HOLIDAYS]
    out = {}
    for d in dates:
        eff = 0.0
        for h, w in hol:
            off = (d - h).days
            if off in HOL_OFFSET_EFFECT:
                eff += w * HOL_OFFSET_EFFECT[off]
        out[d] = eff
    return out


def generate_dataset(start=START_DATE, end=END_DATE, seed=SEED, inject_quality_issues=True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    n_days = len(dates)

    slot_rows = []
    for ri, r in enumerate(ROUTES):
        for t, w in r["slots"]:
            slot_rows.append({**{k: r[k] for k in ("route_id", "route_name", "route_type", "distance_km", "capacity",
                                                    "base", "growth", "fare_per_km")},
                              "route_idx": ri, "departure_time": t, "slot_w": w})
    df = pd.DataFrame(slot_rows).merge(pd.DataFrame({"date": dates}), how="cross")
    df = df.sort_values(["date", "route_id", "departure_time"]).reset_index(drop=True)

    dow = df["date"].dt.dayofweek.to_numpy()
    month = df["date"].dt.month.to_numpy()
    hour = df["departure_time"].str.slice(0, 2).astype(int).to_numpy()
    t_days = (df["date"] - dates[0]).dt.days.to_numpy()
    rtype = df["route_type"].to_numpy()

    dow_f = np.ones(len(df)); month_f = np.ones(len(df)); hol_mult = np.ones(len(df))
    for ty in DOW:
        m = rtype == ty
        dow_f[m] = np.array(DOW[ty])[dow[m]]
        month_f[m] = np.array(MONTH[ty])[month[m] - 1]
        hol_mult[m] = HOL_TYPE_MULT[ty]
    hol_eff = df["date"].map(_holiday_effect(dates)).to_numpy()
    hol_f = np.clip(1 + hol_mult * hol_eff, 0.5, None)

    # route-type specific slot x weekday interactions (commuter peaks: Monday morning, Friday/Sunday evening)
    inter = np.ones(len(df))
    comm = rtype == "commuter"
    inter[comm & (hour < 10) & (dow == 0)] = 1.25
    inter[comm & (hour >= 17) & np.isin(dow, [4, 6])] = 1.25

    growth_f = 1 + df["growth"].to_numpy() * t_days / 365.0

    # disruptions: statewide event days + route-specific events (unpredictable at forecast time)
    glob_day = rng.random(n_days) < 0.008
    route_day = rng.random((n_days, len(ROUTES))) < 0.02
    route_cut = rng.uniform(0.3, 0.7, size=(n_days, len(ROUTES)))
    di, ri = t_days, df["route_idx"].to_numpy()
    disruption = glob_day[di] | route_day[di, ri]
    disr_f = np.ones(len(df))
    disr_f[route_day[di, ri]] = route_cut[di, ri][route_day[di, ri]]
    disr_f[glob_day[di]] = 0.5

    mu = DEMAND_SCALE * df["base"].to_numpy() * df["slot_w"].to_numpy() * dow_f * month_f * hol_f * inter * growth_f * disr_f
    mu = mu * rng.lognormal(0.0, 0.10, size=len(df))
    demand = rng.poisson(mu)
    passengers = np.minimum(demand, df["capacity"].to_numpy())  # censoring at capacity

    is_weekend = dow >= 5
    fare = df["fare_per_km"].to_numpy() * df["distance_km"].to_numpy() * (
        1 + 0.10 * is_weekend + 0.5 * np.clip(hol_f - 1, 0, None))
    fare = np.round(fare / 10) * 10

    out = pd.DataFrame({
        "trip_id": df["route_id"] + "_" + df["date"].dt.strftime("%Y%m%d") + "_" + df["departure_time"].str.replace(":", ""),
        "route_id": df["route_id"], "route_name": df["route_name"], "route_type": df["route_type"],
        "distance_km": df["distance_km"], "capacity": df["capacity"], "date": df["date"],
        "departure_time": df["departure_time"], "fare_inr": fare, "passengers": passengers.astype(int),
        "revenue_inr": passengers * fare,        # post-hoc column: deliberately a LEAKAGE TRAP
        "disruption": disruption.astype(int),    # known only after the fact: also excluded from features
    })

    # data-feed outage: no records for R05 for five days
    outage = (out["route_id"] == "R05") & (out["date"] >= "2024-03-10") & (out["date"] <= "2024-03-14")
    out = out[~outage].reset_index(drop=True)

    if inject_quality_issues:
        n = len(out)
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        out.loc[rng.choice(n, int(0.005 * n), replace=False), "fare_inr"] = np.nan          # missing fares
        over = rng.choice(n, int(0.001 * n), replace=False)                                  # over-capacity glitches
        out.loc[over, "passengers"] = out.loc[over, "capacity"] + rng.integers(3, 15, size=len(over))
        neg = rng.choice(n, int(0.0005 * n), replace=False)                                  # sensor errors
        out.loc[neg, "passengers"] = -1
        dups = out.sample(int(0.0015 * n), random_state=seed)                                # duplicate rows
        out = pd.concat([out, dups], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    else:
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic bus-trip dataset")
    ap.add_argument("--out", default=str(get_paths()["raw"]))
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    df = generate_dataset(seed=a.seed)
    from pathlib import Path
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df):,} rows -> {a.out}")


if __name__ == "__main__":
    main()
