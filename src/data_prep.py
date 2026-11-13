"""Loading, cleaning and chronological splitting."""
import numpy as np
import pandas as pd

from .config import WARMUP_DAYS


def to_dt(s) -> pd.Series:
    """Parse to a consistent datetime64[ns], midnight-normalised (avoids pandas 2/3 unit mismatches)."""
    return pd.to_datetime(s).dt.normalize().astype("datetime64[ns]")


def load_raw(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = to_dt(df["date"])
    return df


def clean(df: pd.DataFrame):
    """Data-quality fixes. Returns (clean_df, report). Missing fares are imputed later (train-only statistics)."""
    report = {"rows_raw": int(len(df))}
    df = df.copy()

    dup = df.duplicated("trip_id", keep="first")
    report["duplicate_trips_removed"] = int(dup.sum())
    df = df[~dup]

    neg = df["passengers"] < 0
    report["negative_passengers_removed"] = int(neg.sum())
    df = df[~neg].copy()

    # impossible value (more boarded than seats): the true count is unknown, so drop it - clipping would invent a label
    over = df["passengers"] > df["capacity"]
    report["over_capacity_removed"] = int(over.sum())
    df = df[~over].copy()

    report["missing_fare_rows(imputed_later)"] = int(df["fare_inr"].isna().sum())
    report["rows_clean"] = int(len(df))
    df = df.sort_values(["date", "route_id", "departure_time"]).reset_index(drop=True)
    return df, report


def split_masks(df: pd.DataFrame, train_end, val_end, warmup_days: int = WARMUP_DAYS) -> dict:
    """Chronological split by calendar date. Rows in the warm-up window (no lag history) are excluded."""
    train_end, val_end = pd.Timestamp(train_end), pd.Timestamp(val_end)
    first_usable = df["date"].min() + pd.Timedelta(days=warmup_days)
    usable = df["date"] >= first_usable
    return {
        "train": usable & (df["date"] <= train_end),
        "val": usable & (df["date"] > train_end) & (df["date"] <= val_end),
        "test": usable & (df["date"] > val_end),
    }
