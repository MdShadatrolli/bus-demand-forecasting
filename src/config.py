"""Central configuration: paths, dates, split boundaries, thresholds, holiday calendar."""
from pathlib import Path

SEED = 42
ROOT = Path(__file__).resolve().parents[1]


def get_paths(root: Path = ROOT) -> dict:
    root = Path(root)
    return {
        "root": root,
        "raw": root / "data" / "raw" / "bus_trips.csv",
        "history": root / "data" / "processed" / "history.csv",
        "artifact": root / "models" / "forecaster.joblib",
        "reports": root / "reports",
        "figures": root / "reports" / "figures",
    }


# ---- data window and chronological split (test = last quarter, never used for tuning)
START_DATE = "2023-01-01"
END_DATE = "2024-12-31"
TRAIN_END = "2024-06-30"   # train: start+warm-up .. 2024-06-30
VAL_END = "2024-09-30"     # val:   2024-07-01 .. 2024-09-30  (hyper-parameter tuning / early stopping)
#                            test:  2024-10-01 .. 2024-12-31  (touched once, for final reporting)
WARMUP_DAYS = 28           # first 28 days have no 4-week lag history -> not used for modelling

# ---- problem definition
TARGET = "passengers"
HORIZON_DAYS = 2           # forecast is issued 48h before departure
PEAK_HOURS = [(6, 10), (17, 22)]  # [start, end) hours counted as "peak"

# ---- capacity-aware flag thresholds (occupancy = passengers / capacity)
FULL_OCC = 0.90    # predicted occupancy >= 90%  -> ADD_CAPACITY
WATCH_OCC = 0.75   # >= 75% (or P90 reaches capacity) -> WATCH
LOW_OCC = 0.35     # < 35% -> UNDER_USED
HIGH_LOAD_ACTUAL = 0.90  # ground truth definition of a "high-load" trip for flag evaluation

# ---- columns that must NEVER be model inputs (not known at forecast time / are the target)
FORBIDDEN_FEATURES = ["passengers", "revenue_inr", "disruption", "trip_id"]

# ---- public-holiday / festival calendar (date, name, weight). Approximate dates, used for synthetic data.
HOLIDAYS = [
    # 2023
    ("2023-01-15", "Pongal", 1.0), ("2023-01-26", "Republic Day", 0.5), ("2023-03-08", "Holi", 0.7),
    ("2023-04-22", "Eid al-Fitr", 0.8), ("2023-08-15", "Independence Day", 0.5),
    ("2023-09-19", "Ganesh Chaturthi", 0.8), ("2023-10-02", "Gandhi Jayanti", 0.5),
    ("2023-10-24", "Dussehra", 1.0), ("2023-11-12", "Diwali", 1.5), ("2023-12-25", "Christmas", 0.8),
    ("2023-12-31", "New Year's Eve", 0.6),
    # 2024
    ("2024-01-15", "Pongal", 1.0), ("2024-01-26", "Republic Day", 0.5), ("2024-03-25", "Holi", 0.7),
    ("2024-04-11", "Eid al-Fitr", 0.8), ("2024-08-15", "Independence Day", 0.5),
    ("2024-09-07", "Ganesh Chaturthi", 0.8), ("2024-10-02", "Gandhi Jayanti", 0.5),
    ("2024-10-12", "Dussehra", 1.0), ("2024-11-01", "Diwali", 1.5), ("2024-12-25", "Christmas", 0.8),
    ("2024-12-31", "New Year's Eve", 0.6),
    # 2025 (calendar is known in advance -> allowed for forecasting future dates)
    ("2025-01-14", "Pongal", 1.0), ("2025-01-26", "Republic Day", 0.5), ("2025-03-14", "Holi", 0.7),
    ("2025-03-31", "Eid al-Fitr", 0.8), ("2025-08-15", "Independence Day", 0.5),
    ("2025-08-27", "Ganesh Chaturthi", 0.8), ("2025-10-02", "Gandhi Jayanti", 0.5),
    ("2025-10-02", "Dussehra", 1.0), ("2025-10-20", "Diwali", 1.5), ("2025-12-25", "Christmas", 0.8),
    ("2025-12-31", "New Year's Eve", 0.6),
]
