# Architecture & data schema

```mermaid
flowchart LR
  G[generate_data.py<br/>seeded synthetic trips] --> R[(data/raw/bus_trips.csv)]
  R --> C[data_prep.clean<br/>dupes, negatives, over-capacity]
  C --> S[chronological split<br/>train / val / test]
  S --> F[features.FeatureBuilder<br/>calendar, holiday, lag>=7d, route attrs]
  F --> B[Baselines<br/>route-time-weekday mean, last week]
  F --> M[Ridge / RandomForest / LightGBM<br/>tuned on validation only]
  M --> A[(models/forecaster.joblib<br/>point + P10 + P90 + FeatureBuilder)]
  C --> H[(data/processed/history.csv)]
  A --> P[predict.Forecaster]
  H --> P
  P --> U[Gradio app / API<br/>forecast + interval + flag + drivers + review notes]
  U --> D{{Human planner approves}}
```

## Raw schema (`data/raw/bus_trips.csv`)
| column | type | known at forecast time (T-48h)? | used as feature? |
|---|---|---|---|
| trip_id | str | yes | no (identifier) |
| route_id, route_name, route_type | str | yes | yes (route_type, route one-hot) |
| distance_km, capacity | int | yes | yes |
| date, departure_time | date, HH:MM | yes | yes (calendar + slot features) |
| fare_inr | float | yes (pricing rule) | yes (missing → route median) |
| passengers | int | **no – TARGET** | target only; lagged ≥7 days as history |
| revenue_inr | float | **no – derived from target** | **never** |
| disruption | 0/1 | **no – known afterwards** | **never** (used only in error analysis) |
