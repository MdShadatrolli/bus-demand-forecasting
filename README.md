# 🚌 Bus Passenger-Demand Forecasting

Decision-support ML system that forecasts **how many passengers will board a scheduled bus trip**, 48 hours before departure, so a planner can add or consolidate capacity. It is advisory only: every flag needs human approval.

## 1. Problem definition
| Item | Definition |
|---|---|
| Business decision | Add an extra bus / keep as scheduled / consolidate an under-used trip |
| Prediction unit | One scheduled trip = (route, departure date, departure time) |
| Target | `passengers` boarded (integer, ≤ bus capacity) |
| Horizon / point in time | Forecast issued **48 h before departure** (`HORIZON_DAYS = 2`) |
| Allowed inputs | Route/schedule attributes, fare rule, calendar & holiday list, demand history of the same route+slot **≥ 7 days earlier** |
| Forbidden inputs | `passengers`, `revenue_inr` (= passengers × fare), `disruption` (known only afterwards) |
| Output | Point forecast, 80 % interval (P10–P90), occupancy %, capacity flag (`ADD_CAPACITY` / `WATCH` / `NORMAL` / `UNDER_USED`), top drivers, human-review notes |
| Metrics | MAE, RMSE, **WAPE** (primary), plus per-route and peak/off-peak breakdowns; flag precision/recall; interval coverage |

## 2. Data
Fully **synthetic** (no licence issues, no personal data): `src/generate_data.py` (seeded, reproducible). 8 routes · 34 daily slots · 2023-01-01 → 2024-12-31 ≈ 24.8 k trip records.
Embedded real-world behaviours: weekday/season/slot patterns, festival run-ups, growth, capacity censoring (sold-out buses), random disruptions, and **deliberately injected data-quality issues** (missing fares, duplicates, impossible over-capacity and negative counts, a 5-day data outage).

## 3. Quick start (local)
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.generate_data        # writes data/raw/bus_trips.csv
python -m src.train                # clean -> split -> baselines/models -> tune -> test -> saves models/forecaster.joblib
pytest -q                          # preprocessing, leakage, metrics, inference tests
python app/app.py                  # http://127.0.0.1:7861  (add --share for a public link)
```
Or open `notebooks/Bus_Demand_Forecasting.ipynb` in **Google Colab** and *Runtime → Run all* (it writes this whole repo, trains, evaluates and launches the app).

## 4. Split strategy (chronological, no shuffling)
| Set | Dates | Use |
|---|---|---|
| Train | start+28 d … 2024-06-30 | fit models |
| Validation | 2024-07-01 … 2024-09-30 | hyper-parameter search, early stopping, model selection |
| Test | 2024-10-01 … 2024-12-31 | touched once for final metrics (Dussehra, Diwali, Christmas, NYE inside) |

Final models are refit on train+val with the chosen settings, then scored once on test. First 28 days are dropped (no 4-week history).

## 5. Models
1. Baseline A – historical mean by route × departure time × weekday (**required baseline**)
2. Baseline B – same slot last week (seasonal naive)
3. Ridge regression · 4. Random Forest · 5. **LightGBM** (random-search tuned) + LightGBM quantile models (P10/P90) for uncertainty

## 6. Results snapshot (held-out test, seed 42)
| Model | MAE | RMSE | WAPE % |
|---|---|---|---|
| Baseline: route/time/weekday mean | 5.99 | 7.51 | 20.08 |
| Baseline: same slot last week | 6.62 | 8.71 | 22.21 |
| Ridge | 5.12 | 6.49 | 17.15 |
| Random Forest | 5.24 | 6.57 | 17.56 |
| **LightGBM (tuned)** | **5.00** | **6.32** | **16.78** |

On the capacity flag ("will the trip be ≥ 90 % full?") the model catches 78 % of high-load trips vs 59 % for the baseline. Full numbers incl. per-route, peak/off-peak, disruption and flag metrics: `reports/metrics.json`, the notebook and the technical report. Values may differ slightly with other library versions.

## 7. Repository layout
```
src/            config, generate_data, data_prep, features, evaluate, train, experiments, predict
app/app.py      Gradio app (loads models/forecaster.joblib; exposes /predict API)
tests/          pytest: reproducibility, cleaning, split, leakage, metrics, flags, inference
notebooks/      Colab notebook (full walkthrough: EDA -> models -> error analysis -> app)
data/           raw/ (generated CSV)  processed/ (cleaned history used by inference)
models/         forecaster.joblib (saved artifact)
reports/        metrics.json, test_predictions.csv, figures/
docs/           architecture.md (pipeline + schema diagrams)
```

## 8. Acceptance-test map
1. **Reproduce** → commands in §3 (fixed seed 42; two runs match to ~1e-15).
2. **Baseline vs candidate on held-out test** → `reports/metrics.json`, notebook §6.
3. **Real prediction via app/API** → `python app/app.py`; `src/predict.py` loads the saved artifact (no hard-coded outputs).
4. **Explain a prediction/error** → notebook §9 (LightGBM `pred_contrib`) and the "Why?" chart in the app.
5. **No leakage / no sensitive data** → `FORBIDDEN_FEATURES` assertion, `tests/test_pipeline.py::test_lag_features_*`, leakage audit table + "leaky model" demo in the notebook; data is synthetic aggregate counts.

## 9. Limitations & responsible use (short)
* Synthetic data: results show the method works, **not** real-world accuracy. Real deployment needs real booking data and re-validation.
* Ridership is **censored at capacity** – a sold-out bus hides true demand, so peaks are under-estimated.
* **Disruptions and special events are unpredictable** (WAPE ≈ 87 % on disruption trips vs ≈ 16 % on normal trips in test); the app tells the planner to review.
* **New routes** have no history; the model refuses to forecast them. Stress test shows higher error when history is removed.
* Interval coverage is ~73 % vs the 80 % target – treat intervals as slightly optimistic.
* Held-out test is a festival-heavy quarter; one year of festival history only (Diwali seen once in training).
* No personal data is used or needed (aggregate trip counts only). Pricing/fairness: forecasts must not be used to deny service to low-demand routes without human, equity-aware review.

## 10. AI assistance disclosure
> Edit this before submitting so it is true for you: *"Claude (Anthropic) was used to help scaffold the code and plan the project; I reviewed, ran and can explain every part."*
# bus-demand-forecasting
