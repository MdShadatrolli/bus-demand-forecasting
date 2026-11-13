"""Gradio app: loads the saved artifact and returns real forecasts.   python app/app.py  [--share]"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.predict import Forecaster

fc = Forecaster(ROOT)
ROUTE_CHOICES = [(f"{rid} - {name}", rid) for rid, name in fc.routes().items()]
FLAG_TEXT = {"ADD_CAPACITY": "🔴 ADD CAPACITY - likely full", "WATCH": "🟠 WATCH - may fill up",
             "NORMAL": "🟢 NORMAL", "UNDER_USED": "🔵 UNDER-USED - consider consolidating"}


def forecast(route_id, date_str, time_str, fare):
    try:
        r = fc.predict(route_id, date_str.strip(), time_str, fare if fare else None)
    except Exception as e:  # show a readable message instead of a stack trace
        return f"⚠️ {e}", None, {}
    lo, hi = r["interval_80"]
    md = (f"### {r['route_name']} · {r['date']} · {r['departure_time']}\n"
          f"**Forecast: {r['predicted_passengers']} passengers** (80% interval {lo} – {hi}) of {r['capacity']} seats "
          f"→ **{r['expected_occupancy_pct']}% occupancy**\n\n**{FLAG_TEXT[r['flag']]}**\n\n"
          f"*Date is {r['date_location']}.*" + (f" Actual passengers: **{r['actual_passengers_if_known']}**."
                                                if r["actual_passengers_if_known"] is not None else "")
          + "\n\n**Human-review notes**\n" + "\n".join(f"- {n}" for n in r["review_notes"]))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1, 1.5]})
    ax[0].barh([0], [r["capacity"]], color="#e5e7eb", height=.5)
    ax[0].barh([0], [r["predicted_passengers"]], color="#2563eb", height=.5)
    ax[0].errorbar([r["predicted_passengers"]], [0], xerr=[[r["predicted_passengers"] - lo], [hi - r["predicted_passengers"]]],
                   color="black", capsize=5)
    ax[0].set_yticks([]); ax[0].set_xlim(0, r["capacity"] * 1.02); ax[0].set_title("Forecast vs capacity")
    names = [d["feature"] for d in r["top_drivers"]][::-1]
    vals = [d["contribution_passengers"] for d in r["top_drivers"]][::-1]
    ax[1].barh(names, vals, color=["#16a34a" if v > 0 else "#dc2626" for v in vals])
    ax[1].axvline(0, color="black", lw=.8); ax[1].set_title("Why? effect on forecast (passengers)")
    plt.tight_layout()
    return md, fig, r


def update_slots(route_id):
    s = fc.slots(route_id)
    return gr.Dropdown(choices=s, value=s[0], allow_custom_value=True)


with gr.Blocks(title="Bus Passenger-Demand Forecast") as demo:
    gr.Markdown("# 🚌 Bus Passenger-Demand Forecast\nDecision-support tool: forecasts passengers per scheduled trip "
                "48h+ ahead. Advisory only - a planner approves every capacity decision.")
    with gr.Row():
        route = gr.Dropdown(ROUTE_CHOICES, value=ROUTE_CHOICES[0][1], label="Route")
        date = gr.Textbox(value=str((fc.last_date + __import__("pandas").Timedelta(days=2)).date()),
                          label="Departure date (YYYY-MM-DD)")
        time = gr.Dropdown(fc.slots(ROUTE_CHOICES[0][1]), value=fc.slots(ROUTE_CHOICES[0][1])[0], label="Departure time", allow_custom_value=True)
        fare = gr.Number(label="Fare INR (optional)", value=None)
    btn = gr.Button("Forecast demand", variant="primary")
    out_md, out_plot, out_json = gr.Markdown(), gr.Plot(), gr.JSON(label="Raw API response")
    route.change(update_slots, route, time)
    btn.click(forecast, [route, date, time, fare], [out_md, out_plot, out_json], api_name="predict")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    demo.launch(share=a.share, server_port=a.port)
