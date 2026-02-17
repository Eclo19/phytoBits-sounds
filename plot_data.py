#!/usr/bin/env python3
import re
import pandas as pd
from datetime import datetime, timedelta
from pytz import timezone
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =========================
# CONFIG
# =========================
CHICAGO_TZ = timezone("America/Chicago")
CSV_FILE = "ddps_patrice_COM22.csv"
OUTPUT_FILE = "output.png"
plot_title = "sensor data"

# Light schedule (hour, minute)
LIGHT_ON = (10, 0)   # 10:00 AM
LIGHT_OFF = (22, 0)  # 10:00 PM

# Plot only from this start time (Jan 9 @ 10:00 AM Chicago time) to the end
CUTOFF_START = CHICAGO_TZ.localize(datetime(2026, 1, 9, 10, 0, 0))

# =========================
# Parsing helpers
# =========================
_num_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def _to_float(s: str):
    m = _num_re.search(s)
    return float(m.group(0)) if m else None

def parse_raw(raw: str) -> dict:
    """
    Example raw:
    "co2=588 ppm, scd_t=23.76 C, scd_h=31.78 %, dht_t=23.80 C, dht_h=37.00 %"
    """
    out = {}
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return out

    for part in str(raw).split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        key = k.strip().lower()
        val = _to_float(v)
        if val is None:
            continue

        if key == "co2":
            out["CO2"] = val
        elif key == "scd_t":
            out["SCD_T"] = val
        elif key == "scd_h":
            out["SCD_H"] = val
        elif key == "dht_t":
            out["DHT_T"] = val
        elif key == "dht_h":
            out["DHT_H"] = val

    return out

# =========================
# Load CSV file
# =========================
def load_file(csv_file: str) -> pd.DataFrame:
    df0 = pd.read_csv(csv_file)
    if "Timestamp" not in df0.columns or "Raw" not in df0.columns:
        raise ValueError("CSV must have columns: Timestamp, Raw")

    df0["Timestamp"] = pd.to_datetime(df0["Timestamp"], errors="coerce")
    df0 = df0.dropna(subset=["Timestamp"]).copy()

    parsed = df0["Raw"].apply(parse_raw).apply(pd.Series)
    df = pd.concat([df0[["Timestamp"]], parsed], axis=1)

    # Timezone: Chicago (assume timestamps are local Chicago time)
    if getattr(df["Timestamp"].dt, "tz", None) is None:
        df["Timestamp"] = df["Timestamp"].dt.tz_localize(CHICAGO_TZ)
    else:
        df["Timestamp"] = df["Timestamp"].dt.tz_convert(CHICAGO_TZ)

    # If duplicate timestamps exist, average the numeric values
    df = df.groupby("Timestamp", as_index=False).mean(numeric_only=True)

    df = df.sort_values("Timestamp").reset_index(drop=True)
    return df

def apply_start_cutoff(df: pd.DataFrame) -> pd.DataFrame:
    # Keep everything at or after the cutoff start timestamp
    return df[df["Timestamp"] >= CUTOFF_START].copy()

# =========================
# Create plot
# =========================
def make_figure(df: pd.DataFrame):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 9), sharex=True)

    start_date = df["Timestamp"].min().date()
    end_date = df["Timestamp"].max().date()

    # Add day/night shading for each day in the displayed range
    current = start_date
    while current <= end_date:
        day_start = CHICAGO_TZ.localize(datetime.combine(current, datetime.min.time()))
        light_on = day_start + timedelta(hours=LIGHT_ON[0], minutes=LIGHT_ON[1])
        light_off = day_start + timedelta(hours=LIGHT_OFF[0], minutes=LIGHT_OFF[1])
        day_end = day_start + timedelta(days=1)

        for ax in (ax1, ax2, ax3):
            ax.axvspan(day_start, light_on, alpha=0.3, color="darkgrey", zorder=0)
            ax.axvspan(light_off, day_end, alpha=0.3, color="darkgrey", zorder=0)

        current += timedelta(days=1)

    def plot_if_present(ax, col, label):
        if col in df.columns and df[col].notna().any():
            ax.plot(df["Timestamp"], df[col], "o", markersize=2, label=label)

    # CO2
    plot_if_present(ax1, "CO2", "CO2")
    ax1.set_ylabel("CO₂ (ppm)", fontsize=12)
    ax1.set_title("CO₂", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Humidity
    plot_if_present(ax2, "SCD_H", "SCD humidity")
    plot_if_present(ax2, "DHT_H", "DHT humidity")
    ax2.set_ylabel("Humidity (%)", fontsize=12)
    ax2.set_title("Humidity", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    if ax2.get_legend_handles_labels()[0]:
        ax2.legend(loc="best")

    # Temperature
    plot_if_present(ax3, "SCD_T", "SCD temp")
    plot_if_present(ax3, "DHT_T", "DHT temp")
    ax3.set_ylabel("Temperature (°C)", fontsize=12)
    ax3.set_title("Temperature", fontsize=14, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    if ax3.get_legend_handles_labels()[0]:
        ax3.legend(loc="best")

    # Format x-axis: show 10 AM and 10 PM for each day
    ticks = []
    current = start_date
    while current <= end_date:
        day = CHICAGO_TZ.localize(datetime.combine(current, datetime.min.time()))
        ticks.append(day + timedelta(hours=LIGHT_ON[0], minutes=LIGHT_ON[1]))
        ticks.append(day + timedelta(hours=LIGHT_OFF[0], minutes=LIGHT_OFF[1]))
        current += timedelta(days=1)

    ax3.set_xticks(ticks)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%B %d, %I %p", tz=CHICAGO_TZ))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=-30, ha="left")

    # Tight x-limits around filtered data
    ax3.set_xlim(df["Timestamp"].min(), df["Timestamp"].max())

    fig.suptitle(plot_title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    return fig

# =========================
# Main
# =========================
if __name__ == "__main__":
    print(f"Loading {CSV_FILE}...")
    df = load_file(CSV_FILE)
    print(f"Loaded {len(df)} points (after timestamp-averaging duplicates).")

    df = apply_start_cutoff(df)
    if df.empty:
        raise RuntimeError(f"No data points at or after cutoff start {CUTOFF_START}.")

    print(f"After start cutoff ({CUTOFF_START}), plotting {len(df)} points.")
    print("Creating plot...")
    fig = make_figure(df)

    print(f"Saving {OUTPUT_FILE}...")
    fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")

    plt.show()
    print("Done!")
