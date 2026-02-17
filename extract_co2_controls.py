#!/usr/bin/env python3
import re
import numpy as np
import pandas as pd
from datetime import datetime
from pytz import timezone
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =========================
# CONFIG
# =========================
CHICAGO_TZ = timezone("America/Chicago")
CSV_FILE = "Data/facultative CAM (coleus)/D112_ttyUSB1_coleus.csv"

# Control windows (Chicago local time)
SIG1_START = CHICAGO_TZ.localize(datetime(2026, 1, 16, 10, 0, 0))
SIG1_END   = CHICAGO_TZ.localize(datetime(2026, 1, 17, 22, 0, 0))

SIG2_START = CHICAGO_TZ.localize(datetime(2026, 1, 18, 10, 0, 0))
SIG2_END   = CHICAGO_TZ.localize(datetime(2026, 1, 20, 22, 0, 0))

# Output files (raw samples, no interpolation)
OUT_SIG1_NPY = "co2_sig1_raw.npy"
OUT_SIG2_NPY = "co2_sig2_raw.npy"
OUT_SIG1_TIME_NPY = "co2_sig1_time.npy"  # datetime64[ns]
OUT_SIG2_TIME_NPY = "co2_sig2_time.npy"  # datetime64[ns]

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
# Load + parse CSV (same style as your plotting script)
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

    # If duplicate timestamps exist, average numeric values (like your script)
    df = df.groupby("Timestamp", as_index=False).mean(numeric_only=True)
    df = df.sort_values("Timestamp").reset_index(drop=True)
    return df

def extract_co2_window(df: pd.DataFrame, start, end) -> pd.DataFrame:
    seg = df[(df["Timestamp"] >= start) & (df["Timestamp"] <= end)].copy()
    seg = seg.sort_values("Timestamp").reset_index(drop=True)
    if "CO2" not in seg.columns or not seg["CO2"].notna().any():
        raise RuntimeError("No CO2 data found in the requested window.")
    return seg

# =========================
# Plot
# =========================
def plot_segment(seg: pd.DataFrame, title: str):
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    ax.plot(seg["Timestamp"], seg["CO2"], "o-", markersize=2, linewidth=1.0)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("CO₂ (ppm)")
    ax.set_xlabel("Time (Chicago)")
    ax.grid(True, alpha=0.3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d, %I:%M %p", tz=CHICAGO_TZ))
    plt.setp(ax.get_xticklabels(), rotation=-25, ha="left")
    plt.tight_layout()
    return fig

# =========================
# Main
# =========================
if __name__ == "__main__":
    print(f"Loading {CSV_FILE}...")
    df = load_file(CSV_FILE)
    print(f"Loaded {len(df)} points (after timestamp-averaging duplicates).")

    # --- Signal 1 ---
    seg1 = extract_co2_window(df, SIG1_START, SIG1_END)
    sig1 = seg1["CO2"].to_numpy(dtype=np.float32)
    # Save times in UTC as timezone-naive datetime64[ns] (safe + portable)
    t1 = seg1["Timestamp"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")


    np.save(OUT_SIG1_NPY, sig1)
    np.save(OUT_SIG1_TIME_NPY, t1)

    print(f"SIG1: {len(sig1)} samples | {seg1['Timestamp'].min()} -> {seg1['Timestamp'].max()}")
    print(f"Saved: {OUT_SIG1_NPY}, {OUT_SIG1_TIME_NPY}")

    fig1 = plot_segment(seg1, "CO2 Control Signal 1 (Jan 16 10:00 AM → Jan 17 10:00 PM)")
    plt.show()

    # --- Signal 2 ---
    seg2 = extract_co2_window(df, SIG2_START, SIG2_END)
    sig2 = seg2["CO2"].to_numpy(dtype=np.float32)
    t2 = seg2["Timestamp"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")


    np.save(OUT_SIG2_NPY, sig2)
    np.save(OUT_SIG2_TIME_NPY, t2)

    print(f"SIG2: {len(sig2)} samples | {seg2['Timestamp'].min()} -> {seg2['Timestamp'].max()}")
    print(f"Saved: {OUT_SIG2_NPY}, {OUT_SIG2_TIME_NPY}")

    fig2 = plot_segment(seg2, "CO2 Control Signal 2 (Jan 18 10:00 AM → Jan 20 10:00 PM)")
    plt.show()

    print("Done!")
