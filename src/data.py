"""Load NAB (Numenta Anomaly Benchmark) series together with their labelled anomaly windows."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

# The five "realKnownCause" series used in the project (each has a documented real-world cause).
SERIES = [
    "ambient_temperature_system_failure",
    "cpu_utilization_asg_misconfiguration",
    "ec2_request_latency_system_failure",
    "machine_temperature_system_failure",
    "nyc_taxi",
]

SHORT = {
    "ambient_temperature_system_failure": "ambient_temp",
    "cpu_utilization_asg_misconfiguration": "cpu_asg",
    "ec2_request_latency_system_failure": "ec2_latency",
    "machine_temperature_system_failure": "machine_temp",
    "nyc_taxi": "nyc_taxi",
}


def load_nab(series):
    """Return a DataFrame with columns: timestamp, value, is_anomaly (bool), window_id (int, -1 = normal).

    NAB labels anomalies as time *windows*; a point is anomalous if it lies inside any window.
    """
    df = pd.read_csv(RAW / f"{series}.csv", parse_dates=["timestamp"])
    windows = json.load(open(RAW / "combined_windows.json"))[f"realKnownCause/{series}.csv"]
    win_id = np.full(len(df), -1)
    for i, (start, end) in enumerate(windows):
        inside = (df.timestamp >= pd.Timestamp(start)).values & (df.timestamp <= pd.Timestamp(end)).values
        win_id[inside] = i
    df["window_id"] = win_id
    df["is_anomaly"] = win_id >= 0
    return df


def chronological_split(n, train_frac=0.40, calib_frac=0.15):
    """Indices for train / calibration / test, in time order (never shuffled)."""
    a = int(n * train_frac)
    b = int(n * (train_frac + calib_frac))
    return np.arange(0, a), np.arange(a, b), np.arange(b, n)


def steps_per_day(df):
    return int(round(pd.Timedelta("1D") / df.timestamp.diff().mode()[0]))
