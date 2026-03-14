"""Threat model simulations."""

from __future__ import annotations

import math
from collections import Counter
from typing import Tuple

import pandas as pd

from .metrics import haversine_m


def _to_local_xy(lat: float, lon: float, ref_lat: float, ref_lon: float) -> Tuple[float, float]:
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(ref_lat))
    x = (lon - ref_lon) * meters_per_deg_lon
    y = (lat - ref_lat) * meters_per_deg_lat
    return x, y


def infer_home_location(df: pd.DataFrame, use_noisy: bool = False, grid_m: float = 150.0) -> Tuple[float, float]:
    lat_col = "noisy_latitude" if use_noisy else "latitude"
    lon_col = "noisy_longitude" if use_noisy else "longitude"

    ref_lat = float(df[lat_col].mean())
    ref_lon = float(df[lon_col].mean())

    bins = []
    for r in df.itertuples():
        lat = getattr(r, lat_col)
        lon = getattr(r, lon_col)
        x, y = _to_local_xy(lat, lon, ref_lat, ref_lon)
        bx = int(x // grid_m)
        by = int(y // grid_m)
        bins.append((bx, by))

    most_common, _ = Counter(bins).most_common(1)[0]
    bx, by = most_common

    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(ref_lat))

    est_lon = ref_lon + ((bx + 0.5) * grid_m) / meters_per_deg_lon
    est_lat = ref_lat + ((by + 0.5) * grid_m) / meters_per_deg_lat
    return est_lat, est_lon


def trajectory_reconstruction_error(df: pd.DataFrame, use_noisy: bool = True) -> float:
    """Predict next location using constant-velocity assumption and compare to true location."""
    total_error = 0.0
    count = 0

    for _, group in df.groupby("user_id"):
        group = group.sort_values("timestamp")
        if len(group) < 3:
            continue

        lat_col = "noisy_latitude" if use_noisy else "latitude"
        lon_col = "noisy_longitude" if use_noisy else "longitude"

        lats = group[lat_col].to_numpy()
        lons = group[lon_col].to_numpy()
        true_lats = group["latitude"].to_numpy()
        true_lons = group["longitude"].to_numpy()

        for i in range(2, len(group)):
            dlat = lats[i - 1] - lats[i - 2]
            dlon = lons[i - 1] - lons[i - 2]
            pred_lat = lats[i - 1] + dlat
            pred_lon = lons[i - 1] + dlon
            total_error += haversine_m(pred_lat, pred_lon, true_lats[i], true_lons[i])
            count += 1

    return total_error / count if count else 0.0


def clustering_top_cell_share(df: pd.DataFrame, use_noisy: bool = False, grid_m: float = 150.0) -> float:
    """Measure how concentrated points are in the most frequent grid cell."""
    lat_col = "noisy_latitude" if use_noisy else "latitude"
    lon_col = "noisy_longitude" if use_noisy else "longitude"

    ref_lat = float(df[lat_col].mean())
    ref_lon = float(df[lon_col].mean())

    bins = []
    for r in df.itertuples():
        lat = getattr(r, lat_col)
        lon = getattr(r, lon_col)
        x, y = _to_local_xy(lat, lon, ref_lat, ref_lon)
        bx = int(x // grid_m)
        by = int(y // grid_m)
        bins.append((bx, by))

    if not bins:
        return 0.0

    most_common, count = Counter(bins).most_common(1)[0]
    return count / len(bins)


def evaluate_attacks(df: pd.DataFrame) -> dict:
    true_home = infer_home_location(df, use_noisy=False)
    noisy_home = infer_home_location(df, use_noisy=True)
    error_m = haversine_m(true_home[0], true_home[1], noisy_home[0], noisy_home[1])

    traj_error_noisy = trajectory_reconstruction_error(df, use_noisy=True)
    traj_error_true = trajectory_reconstruction_error(df, use_noisy=False)

    cluster_true = clustering_top_cell_share(df, use_noisy=False)
    cluster_noisy = clustering_top_cell_share(df, use_noisy=True)

    return {
        "true_home_lat": true_home[0],
        "true_home_lon": true_home[1],
        "noisy_home_lat": noisy_home[0],
        "noisy_home_lon": noisy_home[1],
        "home_inference_error_m": error_m,
        "traj_reconstruction_error_noisy_m": traj_error_noisy,
        "traj_reconstruction_error_true_m": traj_error_true,
        "cluster_top_cell_share_true": cluster_true,
        "cluster_top_cell_share_noisy": cluster_noisy,
    }
