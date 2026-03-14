"""Utility and privacy metrics."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from .geo_noise import EARTH_RADIUS_M


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_M * c


def mean_location_error(df: pd.DataFrame) -> float:
    errors = [
        haversine_m(r.latitude, r.longitude, r.noisy_latitude, r.noisy_longitude)
        for r in df.itertuples()
    ]
    return float(np.mean(errors)) if errors else 0.0


def max_privacy_radius(df: pd.DataFrame) -> float:
    if "noise_distance" in df.columns:
        return float(df["noise_distance"].max())
    distances = [
        haversine_m(r.latitude, r.longitude, r.noisy_latitude, r.noisy_longitude)
        for r in df.itertuples()
    ]
    return float(np.max(distances)) if distances else 0.0


def generate_pois(
    center_lat: float,
    center_lon: float,
    count: int = 20,
    seed: int = 7,
) -> list[tuple[float, float]]:
    rng = np.random.default_rng(seed)
    pois = []
    for _ in range(count):
        dlat = rng.uniform(-0.02, 0.02)
        dlon = rng.uniform(-0.02, 0.02)
        pois.append((center_lat + dlat, center_lon + dlon))
    return pois


def nearest_poi(lat: float, lon: float, pois: Iterable[tuple[float, float]]) -> int:
    min_idx = -1
    min_dist = float("inf")
    for idx, (plat, plon) in enumerate(pois):
        d = haversine_m(lat, lon, plat, plon)
        if d < min_dist:
            min_dist = d
            min_idx = idx
    return min_idx


def service_accuracy(df: pd.DataFrame, pois: Iterable[tuple[float, float]]) -> float:
    matches = 0
    total = 0
    for r in df.itertuples():
        true_idx = nearest_poi(r.latitude, r.longitude, pois)
        noisy_idx = nearest_poi(r.noisy_latitude, r.noisy_longitude, pois)
        matches += int(true_idx == noisy_idx)
        total += 1
    return matches / total if total else 0.0
