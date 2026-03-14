"""Privacy pipeline for applying geo-indistinguishable noise."""

from __future__ import annotations

import pandas as pd
import numpy as np

from .geo_noise import apply_geo_indistinguishability


def apply_noise_to_df(df: pd.DataFrame, epsilon: float, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    noisy_lats = []
    noisy_lons = []
    distances = []

    for _, row in df.iterrows():
        noisy_lat, noisy_lon, dist = apply_geo_indistinguishability(
            row["latitude"], row["longitude"], epsilon, rng
        )
        noisy_lats.append(noisy_lat)
        noisy_lons.append(noisy_lon)
        distances.append(dist)

    out = df.copy()
    out["noisy_latitude"] = noisy_lats
    out["noisy_longitude"] = noisy_lons
    out["epsilon"] = epsilon
    out["noise_distance"] = distances
    return out


def run_pipeline(input_path: str, output_path: str, epsilon: float, seed: int = 42) -> None:
    df = pd.read_csv(input_path)
    out = apply_noise_to_df(df, epsilon, seed=seed)
    out.to_csv(output_path, index=False)
