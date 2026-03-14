"""Geo-indistinguishability noise generation (Planar Laplace)."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

EARTH_RADIUS_M = 6_371_000.0


def _normalize_lon(lon: float) -> float:
    return (lon + 180.0) % 360.0 - 180.0


def destination_point(lat: float, lon: float, distance_m: float, bearing_rad: float) -> Tuple[float, float]:
    """Move from (lat, lon) along bearing by distance on a spherical Earth."""
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)

    ang_dist = distance_m / EARTH_RADIUS_M

    new_lat = math.asin(
        math.sin(lat_rad) * math.cos(ang_dist)
        + math.cos(lat_rad) * math.sin(ang_dist) * math.cos(bearing_rad)
    )
    new_lon = lon_rad + math.atan2(
        math.sin(bearing_rad) * math.sin(ang_dist) * math.cos(lat_rad),
        math.cos(ang_dist) - math.sin(lat_rad) * math.sin(new_lat),
    )

    return math.degrees(new_lat), _normalize_lon(math.degrees(new_lon))


def sample_planar_laplace(epsilon: float, rng: np.random.Generator | None = None) -> Tuple[float, float]:
    """Sample radius and angle for Planar Laplace noise.

    Radius distribution: f(r) = epsilon^2 * r * exp(-epsilon * r)
    which is Gamma(k=2, scale=1/epsilon).
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")

    rng = rng or np.random.default_rng()
    theta = rng.uniform(0.0, 2.0 * math.pi)
    r = rng.gamma(shape=2.0, scale=1.0 / epsilon)
    return r, theta


def apply_geo_indistinguishability(
    lat: float,
    lon: float,
    epsilon: float,
    rng: np.random.Generator | None = None,
) -> Tuple[float, float, float]:
    """Return (noisy_lat, noisy_lon, noise_distance_m)."""
    r, theta = sample_planar_laplace(epsilon, rng)
    noisy_lat, noisy_lon = destination_point(lat, lon, r, theta)
    return noisy_lat, noisy_lon, r


def privacy_radius(epsilon: float) -> float:
    """Compute the expected privacy radius R = 2/epsilon (meters)."""
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    return 2.0 / epsilon
