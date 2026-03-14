"""GPS data simulation utilities."""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta
from typing import Iterable

import pandas as pd

from .geo_noise import destination_point


def _random_walk(
    start_lat: float,
    start_lon: float,
    steps: int,
    step_m: float,
    rng: random.Random,
) -> Iterable[tuple[float, float]]:
    lat, lon = start_lat, start_lon
    bearings = [
        0.0,
        math.pi / 4.0,
        math.pi / 2.0,
        3.0 * math.pi / 4.0,
        math.pi,
        5.0 * math.pi / 4.0,
        3.0 * math.pi / 2.0,
        7.0 * math.pi / 4.0,
    ]
    for _ in range(steps):
        bearing = rng.choice(bearings)
        lat, lon = destination_point(lat, lon, step_m, bearing)
        yield lat, lon


def simulate_city_grid(
    center_lat: float,
    center_lon: float,
    num_users: int = 5,
    points_per_user: int = 100,
    step_m: float = 60.0,
    seed: int = 42,
    start_time: datetime | None = None,
    time_step_s: int = 60,
) -> pd.DataFrame:
    """Simulate user trajectories using random walks across a city grid."""
    rng = random.Random(seed)
    start_time = start_time or datetime.utcnow()

    records: list[dict] = []
    for user_id in range(1, num_users + 1):
        lat = center_lat + rng.uniform(-0.01, 0.01)
        lon = center_lon + rng.uniform(-0.01, 0.01)
        timestamp = start_time
        for lat, lon in _random_walk(lat, lon, points_per_user, step_m, rng):
            records.append(
                {
                    "user_id": f"user_{user_id}",
                    "timestamp": timestamp.isoformat(),
                    "latitude": lat,
                    "longitude": lon,
                }
            )
            timestamp += timedelta(seconds=time_step_s)

    return pd.DataFrame.from_records(records)


def generate_random_trajectories(
    city_center: tuple[float, float] = (37.7749, -122.4194),
    num_users: int = 5,
    points_per_user: int = 100,
    step_m: float = 60.0,
    seed: int = 42,
) -> pd.DataFrame:
    return simulate_city_grid(
        center_lat=city_center[0],
        center_lon=city_center[1],
        num_users=num_users,
        points_per_user=points_per_user,
        step_m=step_m,
        seed=seed,
    )


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
