"""Run benchmark experiments across epsilon values."""

from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt

from src.gps_simulator import generate_random_trajectories
from src.privacy_pipeline import apply_noise_to_df
from src.metrics import mean_location_error, max_privacy_radius, service_accuracy, generate_pois
from src.attacks import evaluate_attacks


def run_benchmark(output_dir: str = "experiments/plots") -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    df = generate_random_trajectories()
    center_lat = float(df["latitude"].mean())
    center_lon = float(df["longitude"].mean())
    pois = generate_pois(center_lat, center_lon)

    eps_values = [0.1, 0.3, 0.5, 1.0, 2.0]
    records = []
    for eps in eps_values:
        noisy = apply_noise_to_df(df, eps)
        attack = evaluate_attacks(noisy)
        records.append(
            {
                "epsilon": eps,
                "mean_error_m": mean_location_error(noisy),
                "max_radius_m": max_privacy_radius(noisy),
                "service_accuracy": service_accuracy(noisy, pois),
                "traj_reconstruction_error_noisy_m": attack["traj_reconstruction_error_noisy_m"],
                "cluster_top_cell_share_noisy": attack["cluster_top_cell_share_noisy"],
            }
        )

    results = pd.DataFrame(records)
    results.to_csv("experiments/benchmark_results.csv", index=False)

    plt.figure(figsize=(6, 4))
    plt.plot(results["epsilon"], results["mean_error_m"], marker="o")
    plt.title("Privacy vs Utility (Mean Error)")
    plt.xlabel("Epsilon")
    plt.ylabel("Mean Error (m)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "epsilon_vs_error.png"))

    plt.figure(figsize=(6, 4))
    plt.plot(results["epsilon"], results["service_accuracy"], marker="o")
    plt.title("Service Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "epsilon_vs_accuracy.png"))

    plt.figure(figsize=(6, 4))
    plt.plot(results["epsilon"], results["traj_reconstruction_error_noisy_m"], marker="o")
    plt.title("Trajectory Reconstruction Error vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Error (m)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "epsilon_vs_traj_error.png"))

    return results


if __name__ == "__main__":
    run_benchmark()
