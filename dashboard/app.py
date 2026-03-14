from pathlib import Path
import sys

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gps_simulator import generate_random_trajectories
from src.privacy_pipeline import apply_noise_to_df
from src.metrics import mean_location_error, max_privacy_radius, service_accuracy, generate_pois
from src.attacks import evaluate_attacks

st.set_page_config(page_title="Geo-Indistinguishability Dashboard", layout="wide")

st.title("Privacy-Preserving Location Sharing Using Geo-Indistinguishability")

with st.sidebar:
    st.header("Data Source")
    data_mode = st.selectbox("Select data source", ["Simulate", "Load sample CSV"])
    epsilon = st.slider("Epsilon (privacy parameter)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
    st.caption("Lower epsilon = stronger privacy (more noise)")

@st.cache_data
def load_data(mode: str) -> pd.DataFrame:
    if mode == "Load sample CSV":
        return pd.read_csv("data/sample_locations.csv")
    return generate_random_trajectories()

raw_df = load_data(data_mode)
noisy_df = apply_noise_to_df(raw_df, epsilon)

center_lat = float(raw_df["latitude"].mean())
center_lon = float(raw_df["longitude"].mean())
pois = generate_pois(center_lat, center_lon)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Map View")
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="CartoDB positron")

    for r in raw_df.itertuples():
        folium.CircleMarker(
            location=[r.latitude, r.longitude],
            radius=2,
            color="blue",
            fill=True,
            fill_opacity=0.6,
        ).add_to(fmap)

    for r in noisy_df.itertuples():
        folium.CircleMarker(
            location=[r.noisy_latitude, r.noisy_longitude],
            radius=2,
            color="red",
            fill=True,
            fill_opacity=0.6,
        ).add_to(fmap)

    for lat, lon in pois:
        folium.Marker(location=[lat, lon], icon=folium.Icon(color="green", icon="info-sign")).add_to(fmap)

    folium_static(fmap, width=900, height=520)

with col2:
    st.subheader("Metrics")
    st.metric("Mean Location Error (m)", f"{mean_location_error(noisy_df):.2f}")
    st.metric("Max Privacy Radius (m)", f"{max_privacy_radius(noisy_df):.2f}")
    st.metric("Service Accuracy", f"{service_accuracy(noisy_df, pois):.2%}")

    attack = evaluate_attacks(noisy_df)
    st.markdown("**Threat Model (Home Inference)**")
    st.write(f"Inference error: {attack['home_inference_error_m']:.2f} m")
    st.markdown("**Threat Model (Trajectory Reconstruction)**")
    st.write(f"Error (noisy): {attack['traj_reconstruction_error_noisy_m']:.2f} m")
    st.write(f"Error (true): {attack['traj_reconstruction_error_true_m']:.2f} m")
    st.markdown("**Threat Model (Clustering)**")
    st.write(f"Top cell share (true): {attack['cluster_top_cell_share_true']:.2%}")
    st.write(f"Top cell share (noisy): {attack['cluster_top_cell_share_noisy']:.2%}")

st.subheader("Privacy-Utility Trade-off")

def compute_tradeoff(df: pd.DataFrame):
    eps_values = [0.1, 0.3, 0.5, 1.0, 2.0]
    records = []
    for eps in eps_values:
        noisy = apply_noise_to_df(df, eps)
        center_lat = float(df["latitude"].mean())
        center_lon = float(df["longitude"].mean())
        pois = generate_pois(center_lat, center_lon)
        records.append(
            {
                "epsilon": eps,
                "mean_error_m": mean_location_error(noisy),
                "max_radius_m": max_privacy_radius(noisy),
                "service_accuracy": service_accuracy(noisy, pois),
                "traj_error_m": evaluate_attacks(noisy)["traj_reconstruction_error_noisy_m"],
            }
        )
    return pd.DataFrame(records)

tradeoff_df = compute_tradeoff(raw_df.copy())

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sns.lineplot(data=tradeoff_df, x="epsilon", y="mean_error_m", marker="o", ax=ax[0])
ax[0].set_title("Epsilon vs Mean Error")
ax[0].set_ylabel("Mean Error (m)")

sns.lineplot(data=tradeoff_df, x="epsilon", y="service_accuracy", marker="o", ax=ax[1])
ax[1].set_title("Epsilon vs Service Accuracy")
ax[1].set_ylabel("Accuracy")

st.pyplot(fig)

fig2, ax2 = plt.subplots(figsize=(5, 4))
sns.lineplot(data=tradeoff_df, x="epsilon", y="traj_error_m", marker="o", ax=ax2)
ax2.set_title("Epsilon vs Trajectory Reconstruction Error")
ax2.set_ylabel("Error (m)")
st.pyplot(fig2)

st.subheader("Privatized Data Preview")
st.dataframe(noisy_df.head(20).astype(str))
