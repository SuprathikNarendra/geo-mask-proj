from pathlib import Path
import sys
import time

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import seaborn as sns
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gps_simulator import generate_random_trajectories
from src.privacy_pipeline import apply_noise_to_df
from src.geo_noise import apply_geo_indistinguishability
from src.metrics import mean_location_error, max_privacy_radius, service_accuracy, generate_pois, haversine_m
from src.attacks import evaluate_attacks

st.set_page_config(page_title="Geo-Indistinguishability Dashboard", layout="wide")

st.title("Privacy-Preserving Location Sharing Using Geo-Indistinguishability")
st.caption("Live demo: enter a current location and destination in Bangalore, then mask the current location based on epsilon.")

with st.sidebar:
    st.header("Data Source")
    data_mode = st.selectbox("Select data source", ["Simulate", "Load sample CSV"])
    seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1)
    poi_mode = st.selectbox("POI source", ["Simulated POIs", "Bangalore Cached POIs"])
    poi_count = st.slider("Simulated POI count", min_value=20, max_value=300, value=120, step=20)
    privacy_mode = st.selectbox("Privacy control", ["Set epsilon", "Set radius (meters)"])
    if privacy_mode == "Set epsilon":
        epsilon = st.slider("Epsilon (privacy parameter)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
        st.caption("Lower epsilon = stronger privacy (more noise)")
    else:
        radius_m = st.slider("Approx. privacy radius (m)", min_value=100, max_value=5000, value=800, step=100)
        epsilon = 2.0 / float(radius_m)
        st.caption(f"Epsilon set to {epsilon:.3f} for ~{radius_m} m radius")

@st.cache_data
def load_data(mode: str, seed: int) -> pd.DataFrame:
    if mode == "Load sample CSV":
        return pd.read_csv(ROOT / "data" / "sample_locations.csv")
    return generate_random_trajectories(seed=seed)

@st.cache_data
def load_bangalore_pois() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "bangalore_pois.csv")


def nearest_poi_info(lat: float, lon: float, poi_rows: list[dict]) -> dict | None:
    if not poi_rows:
        return None
    best = None
    best_dist = float("inf")
    for row in poi_rows:
        d = haversine_m(lat, lon, row["lat"], row["lon"])
        if d < best_dist:
            best_dist = d
            best = {**row, "distance_m": d}
    return best

@st.cache_data(ttl=3600)
def geocode_place(query: str) -> tuple[float, float] | None:
    if not query.strip():
        return None
    time.sleep(1.0)
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": query, "format": "json", "limit": 1}
    headers = {"User-Agent": "geo-mask-proj/1.0 (research demo)"}
    resp = requests.get(url, params=params, headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return None
    return float(data[0]["lat"]), float(data[0]["lon"])

raw_df = load_data(data_mode, int(seed))
noisy_df = apply_noise_to_df(raw_df, epsilon, seed=int(seed))

center_lat = float(raw_df["latitude"].mean())
center_lon = float(raw_df["longitude"].mean())
poi_rows = []
if poi_mode == "Bangalore Cached POIs":
    try:
        df_pois = load_bangalore_pois()
        poi_rows = [
            {
                "name": row["name"],
                "category": row["category"],
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
            }
            for _, row in df_pois.iterrows()
        ]
    except Exception as exc:
        st.warning("Cached POIs unavailable. Falling back to simulated POIs.")
        st.error(f"POI load error: {exc}")
        poi_rows = []

if not poi_rows:
    sim_pois = generate_pois(center_lat, center_lon, count=int(poi_count), seed=int(seed))
    poi_rows = [
        {"name": f"Sim POI {i+1}", "category": "simulated", "lat": lat, "lon": lon}
        for i, (lat, lon) in enumerate(sim_pois)
    ]

pois = [(row["lat"], row["lon"]) for row in poi_rows]

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Map View (Bangalore)")
    default_center = (12.9716, 77.5946)

    with st.expander("Live Bangalore Demo (Current → Destination)", expanded=True):
        origin_text = st.text_input("Current location (Bangalore)", value="MG Road, Bangalore")
        dest_text = st.text_input("Destination (Bangalore)", value="Koramangala, Bangalore")
        st.caption("Current location is masked; destination is shown as entered.")

    origin = geocode_place(origin_text + ", Bangalore, India")
    dest = geocode_place(dest_text + ", Bangalore, India")

    demo_center = default_center
    if origin:
        demo_center = origin

    fmap = folium.Map(location=[demo_center[0], demo_center[1]], zoom_start=12, tiles="CartoDB positron")

    if origin:
        noisy_origin_lat, noisy_origin_lon, _ = apply_geo_indistinguishability(
            origin[0], origin[1], epsilon
        )
        folium.Marker(
            location=[origin[0], origin[1]],
            popup="True Current Location",
            icon=folium.Icon(color="blue", icon="user"),
        ).add_to(fmap)
        folium.Marker(
            location=[noisy_origin_lat, noisy_origin_lon],
            popup="Masked Current Location",
            icon=folium.Icon(color="red", icon="eye-slash"),
        ).add_to(fmap)
        folium.PolyLine(
            locations=[origin, (noisy_origin_lat, noisy_origin_lon)],
            color="red",
            weight=2,
            opacity=0.7,
        ).add_to(fmap)

        true_poi = nearest_poi_info(origin[0], origin[1], poi_rows)
        noisy_poi = nearest_poi_info(noisy_origin_lat, noisy_origin_lon, poi_rows)
        if true_poi and noisy_poi:
            st.markdown("**Nearest POI Comparison**")
            st.write(
                f"True location → {true_poi['name']} ({true_poi['category']}), "
                f"{true_poi['distance_m']:.0f} m"
            )
            st.write(
                f"Masked location → {noisy_poi['name']} ({noisy_poi['category']}), "
                f"{noisy_poi['distance_m']:.0f} m"
            )
    else:
        st.warning("Origin not found. Try a more specific Bangalore location.")

    if dest:
        folium.Marker(
            location=[dest[0], dest[1]],
            popup="Destination (Unmasked)",
            icon=folium.Icon(color="green", icon="flag"),
        ).add_to(fmap)
    else:
        st.warning("Destination not found. Try a more specific Bangalore location.")

    st.markdown("---")
    st.subheader("Simulated Data Overlay")

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
    if len(pois) < 20:
        st.info("Service accuracy may appear high with very few POIs. Increase POI count for stronger signal.")

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
        pois = generate_pois(center_lat, center_lon, count=120, seed=int(seed))
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

col_t1, col_t2, col_t3 = st.columns(3)
with col_t1:
    fig, ax = plt.subplots(figsize=(3.6, 3))
    sns.lineplot(data=tradeoff_df, x="epsilon", y="mean_error_m", marker="o", ax=ax)
    ax.set_title("Epsilon vs Mean Error")
    ax.set_ylabel("Mean Error (m)")
    ax.set_xlabel("Epsilon")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

with col_t2:
    fig, ax = plt.subplots(figsize=(3.6, 3))
    sns.lineplot(data=tradeoff_df, x="epsilon", y="service_accuracy", marker="o", ax=ax)
    ax.set_title("Epsilon vs Service Accuracy")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Epsilon")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

with col_t3:
    fig, ax = plt.subplots(figsize=(3.6, 3))
    sns.lineplot(data=tradeoff_df, x="epsilon", y="traj_error_m", marker="o", ax=ax)
    ax.set_title("Epsilon vs Trajectory Error")
    ax.set_ylabel("Error (m)")
    ax.set_xlabel("Epsilon")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

st.subheader("Privatized Data Preview")
st.dataframe(noisy_df.head(20).astype(str))

st.subheader("Additional Visuals")
tab1, tab2 = st.tabs(["Error Distributions", "Error vs Epsilon"])

with tab1:
    errors = [
        haversine_m(r.latitude, r.longitude, r.noisy_latitude, r.noisy_longitude)
        for r in noisy_df.itertuples()
    ]
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        fig, ax = plt.subplots(figsize=(3.8, 3))
        sns.histplot(errors, bins=30, ax=ax, color="#d62728")
        ax.set_title("Noise Distance Histogram")
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Count")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col_e2:
        fig, ax = plt.subplots(figsize=(3.8, 3))
        sorted_err = np.sort(errors)
        cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err) if len(sorted_err) else []
        ax.plot(sorted_err, cdf, color="#1f77b4")
        ax.set_title("Noise Distance CDF")
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("CDF")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

with tab2:
    eps_values = [0.1, 0.3, 0.5, 1.0, 2.0]
    samples = []
    for eps in eps_values:
        noisy = apply_noise_to_df(raw_df, eps, seed=int(seed))
        for r in noisy.itertuples():
            samples.append(
                {
                    "epsilon": eps,
                    "error_m": haversine_m(r.latitude, r.longitude, r.noisy_latitude, r.noisy_longitude),
                }
            )
    err_df = pd.DataFrame(samples)
    fig, ax = plt.subplots(figsize=(6.5, 3))
    sns.boxplot(data=err_df, x="epsilon", y="error_m", ax=ax)
    ax.set_title("Location Error Distribution by Epsilon")
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Error (m)")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

st.subheader("Downloads")
metrics_summary = {
    "epsilon": epsilon,
    "mean_location_error_m": mean_location_error(noisy_df),
    "max_privacy_radius_m": max_privacy_radius(noisy_df),
    "service_accuracy": service_accuracy(noisy_df, pois),
}
metrics_df = pd.DataFrame([metrics_summary])
st.download_button(
    "Download noisy locations (CSV)",
    noisy_df.to_csv(index=False).encode("utf-8"),
    "noisy_locations.csv",
    "text/csv",
)
st.download_button(
    "Download metrics summary (CSV)",
    metrics_df.to_csv(index=False).encode("utf-8"),
    "metrics_summary.csv",
    "text/csv",
)
