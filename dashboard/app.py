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
    poi_mode = st.selectbox("POI source", ["Simulated POIs", "Bangalore OSM POIs"])
    load_pois = st.button("Load OSM POIs")
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
        return pd.read_csv("data/sample_locations.csv")
    return generate_random_trajectories(seed=seed)

raw_df = load_data(data_mode, int(seed))
noisy_df = apply_noise_to_df(raw_df, epsilon, seed=int(seed))

center_lat = float(raw_df["latitude"].mean())
center_lon = float(raw_df["longitude"].mean())
if poi_mode == "Bangalore OSM POIs" and load_pois:
    try:
        pois = fetch_bangalore_pois((12.9716, 77.5946))
        if not pois:
            st.warning("OSM POIs returned empty. Falling back to simulated POIs.")
            pois = generate_pois(center_lat, center_lon)
    except Exception:
        st.warning("OSM POIs unavailable (rate limit or network). Falling back to simulated POIs.")
        pois = generate_pois(center_lat, center_lon)
else:
    pois = generate_pois(center_lat, center_lon)

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

@st.cache_data(ttl=3600)
def fetch_bangalore_pois(center: tuple[float, float]) -> list[tuple[float, float]]:
    overpass_url = "https://overpass-api.de/api/interpreter"
    lat, lon = center
    query = f"""
    [out:json][timeout:25];
    (
      node["amenity"~"hospital|restaurant|school|bank"](around:15000,{lat},{lon});
      node["tourism"="attraction"](around:15000,{lat},{lon});
      node["shop"="mall"](around:15000,{lat},{lon});
    );
    out 200;
    """
    headers = {"User-Agent": "geo-mask-proj/1.0 (research demo)"}
    resp = requests.get(overpass_url, params={"data": query}, headers=headers, timeout=25)
    resp.raise_for_status()
    data = resp.json()
    pois = []
    for el in data.get("elements", []):
        lat = el.get("lat")
        lon = el.get("lon")
        if lat is not None and lon is not None:
            pois.append((lat, lon))
    return pois

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

st.subheader("Additional Visuals")
tab1, tab2 = st.tabs(["Error Distributions", "Error vs Epsilon"])

with tab1:
    errors = [
        haversine_m(r.latitude, r.longitude, r.noisy_latitude, r.noisy_longitude)
        for r in noisy_df.itertuples()
    ]
    fig3, ax3 = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(errors, bins=30, ax=ax3[0], color="#d62728")
    ax3[0].set_title("Noise Distance Histogram")
    ax3[0].set_xlabel("Distance (m)")
    ax3[0].set_ylabel("Count")

    sorted_err = np.sort(errors)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err) if len(sorted_err) else []
    ax3[1].plot(sorted_err, cdf, color="#1f77b4")
    ax3[1].set_title("Noise Distance CDF")
    ax3[1].set_xlabel("Distance (m)")
    ax3[1].set_ylabel("CDF")
    st.pyplot(fig3)

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
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=err_df, x="epsilon", y="error_m", ax=ax4)
    ax4.set_title("Location Error Distribution by Epsilon")
    ax4.set_xlabel("Epsilon")
    ax4.set_ylabel("Error (m)")
    st.pyplot(fig4)

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
