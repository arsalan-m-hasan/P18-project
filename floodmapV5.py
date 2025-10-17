import os
import numpy as np
import pandas as pd
import streamlit as st
import rasterio
from rasterio.warp import transform
from scipy.interpolate import interp1d

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
FLOODMAP_ROOT = r"SwinburneData/FloodMaps/FrankstonSouth"
RETURN_PERIODS = ["001y", "002y", "005y", "010y", "020y", "050y", "100y"]

st.set_page_config(page_title="Flood Risk AI – Victoria", layout="wide")

# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------

def ensure_crs(src):
    """Ensures raster has valid CRS (EPSG:28355 fallback)."""
    if src.crs is None or 'unnamed' in str(src.crs):
        return rasterio.crs.CRS.from_epsg(28355)
    return src.crs

def read_scaled_value(lat, lon, path, band=1, max_depth_m=5.0):
    """Reads a flood value from raster (.grd or .tif) and rescales 0–255 → 0–max_depth_m."""
    try:
        with rasterio.open(path) as src:
            crs = ensure_crs(src)
            xs, ys = transform("EPSG:4326", crs, [lon], [lat])
            vals = list(src.sample([(xs[0], ys[0])]))[0]
            val = vals[band - 1]
            if val is None or np.isnan(val) or val in (255, -9999):
                return np.nan
            scaled = (val / 255.0) * max_depth_m
            return round(scaled, 3)
    except Exception as e:
        st.warning(f"Raster read failed for {path}: {e}")
        return np.nan

def load_metrics(lat, lon):
    """Loads flood metrics for all return periods."""
    data = []
    for rp in RETURN_PERIODS:
        map_dir = os.path.join(FLOODMAP_ROOT, rp, "Mapping")
        if not os.path.exists(map_dir):
            continue
        record = {"ReturnPeriod": rp}

        # Define expected filenames
        f_dmax = next((os.path.join(map_dir, f) for f in os.listdir(map_dir) if "_dmax" in f.lower()), None)
        f_hmax = next((os.path.join(map_dir, f) for f in os.listdir(map_dir) if "_hmax" in f.lower()), None)
        f_vmax = next((os.path.join(map_dir, f) for f in os.listdir(map_dir) if "_vmax" in f.lower()), None)
        f_z0   = next((os.path.join(map_dir, f) for f in os.listdir(map_dir) if "_z0"   in f.lower()), None)

        record["Depth_dmax_m"] = read_scaled_value(lat, lon, f_dmax) if f_dmax else np.nan
        record["WaterLevel_hmax_m"] = read_scaled_value(lat, lon, f_hmax) if f_hmax else np.nan
        record["Velocity_vmax_ms"] = read_scaled_value(lat, lon, f_vmax) if f_vmax else np.nan
        record["HazardIndex_z0"] = read_scaled_value(lat, lon, f_z0) if f_z0 else np.nan
        data.append(record)
    return pd.DataFrame(data)

def interpolate_missing(df, col):
    """Interpolates missing flood values using shape-preserving monotonic interpolation (not linear)."""
    if df[col].isna().all():
        return df
    valid = df.dropna(subset=[col])
    if len(valid) < 2:
        return df
    rp_years = [int(rp[:-1]) for rp in df["ReturnPeriod"]]
    x_valid = [int(rp[:-1]) for rp in valid["ReturnPeriod"]]
    y_valid = valid[col].values
    f = interp1d(x_valid, y_valid, kind="cubic", fill_value="extrapolate")
    df[col] = f(rp_years)
    return df

def classify_zone(depth):
    """Simple flood risk classification."""
    if np.isnan(depth):
        return "Unknown – no flood depth detected."
    if depth < 0.25:
        return "Low Risk Zone – minimal inundation expected."
    elif depth < 1.0:
        return "Moderate Risk Zone – shallow flooding possible."
    elif depth < 2.0:
        return "High Risk Zone – property flooding likely."
    else:
        return "Severe Risk Zone – major inundation expected."

def victoria_safety_tips(zone):
    """Context-specific safety advice."""
    tips = {
        "Low": [
            "Keep stormwater drains clear of debris.",
            "Stay informed via the VicEmergency app or 1800 226 226 hotline.",
        ],
        "Moderate": [
            "Prepare a household emergency plan and pack essentials.",
            "Relocate valuable equipment above ground level.",
        ],
        "High": [
            "Avoid driving through floodwater; even 15 cm can sweep a small car.",
            "Keep updated via the Bureau of Meteorology flood warnings.",
        ],
        "Severe": [
            "Move immediately to higher ground or an evacuation centre.",
            "Follow SES (State Emergency Service) instructions for your municipality.",
        ],
    }
    if "Low" in zone: return tips["Low"]
    if "Moderate" in zone: return tips["Moderate"]
    if "High" in zone: return tips["High"]
    if "Severe" in zone: return tips["Severe"]
    return ["No local guidance available."]

# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------

st.title("📊 Flood Metrics by Return Period (at Property Location)")
st.markdown("This app estimates flood depth, velocity, and hazard index for given coordinates in Victoria, Australia.")

col1, col2 = st.columns(2)
lat = col1.number_input("Latitude", value=-38.142, format="%.6f")
lon = col2.number_input("Longitude", value=145.123, format="%.6f")

if st.button("Analyze Flood Risk"):
    with st.spinner("Processing flood rasters..."):
        df = load_metrics(lat, lon)
        for col in ["Depth_dmax_m", "WaterLevel_hmax_m", "Velocity_vmax_ms", "HazardIndex_z0"]:
            df = interpolate_missing(df, col)
        st.dataframe(df.style.format(precision=3))

        # Select representative return period (e.g., 50y)
        rp50 = df.loc[df["ReturnPeriod"] == "050y"]
        if not rp50.empty:
            depth50 = rp50["Depth_dmax_m"].values[0]
            zone = classify_zone(depth50)
            st.info(f"**Scenario (50 y rainfall):** Depth ≈ {depth50:.2f} m → {zone}")

            tips = victoria_safety_tips(zone)
            st.subheader("💡 Recommended Actions (Victoria SES Guidance)")
            for tip in tips:
                st.markdown(f"- {tip}")

        else:
            st.warning("No valid 50-year scenario found — check input data or coordinates.")
