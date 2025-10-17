# floodai_app_v4.py
import os, re, datetime, math
import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.warp import transform
from shapely.geometry import Point
import folium
from streamlit_folium import st_folium


# -----------------------------
# PATHS (edit if your tree differs)
# -----------------------------
PROPERTY_GPKG   = "SwinburneData/Property/Properties.gpkg"
FLOODMAP_ROOT   = "SwinburneData/FloodMaps/FrankstonSouth"

# Frankston South rasters you showed (001..050 in Mapping; 100 in Final, no Vmax/Z0)
RASTER_MAP = {
    "001y": {
        "dmax": "001y/Mapping/FS_001_001y_010m_dmax.grd",
        "hmax": "001y/Mapping/FS_001_001y_010m_hmax.grd",
        "vmax": "001y/Mapping/FS_001_001y_010m_Vmax.grd",
        "z0max":"001y/Mapping/FS_001_001y_010m_Z0max.grd",
    },
    "002y": {
        "dmax": "002y/Mapping/FS_001_002y_010m_dmax.grd",
        "hmax": "002y/Mapping/FS_001_002y_010m_hmax.grd",
        "vmax": "002y/Mapping/FS_001_002y_010m_Vmax.grd",
        "z0max":"002y/Mapping/FS_001_002y_010m_Z0max.grd",
    },
    "005y": {
        "dmax": "005y/Mapping/FS_001_005y_010m_dmax.grd",
        "hmax": "005y/Mapping/FS_001_005y_010m_hmax.grd",
        "vmax": "005y/Mapping/FS_001_005y_010m_Vmax.grd",
        "z0max":"005y/Mapping/FS_001_005y_010m_Z0max.grd",
    },
    "010y": {
        "dmax": "010y/Mapping/FS_001_010y_010m_dmax.grd",
        "hmax": "010y/Mapping/FS_001_010y_010m_hmax.grd",
        "vmax": "010y/Mapping/FS_001_010y_010m_Vmax.grd",
        "z0max":"010y/Mapping/FS_001_010y_010m_Z0max.grd",
    },
    "020y": {
        "dmax": "020y/Mapping/FS_001_020y_010m_dmax.grd",
        "hmax": "020y/Mapping/FS_001_020y_010m_hmax.grd",
        "vmax": "020y/Mapping/FS_001_020y_010m_Vmax.grd",
        "z0max":"020y/Mapping/FS_001_020y_010m_Z0max.grd",
    },
    "050y": {
        "dmax": "050y/Mapping/FS_001_050y_010m_dmax.grd",
        "hmax": "050y/Mapping/FS_001_050y_010m_hmax.grd",
        "vmax": "050y/Mapping/FS_001_050y_010m_Vmax.grd",
        "z0max":"050y/Mapping/FS_001_050y_010m_Z0max.grd",
    },
    # 100y is special (no vmax/z0max; different names)
    "100y": {
        "dmax": "100y/Final/FS_100y_010m_102_d(maxmax)_g002.grd",
        "hmax": "100y/Final/FS_100y_010m_102_h(maxmax)_g002.grd",
        # optional: one 2dm surface exists but not a raster velocity; we skip vmax/z0 for 100y
    }
}

YEARS_ORDER = ["001y","002y","005y","010y","020y","050y","100y"]
YEAR_VALS   = np.array([1,2,5,10,20,50,100], dtype=float)

# -----------------------------
# Streamlit setup
# -----------------------------
st.set_page_config(page_title="FloodAI — MVP v4", layout="wide")
st.title("🌧️ FloodAI — MVP v4 (GPKG + Rasters + Clarifications + Zones)")

# -----------------------------
# Caching loaders
# -----------------------------
@st.cache_data(show_spinner=False)
def load_properties() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(PROPERTY_GPKG)
    # Expect CRS EPSG:7855 in your data; keep native for metric ops, also keep WGS84 for mapping
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=7855)
    # Make a display-friendly address
    def _safe(s): return s.fillna("").astype(str).str.strip()
    gdf["Full_Address"] = (
        _safe(gdf.get("House", pd.Series([""]*len(gdf)))) + " " +
        _safe(gdf.get("Street", pd.Series([""]*len(gdf)))) + ", " +
        _safe(gdf.get("Suburb", pd.Series([""]*len(gdf)))) + " VIC " +
        _safe(gdf.get("Postcode", pd.Series([""]*len(gdf))))
    ).str.replace(r"\s+,", ",", regex=True).str.replace(r",\s+VIC\s+$"," VIC", regex=True)
    return gdf

props_gdf = load_properties()
st.caption(f"📍 Properties loaded: {len(props_gdf):,} | CRS: {props_gdf.crs}")

# @st.cache_data(show_spinner=False)
# def props_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
#     return gdf.to_crs(epsg=4326)



@st.cache_data(show_spinner=False)
def props_wgs84(_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Convert property GeoDataFrame to WGS84 for mapping (avoid hashing error)."""
    return _gdf.to_crs(epsg=4326)
props_gdf_wgs = props_wgs84(props_gdf)

# -----------------------------
# Parsing helpers (robust + clarifications)
# -----------------------------
def parse_rain(text:str):
    m = re.search(r'(\d+(?:\.\d+)?)\s*(mm|millimet(re|er|res|ers)?)', text, re.I)
    return float(m.group(1)) if m else None

def parse_duration(text:str):
    if m := re.search(r'(\d+(?:\.\d+)?)\s*(hour|hr|h|hours)', text, re.I):
        return float(m.group(1))
    if m := re.search(r'(\d+(?:\.\d+)?)\s*(minute|min|mins|m)', text, re.I):
        return float(m.group(1))/60.0
    return None

def parse_coords(text:str):
    # coordinate pair anywhere (lon/lat or lat/lon)
    m = re.search(r'(-?\d{1,3}\.\d+)[,\s]+(-?\d{1,3}\.\d+)', text)
    if not m:
        return None
    a, b = float(m.group(1)), float(m.group(2))
    # Heuristic: latitude ~ -38; longitude ~ 145 in Melbourne.
    if abs(a) < 90 and abs(b) > 90:
        return (a, b)  # (lat, lon)
    if abs(b) < 90 and abs(a) > 90:
        return (b, a)
    # if ambiguous, assume (lat,lon)
    return (a, b)

def parse_address_hint(text:str):
    # Very light rule-based extraction of “near <something>” or trailing phrase
    m = re.search(r'(?:near|in|at|around)\s+([A-Za-z0-9 ,\-\/]+)', text, re.I)
    return m.group(1).strip() if m else None

def clarification_needed(rain, dur, coords, addr_hint):
    need = []
    if rain is None: need.append("rainfall (mm)")
    if dur is None:  need.append("duration")
    if coords is None and not addr_hint: need.append("location (address or coordinates)")
    return need

# -----------------------------
# Property locator (no recursion, EPSG:7855 native)
# -----------------------------
def best_property(addr_hint: str | None, latlon: tuple[float, float] | None):
    gdf = props_gdf

    # --- 1. Coordinate-based search (works fine as-is)
    if latlon:
        lat, lon = latlon
        pt = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(gdf.crs).iloc[0]
        gdf["centroid_tmp"] = gdf.geometry.centroid
        gdf["dist"] = gdf["centroid_tmp"].distance(pt)
        row = gdf.loc[gdf["dist"].idxmin()]
        cen_wgs = gpd.GeoSeries([row["centroid_tmp"]], crs=gdf.crs).to_crs(4326).iloc[0]
        return {
            "match_type": "coords",
            "Full_Address": row.get("Full_Address"),
            "Suburb": row.get("Suburb"),
            "Postcode": row.get("Postcode"),
            "lat": cen_wgs.y,
            "lon": cen_wgs.x,
            "prop_idx": int(row.name),
        }

    # --- 2. Address-text-based search
    if addr_hint:
        # Normalize hint
        hint = addr_hint.upper().strip()

        # Split into parts (street name, house number, suburb)
        parts = re.split(r"[ ,]+", hint)
        num = None
        for p in parts:
            if re.fullmatch(r"\d+[A-Z]?", p):
                num = p
                break

        # Find suburb match
        suburb_candidates = gdf[gdf["Suburb"].astype(str).str.upper().str.contains("FRANKSTON", na=False)]
        # Use entire GDF if suburb missing
        candidates = suburb_candidates if not suburb_candidates.empty else gdf

        # Street match
        for word in parts:
            matches = candidates[candidates["Street"].astype(str).str.upper().str.contains(word, na=False)]
            if not matches.empty:
                candidates = matches
                break

        # House match
        if num:
            candidates = candidates[candidates["House"].astype(str).str.upper() == num]

        # Return closest centroid match if multiple
        if not candidates.empty:
            row = candidates.iloc[0]
            cen_wgs = gpd.GeoSeries([row.geometry.centroid], crs=gdf.crs).to_crs(4326).iloc[0]
            return {
                "match_type": "address",
                "Full_Address": row.get("Full_Address"),
                "Suburb": row.get("Suburb"),
                "Postcode": row.get("Postcode"),
                "lat": cen_wgs.y,
                "lon": cen_wgs.x,
                "prop_idx": int(row.name),
            }

    return None


# -----------------------------
# Raster sampling (robust band pick)
# -----------------------------
def sample_raster_value(lat, lon, raster_relpath, prefer_band=4):
    fpath = os.path.join(FLOODMAP_ROOT, raster_relpath)
    if not os.path.exists(fpath):
        return np.nan
    try:
        with rasterio.open(fpath) as src:
            # Choose a plausible band (4 is common in your grids, else use last)
            band = prefer_band if prefer_band <= src.count else src.count
            # reproject point to raster CRS
            xs, ys = transform("EPSG:4326", src.crs, [lon], [lat])
            val = list(src.read(band, masked=True).reshape(-1))[0]  # force array open
            # sample via window (faster approach: src.sample)
            s = list(src.sample([(xs[0], ys[0])]))[0][band-1]
            # mask sentinel values
            if s is None: return np.nan
            s = float(s)
            if s in (255.0, -9999.0) or s < 0:
                return np.nan
            # many engineer grids store cm; if depths look huge, scale down:
            if s > 20:  # heuristic: >20m likely encoded (e.g., cm)
                s = s/100.0
            return round(s, 3)
    except Exception:
        return np.nan

def collect_metrics_for_point(lat, lon):
    metrics = {y: {"dmax": np.nan, "hmax": np.nan, "vmax": np.nan, "z0max": np.nan} for y in YEARS_ORDER}
    for y in YEARS_ORDER:
        files = RASTER_MAP.get(y, {})
        # 100y may only have dmax & hmax
        if "dmax" in files:
            metrics[y]["dmax"] = sample_raster_value(lat, lon, files["dmax"], prefer_band=4)
        if "hmax" in files:
            metrics[y]["hmax"] = sample_raster_value(lat, lon, files["hmax"], prefer_band=4)
        if "vmax" in files:
            metrics[y]["vmax"] = sample_raster_value(lat, lon, files["vmax"], prefer_band=4)
        if "z0max" in files:
            metrics[y]["z0max"] = sample_raster_value(lat, lon, files["z0max"], prefer_band=4)
    return metrics

# Non-linear interpolation in log-space of return period (years)
def interp_log_year(metric_by_year: dict[str, float]) -> dict[str, float]:
    # Build arrays of available points
    xs = []
    ys = []
    for ytag, val in metric_by_year.items():
        y = int(ytag.replace("y",""))
        if not np.isnan(val):
            xs.append(y)
            ys.append(val)
    if len(xs) < 2:
        return metric_by_year  # not enough to interpolate
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    # interpolate on log(year) → log(y+ε?) we keep ys linear, xs in log to avoid simple linear
    logx = np.log(xs)
    for ytag in metric_by_year.keys():
        y = int(ytag.replace("y",""))
        if np.isnan(metric_by_year[ytag]):
            metric_by_year[ytag] = float(np.interp(np.log(y), logx, ys))
    return metric_by_year

def summarize_risk(dmax_m: float|None, vmax_ms: float|None) -> str:
    if dmax_m is None or np.isnan(dmax_m): return "Unknown risk — no flood depth at this location."
    if dmax_m < 0.05: return "Very low risk — nuisance ponding only."
    if dmax_m < 0.3:  return "Low risk — shallow overland flow possible."
    if dmax_m < 0.6:  return "Moderate risk — curb-depth flow; avoid driving."
    if dmax_m < 1.0:  return "High risk — building ingress possible; avoid floodwater."
    return "Extreme risk — life-threatening conditions possible."

def safety_tips_vic() -> list[str]:
    return [
        "Never enter floodwater — it can be fast-moving or contaminated.",
        "Do not drive through floodwater. As little as 15 cm can float a small car.",
        "Stay informed via VicEmergency app/website and ABC local radio.",
        "For SES flood/storm assistance call 132 500. Call 000 in life-threatening emergencies.",
        "Move vehicles and valuables to higher ground if safe to do so.",
        "Turn off electricity/gas at the mains if flooding is imminent and you can do it safely.",
        "Prepare sandbags to protect low doorways and vents; check council distribution points.",
        "Keep an emergency kit: torch, radio, spare batteries, meds, water, documents.",
        "Avoid walking near drains/culverts; covers may be displaced.",
        "After flooding, assume water is contaminated; photograph damage for insurance."
    ]

# -----------------------------
# UI — Input & Clarifications
# -----------------------------
with st.container():
    st.subheader("💬 Describe the rainfall/flood observation")
    default_txt = "Severe rainfall of 60 mm in Frankston South for 45 minutes"
    user_txt = st.text_area("Free text", height=90, value=default_txt)

    rain = parse_rain(user_txt)
    dur  = parse_duration(user_txt)
    coords = parse_coords(user_txt)
    addr_hint = parse_address_hint(user_txt)

    missing = clarification_needed(rain, dur, coords, addr_hint)
    if missing:
        st.warning("I need a bit more info to proceed:")
        col1,col2,col3 = st.columns(3)
        with col1:
            rain = st.number_input("Rainfall (mm)", value=rain if rain else 50.0, min_value=0.0, step=1.0)
        with col2:
            dur = st.number_input("Duration (hours)", value=dur if dur else 1.0, min_value=0.0, step=0.25)
        with col3:
            mode = st.radio("Location input", ["Address hint","Coordinates"], index=0 if not coords else 1, horizontal=True)
        if mode=="Coordinates":
            lat = st.number_input("Latitude (e.g., -38.15)", value=coords[0] if coords else -38.1539, step=0.0001, format="%.6f")
            lon = st.number_input("Longitude (e.g., 145.10)", value=coords[1] if coords else 145.1038, step=0.0001, format="%.6f")
            coords = (lat, lon)
            addr_hint = None
        else:
            addr_hint = st.text_input("Address / Street / Suburb", value=addr_hint or "Frankston South")
            coords = None

    st.divider()

# -----------------------------
# Property resolution
# -----------------------------
prop = best_property(addr_hint, coords)
if not prop:
    st.error("I couldn't resolve a property from that input. Please provide a clearer address or coordinates.")
    st.stop()

st.success(f"📍 Matched property: **{prop['Full_Address']}**")
st.caption(f"Lat/Lon: {prop['lat']:.6f}, {prop['lon']:.6f} | Suburb: {prop['Suburb']} {prop['Postcode']} | via {prop['match_type']}")

# -----------------------------
# Flood metrics for this point
# -----------------------------
st.subheader("📈 Flood metrics by return period (at property location)")
raw = collect_metrics_for_point(prop["lat"], prop["lon"])

# Interpolate non-linearly (log years) for each metric
interp = {}
for mkey in ["dmax","hmax","vmax","z0max"]:
    series = {y: raw[y][mkey] for y in YEARS_ORDER}
    interp[mkey] = interp_log_year(series)

df = pd.DataFrame({
    "ReturnPeriod": YEARS_ORDER,
    "Depth_dmax_m": [interp["dmax"][y] for y in YEARS_ORDER],
    "WaterLevel_hmax_m": [interp["hmax"][y] for y in YEARS_ORDER],
    "Velocity_vmax_ms": [interp["vmax"][y] for y in YEARS_ORDER],
    "HazardIndex_z0": [interp["z0max"][y] for y in YEARS_ORDER],
})
st.dataframe(df, use_container_width=True)

# Risk summary chosen at nearest “matching” scenario (by rainfall)
def choose_scenario_from_rain(r):
    if r is None: return "001y"
    if r < 20: return "005y"
    if r < 40: return "010y"
    if r < 60: return "020y"
    if r < 80: return "050y"
    return "100y"

scen = choose_scenario_from_rain(rain)
d_here = interp["dmax"].get(scen)
v_here = interp["vmax"].get(scen)
st.info(f"Scenario selected from rainfall: **{scen}** → Depth ≈ **{d_here if not np.isnan(d_here) else 'N/A'} m**, Velocity ≈ **{v_here if not np.isnan(v_here) else 'N/A'} m/s**")
st.warning(summarize_risk(d_here, v_here))

# -----------------------------
# Map with hazard zones
# -----------------------------
st.subheader("🗺️ Map & Safety Zones")
m = folium.Map(location=[prop["lat"], prop["lon"]], zoom_start=15)
popup = f"{prop['Full_Address']}<br><b>{scen}</b>: d={d_here if not np.isnan(d_here) else 'N/A'} m"
folium.Marker([prop["lat"], prop["lon"]], popup=popup, icon=folium.Icon(color="red")).add_to(m)
# zones: 100m, 250m, 500m (buffer in meters requires projected CRS → approximate with folium radius)
for radius, col in [(100, "#FF0000"), (250, "#FFA500"), (500, "#3388ff")]:
    folium.Circle(
        location=[prop["lat"], prop["lon"]],
        radius=radius,
        color=col, fill=True, fill_opacity=0.08, weight=1,
        tooltip=f"Zone {radius} m"
    ).add_to(m)
st_folium(m, height=520, use_container_width=True)

# -----------------------------
# Sentiment + Safety message
# -----------------------------
st.subheader("🧠 Advisory")
sev = "severe" if (rain and rain>=60) or (d_here and not np.isnan(d_here) and d_here>=0.6) else "moderate" if (rain and rain>=20) else "low"
st.write(f"**Overall sentiment:** {('🚨 High Concern' if sev=='severe' else '⚠️ Elevated Caution' if sev=='moderate' else '✅ Low Concern')}")

with st.expander("Safety guidance (Victoria / SES-aligned)"):
    for tip in safety_tips_vic():
        st.markdown(f"- {tip}")

# -----------------------------
# Final summary
# -----------------------------
st.divider()
st.markdown("### 📋 Summary")
st.markdown(f"""
- **Rainfall reported:** {rain if rain is not None else 'Not provided'} mm  
- **Duration:** {dur if dur is not None else 'Not provided'} h  
- **Matched property:** {prop['Full_Address']}  
- **Chosen scenario:** {scen}  
- **Estimated depth:** {d_here if d_here is not None and not np.isnan(d_here) else 'No data'} m  
- **Estimated velocity:** {v_here if v_here is not None and not np.isnan(v_here) else 'No data'} m/s  
- **Risk level:** {('Extreme/High' if sev=='severe' else 'Moderate' if sev=='moderate' else 'Low')}  
- **Timestamp:** {datetime.datetime.now().isoformat(timespec='seconds')}
""")
