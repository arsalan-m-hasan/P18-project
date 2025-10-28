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
from math import radians, sin, cos, sqrt, atan2  
from typing import Optional, Dict


# -----------------------------
PROPERTY_GPKG   = "SwinburneData/Property/Properties.gpkg"
FLOODMAP_ROOT   = "SwinburneData/FloodMaps/FrankstonSouth"

SUBCATCHMENT_GPKG = "SwinburneData/Subcatchments/Subcatchments.gpkg"
subcatchments_gdf = gpd.read_file(SUBCATCHMENT_GPKG)
subcatchments_gdf = subcatchments_gdf.to_crs(epsg=4326)

CANON_AEP_COLS = ['63.20%', '50%', '20%', '10%', '5%', '2%', '1%']
CANON_ARI = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0], dtype=float)

def _haversine_km(lat1, lon1, lat2, lon2):
    """Return distance in km between two points."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2.0)**2 + cos(lat1) * cos(lat2) * sin(dlon/2.0)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def _map_aep_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to find/rename AEP columns in df to canonical names in CANON_AEP_COLS.
    Works with variations like '63.2%', '63_20', '63.20', '63_2', 'p63.2' etc.
    """
    cols = list(df.columns)
    lowcols = [c.lower() for c in cols]

    mapping = {}
    # patterns for each canonical key (ordered to prefer more specific matches)
    patterns = {
        '63.20%': ['63.20', '63.2', '63_20', '63_2', '63'],
        '50%': ['50'],
        '20%': ['20'],
        '10%': ['10'],
        '5%': ['5'],
        '2%': ['2'],
        '1%': ['1']
    }

    used = set()
    for canon, pats in patterns.items():
        found = None
        for pat in pats:
            # match substring excluding common words like 'station' etc.
            for orig, low in zip(cols, lowcols):
                if orig in used:
                    continue
                if pat in low and ('%' in orig or pat in low.split('_') or pat in low):
                    found = orig
                    break
            if found:
                break
        # fallback: try any column that contains the number as substring
        if not found:
            for orig, low in zip(cols, lowcols):
                if orig in used:
                    continue
                if any(ch.isdigit() for ch in pat):
                    if pat in low:
                        found = orig
                        break
        if found:
            mapping[found] = canon
            used.add(found)

    # Final verification: require all canonical columns present
    mapped_values = list(mapping.values())
    missing = [c for c in CANON_AEP_COLS if c not in mapped_values]
    if missing:
        # Try case where csv already has canonical names present
        available = [c for c in cols if c in CANON_AEP_COLS]
        for c in available:
            if c not in mapped_values:
                mapping[c] = c
        mapped_values = list(mapping.values())
        missing = [c for c in CANON_AEP_COLS if c not in mapped_values]

    if missing:
        raise ValueError(f"Could not auto-detect AEP columns for: {missing}. Columns found: {cols}")

    # rename safely (only those mapped)
    df = df.copy()
    df.rename(columns=mapping, inplace=True)
    return df


def load_ifd_data(filepath: str) -> pd.DataFrame:
    """
    Load IFD CSV and normalise column names and AEP columns to CANON_AEP_COLS.
    Expected minimal columns after cleaning: station_id, lat, lon, duration_in_min, (AEP columns)
    """
    df = pd.read_csv(filepath)
    # basic normalisation for column keys similar to your app
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
    # attempt to rename & standardise AEP columns
    df = _map_aep_columns(df)
    # ensure required columns exist
    req = ['station_id', 'lat', 'lon', 'duration_in_min']
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"IFD CSV missing required columns: {missing}")
    # force numeric
    df['duration_in_min'] = pd.to_numeric(df['duration_in_min'], errors='coerce')
    for col in CANON_AEP_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # drop invalid rows
    df = df.dropna(subset=['station_id', 'lat', 'lon', 'duration_in_min'])
    df.reset_index(drop=True, inplace=True)
    return df


def classify_ari_label(ari: float) -> str:
    """Return nearest canonical label for given ari (years)."""
    if ari is None or np.isnan(ari):
        return "unknown"
    bins = np.array([1, 2, 5, 10, 20, 50, 100], dtype=float)
    idx = int(np.argmin(np.abs(np.log(bins) - np.log(max(ari, 1e-9)))))
    return f"1-in-{int(bins[idx])}-year"


def estimate_return_period_from_ifd(
    rain_mm: float,
    duration_h: float,
    lat: float,
    lon: float,
    ifd_df: pd.DataFrame,
    climate_uplift: bool = False,
    uplift_factor: float = 0.10,
    max_distance_km: float = 200.0
) -> Optional[float]:
    """
    Estimate return period (years) from observed rainfall using IFD table.

    Args:
        rain_mm: observed total rainfall (mm)
        duration_h: duration in hours
        lat, lon: location
        ifd_df: pre-loaded IFD DataFrame (use load_ifd_data to prepare)
        climate_uplift: if True, apply multiplicative uplift (e.g., +10%)
        uplift_factor: fraction (0.10 = +10%)
        max_distance_km: if no station within this, function will still attempt closest but will warn

    Returns:
        float return period in years (e.g., 10.0), or np.nan on failure
    """
    try:
        # Validate inputs
        if rain_mm is None or duration_h is None:
            return np.nan
        if duration_h <= 0:
            return np.nan
        # apply uplift
        eff_rain = float(rain_mm) * (1.0 + float(uplift_factor)) if climate_uplift else float(rain_mm)
        intensity = eff_rain / float(duration_h)  # mm per hour
        duration_min = float(duration_h) * 60.0

        # ensure ifd_df has required AEP columns
        for col in CANON_AEP_COLS:
            if col not in ifd_df.columns:
                raise ValueError(f"IFD table missing AEP column: {col}")

        # --- find nearest station (vectorized) ---
        # compute haversine distances vectorized
        lat_arr = ifd_df['lat'].astype(float).to_numpy()
        lon_arr = ifd_df['lon'].astype(float).to_numpy()
        lat_rad = np.radians(lat_arr)
        lon_rad = np.radians(lon_arr)
        lat0 = np.radians(lat)
        lon0 = np.radians(lon)
        dlat = lat_rad - lat0
        dlon = lon_rad - lon0
        a = np.sin(dlat/2.0)**2 + np.cos(lat0) * np.cos(lat_rad) * np.sin(dlon/2.0)**2
        dist_km = 2.0 * 6371.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        nearest_idx = int(np.nanargmin(dist_km))
        min_dist = float(dist_km[nearest_idx])

        # optional: if beyond max_distance_km, still return closest but could warn (app can log)
        if min_dist > max_distance_km:
            # still proceed but caller may want to know
            pass

        station_id = ifd_df['station_id'].iloc[nearest_idx]
        station_rows = ifd_df[ifd_df['station_id'] == station_id].copy()
        station_rows = station_rows.sort_values('duration_in_min').reset_index(drop=True)

        durations = station_rows['duration_in_min'].astype(float).to_numpy()  # minutes
        if np.any(durations <= 0):
            raise ValueError("IFD durations must be positive")

        # build AEP depth matrix for the station (shape: n_durations x 7)
        aep_matrix = station_rows[CANON_AEP_COLS].to_numpy(dtype=float)

        # --- interpolate to requested duration (in log-duration space) ---
        # if exact match (allow small tolerance)
        tol = 1e-6
        match_idxs = np.where(np.isclose(durations, duration_min, atol=tol))[0]
        if match_idxs.size > 0:
            aep_values = aep_matrix[match_idxs[0], :]
        else:
            # We will interpolate for each AEP column across durations using log(duration)
            log_durs = np.log(durations)
            log_target = np.log(duration_min)
            aep_values = np.zeros(len(CANON_AEP_COLS), dtype=float)
            for j in range(len(CANON_AEP_COLS)):
                y = aep_matrix[:, j]
                valid = ~np.isnan(y) & (y > 0)
                if np.sum(valid) == 0:
                    aep_values[j] = np.nan
                elif np.sum(valid) == 1:
                    aep_values[j] = y[valid][0]
                else:
                    # interpolate in log-duration space (linear interp)
                    aep_values[j] = float(np.interp(log_target, log_durs[valid], y[valid]))

        # remove invalid AEP entries
        valid_mask = (~np.isnan(aep_values)) & (aep_values > 0)
        if not np.any(valid_mask):
            return np.nan

        x = aep_values[valid_mask]      # depths (mm) at canonical AEPs
        y = CANON_ARI[valid_mask]       # corresponding ARI values (years)

        # --- estimate ARI by log-log interpolation/extrapolation ---
        # if intensity below min depth -> near 1-year
        if intensity <= np.min(x):
            estimated_ari = float(y[np.argmin(x)])  # usually 1
        elif intensity >= np.max(x):
            # extrapolate on log-log using last two valid points
            log_x = np.log(x)
            log_y = np.log(y)
            # np.interp will extrapolate; use that
            log_est = np.interp(np.log(intensity), log_x, log_y)
            estimated_ari = float(np.exp(log_est))
        else:
            log_est = np.interp(np.log(intensity), np.log(x), np.log(y))
            estimated_ari = float(np.exp(log_est))

        # bounds & sanity
        estimated_ari = float(np.clip(estimated_ari, 0.5, 1000.0))
        return estimated_ari

    except Exception as exc:
        # return nan on failure; caller can log exc
        return np.nan


# Load raster catalog dynamically
raster_catalog = pd.read_csv("SwinburneData/FloodMaps/raster_catalog.csv")

# --- Normalise catalog after reading it ---
def normalise_catalog(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # normalise fields
    df['return_period'] = df['return_period'].astype(str).str.lower().str.strip()
    df['metric'] = df['metric'].astype(str).str.lower().str.strip()
    df['full_path'] = df['full_path'].astype(str).str.replace("\\", "/", regex=False)

    # unify synonyms (your 100y has "dmaxmax" / "hmaxmax")
    metric_map = {
        'dmaxmax': 'dmax',
        'hmaxmax': 'hmax',
        'z0':      'z0max',
        'z0_max':  'z0max'
    }
    df['metric'] = df['metric'].replace(metric_map, regex=False)

    # make sure files exist (useful warning in Streamlit)
    df['exists'] = df['full_path'].apply(lambda p: os.path.exists(p))
    missing = df.loc[~df['exists'], 'full_path'].tolist()
    if len(missing):
        st.warning(f"⚠️ Missing raster files (showing up to 5): {missing[:5]}")

    return df

raster_catalog = normalise_catalog(raster_catalog)
YEARS_ORDER = sorted(raster_catalog['return_period'].unique(), key=lambda x: int(x.replace('y','')))
YEAR_VALS   = np.array([int(x.replace('y','')) for x in YEARS_ORDER], dtype=float)

def collect_metrics_for_point(lat, lon):
    metrics = {y: {"dmax": np.nan, "hmax": np.nan, "vmax": np.nan, "z0max": np.nan} for y in YEARS_ORDER}

    for y in YEARS_ORDER:
        subset = raster_catalog[raster_catalog['return_period'] == y]
        for metric in ['dmax', 'hmax', 'vmax', 'z0max']:
            row = subset[subset['metric'].str.startswith(metric)]
            if not row.empty:
                path = row.iloc[0]['full_path']
                val = sample_raster_value(path,lon, lat, prefer_band=1)
                
                metrics[y][metric] = val
    return metrics


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

def sample_raster_value(raster_path, lon, lat, prefer_band=1, max_depth_m=2.0):
    """
    Sample a flood raster at the given location and scale from 0–255 to real meters.
    """
    import rasterio
    import numpy as np
    from rasterio.warp import transform

    try:
        with rasterio.open(raster_path) as src:
            xs, ys = transform("EPSG:4326", src.crs, [lon], [lat])

            if not (src.bounds.left <= xs[0] <= src.bounds.right and src.bounds.bottom <= ys[0] <= src.bounds.top):
                return 0.0

            val = list(src.sample([(xs[0], ys[0])], indexes=prefer_band))[0][0]

            # Handle data type scaling
            if val is None or np.isnan(val):
                return 0.0

            # Scale if uint8 (0–255)
            if src.dtypes[0] == "uint8":
                if val == 0:
                    return 0.0
                else:
                    # Convert 0–255 → 0–max_depth_m (typically 2 m)
                    return round((val / 255.0) * max_depth_m, 3)

            # If float rasters (already in meters)
            if np.issubdtype(np.array(val).dtype, np.floating):
                return round(float(val), 3)

            return 0.0
    except Exception as e:
        print(f"⚠️ Raster sampling error for {raster_path}: {e}")
        return 0.0


# def sample_raster_value(lat, lon, raster_path, prefer_band=1):
#     """
#     Sample raster safely at given lat/lon.

#     Rules:
#     - If outside raster -> np.nan
#     - If nodata or >=250 for uint8 -> 0.0 (treat as no flood)
#     - If uint8 and <250 -> value/100.0 (cm -> m)
#     - If negative -> 0.0
#     - Clip to sensible range [0, 10] m
#     """
#     try:
#         with rasterio.open(raster_path) as src:
#             # 1) Transform WGS84 -> raster CRS
#             xs, ys = transform("EPSG:4326", src.crs, [lon], [lat])

#             # 2) Inside bounds?
#             if not (src.bounds.left <= xs[0] <= src.bounds.right and
#                     src.bounds.bottom <= ys[0] <= src.bounds.top):
#                 return np.nan

#             # 3) Sample value from chosen band
#             band = prefer_band if 1 <= prefer_band <= src.count else 1
#             val = list(src.sample([(xs[0], ys[0])], indexes=band))[0][0]

#             # 4) Handle nodata/invalids
#             nodata_val = src.nodata
#             if (val is None) or (nodata_val is not None and val == nodata_val) or np.isnan(val):
#                 return 0.0

#             # 5) Scaling logic
#             if src.dtypes[band-1] == "uint8":
#                 # 255 (and often >=250) used as mask/nodata
#                 if val >= 250:
#                     return 0.0
#                 depth_m = val / 100.0  # cm -> m
#             else:
#                 # Other encodings fallback: if very large, assume mm
#                 depth_m = float(val)
#                 if depth_m > 100:  # defensive
#                     depth_m = depth_m / 1000.0

#             # 6) No negatives; clamp to reasonable range
#             if depth_m < 0:
#                 depth_m = 0.0
#             depth_m = float(np.clip(depth_m, 0.0, 10.0))

#             return round(depth_m, 3)

#     except Exception as e:
#         print(f"⚠️ Raster sampling error ({raster_path}): {e}")
#         return np.nan

def collect_metrics_for_point(lat, lon):
    """
    For each return period (001y..100y) gather dmax/hmax/vmax/z0max
    using first matching raster per metric.
    """
    out = {y: {"dmax": np.nan, "hmax": np.nan, "vmax": np.nan, "z0max": np.nan} for y in YEARS_ORDER}

    for y in YEARS_ORDER:
        subset = raster_catalog[(raster_catalog["return_period"] == y) & (raster_catalog["exists"])]
        if subset.empty:
            continue

        for metric in ["dmax", "hmax", "vmax", "z0max"]:
            # "starts with" is handy if your catalog metric contains suffixes
            cand = subset[subset["metric"].str.startswith(metric)]
            if cand.empty:
                continue
            path = cand.iloc[0]["full_path"]
            # out[y][metric] = sample_raster_value(lat, lon, path, prefer_band=1)
            out[y][metric] = sample_raster_value(path, lon, lat, prefer_band=1)


    return out
def find_nearby_wet_pixel(lat, lon, raster_path, radius_px=4, prefer_band=1):
    """
    Search a small window around the given point for first non-zero
    (useful for seeing if you're just outside the flood extent).
    Returns (value_m, lat, lon) or (None, None, None) if not found.
    """
    try:
        with rasterio.open(raster_path) as src:
            xs, ys = transform("EPSG:4326", src.crs, [lon], [lat])
            row, col = src.index(xs[0], ys[0])

            band = prefer_band if 1 <= prefer_band <= src.count else 1
            window = src.read(band, window=rasterio.windows.Window(col-radius_px, row-radius_px,
                                                                   2*radius_px+1, 2*radius_px+1))
            # scan for first non-zero (and <250 for uint8)
            it = np.nditer(window, flags=['multi_index'])
            while not it.finished:
                v = float(it[0])
                if src.dtypes[band-1] == 'uint8':
                    cond = (0 < v < 250)
                else:
                    cond = v > 0

                if cond:
                    # map back to lat/lon of that pixel
                    rr = row - radius_px + it.multi_index[0]
                    cc = col - radius_px + it.multi_index[1]
                    x, y = src.transform * (cc + 0.5, rr + 0.5)
                    llon, llat = transform(src.crs, "EPSG:4326", [x], [y])
                    # scale as in the sampler
                    depth_m = v/100.0 if src.dtypes[band-1] == 'uint8' else v
                    if depth_m > 100: depth_m /= 1000.0
                    return round(float(depth_m),3), float(llon[0]), float(llat[0])
                it.iternext()
    except Exception as e:
        print("nearby search error:", e)
    return None, None, None


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
prop_point = Point(prop["lon"], prop["lat"])
sub_match = subcatchments_gdf[subcatchments_gdf.contains(prop_point)]
if not sub_match.empty:
    sub_name = sub_match.iloc[0].get("Subcatchment", "Unknown")
    st.caption(f"📍 Subcatchment: {sub_name}")
else:
    st.caption("📍 Outside known subcatchments.")



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

# New IFD-based rainfall mapping
if not np.isnan(rain) and not np.isnan(dur):
    scen_rp = estimate_return_period_from_ifd(rain, dur, prop["lat"], prop["lon"], ifd_table)

    # Handle missing or invalid return period safely
    if scen_rp is None or np.isnan(scen_rp):
        scen = "010y"
        st.caption("🌀 Unable to estimate return period — defaulting to 10-year event (010y).")
    else:
        scen_rounded = int(round(scen_rp / 5) * 5)
        scen = f"{scen_rounded:03d}y"
        st.caption(f"🌀 Estimated event intensity corresponds to ~{scen_rp:.1f}-year storm ({scen})")

else:
    scen = "010y"
    st.caption("🌀 Missing rainfall or duration — defaulting to 10-year event (010y).")


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

st.download_button(
    "⬇️ Download Flood Report (CSV)",
    df.to_csv(index=False).encode('utf-8'),
    "flood_metrics.csv",
    "text/csv"
)


