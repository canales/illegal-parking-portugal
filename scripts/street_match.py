"""
street_match.py
─────────────────────────────────────────────────────────────────────────────
Snaps infraction points to the OSM road network and aggregates per street.
Produces data/streets_matched.geojson for the map's "By street" view.

Strategy:
  - Uses osmnx to fetch road networks from OSM via the Overpass API.
  - Runs one small district at a time to stay within Overpass limits.
  - Each district has a hard timeout — if it hangs, it's skipped cleanly.
  - Includes retry with backoff for transient Overpass errors (429/503).
  - Delays between districts to avoid rate-limiting.
  - Results from all districts are merged into one output GeoJSON.

Runtime: ~10–20 min on GitHub Actions.

Usage:
    pip install osmnx==2.0.1 geopandas scipy shapely numpy pandas
    python scripts/street_match.py

    # Single district for quick local testing:
    python scripts/street_match.py --district lisbon_center
"""

import os
import json
import time
import argparse
import signal
from collections import Counter

import osmnx as ox
import geopandas as gpd
import pandas as pd
from scipy.spatial import cKDTree
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_PATH  = os.path.join(ROOT, 'data', 'infractions.geojson')
OUT_PATH = os.path.join(ROOT, 'data', 'streets_matched.geojson')

# ── Districts ──────────────────────────────────────────────────────────────
# Deliberately small bboxes — Overpass chokes on large areas.
# Each bbox: (lat_min, lon_min, lat_max, lon_max)  ← storage format
# Converted to osmnx 2.x order (west, south, east, north) at call time.
# Rule of thumb: keep each box under ~0.15° × 0.15° (~15 km × 15 km)
DISTRICTS = {
    # Lisbon split into quadrants + suburbs
    'lisbon_center':  (38.70, -9.20, 38.80, -9.10),
    'lisbon_west':    (38.70, -9.30, 38.80, -9.20),
    'lisbon_north':   (38.80, -9.20, 38.90, -9.10),
    'lisbon_cascais': (38.68, -9.45, 38.76, -9.30),
    'lisbon_sintra':  (38.76, -9.45, 38.86, -9.30),
    'lisbon_amadora': (38.73, -9.28, 38.80, -9.18),
    'lisbon_setubal': (38.50, -8.95, 38.62, -8.83),
    # Porto metro
    'porto_center':   (41.13, -8.65, 41.22, -8.55),
    'porto_south':    (41.08, -8.65, 41.15, -8.55),
    'matosinhos':     (41.17, -8.74, 41.24, -8.64),
    # Other cities
    'braga':          (41.53, -8.48, 41.62, -8.38),
    'coimbra':        (40.18, -8.48, 40.24, -8.38),
    'aveiro':         (40.60, -8.67, 40.66, -8.58),
    'faro':           (37.00, -8.03, 37.08, -7.92),
}

SNAP_THRESHOLD_M   = 50
FETCH_TIMEOUT_SECS = 300   # hard wall-clock timeout per district
DELAY_BETWEEN_SECS = 12    # polite pause between Overpass requests
MAX_RETRIES        = 3     # retry transient failures with backoff

# ── osmnx 2.x settings ────────────────────────────────────────────────────
ox.settings.log_console       = False
ox.settings.use_cache         = False      # ephemeral CI runner — no cache
ox.settings.requests_timeout  = 120        # client HTTP timeout + server [timeout:N] (osmnx 2.x name)
ox.settings.overpass_memory   = 536870912  # 512 MB — server [maxsize:N], fills {maxsize} in template
# max_query_area_size intentionally left at default (2,500,000,000 m²).
# Our bboxes are ~96 km² each — well under the threshold.
# The old override (50_000_000) was SMALLER than our bboxes, forcing osmnx
# to subdivide every district into hundreds of thousands of sub-queries.

# Identify ourselves so Overpass admins can reach us if needed.
ox.settings.http_user_agent = 'street_match.py/1.0 (GitHub Actions; parking infraction project)'


# ── Timeout helper (Unix only — works on GitHub Actions Linux) ─────────────
class _Timeout(Exception):
    pass

def _timeout_handler(signum, frame):
    raise _Timeout()


# ── Data loading ───────────────────────────────────────────────────────────
def load_infractions():
    with open(IN_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    rows = []
    for feat in data['features']:
        lon, lat = feat['geometry']['coordinates']
        p = feat['properties']
        rows.append({
            'lat': lat, 'lon': lon,
            'infraction_type': p.get('infraction_type', ''),
            'year': p.get('data_data', '')[:4]
        })
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} infraction points.")
    return df


def filter_to_bbox(df, bbox):
    lat_min, lon_min, lat_max, lon_max = bbox
    return df[
        (df.lat >= lat_min) & (df.lat <= lat_max) &
        (df.lon >= lon_min) & (df.lon <= lon_max)
    ].copy()


# ── OSM network fetch (with retry + hard timeout) ─────────────────────────
def fetch_network(name, bbox):
    """Fetch the driveable road network for a bbox, with retry and timeout."""
    lat_min, lon_min, lat_max, lon_max = bbox

    # osmnx 2.x bbox order: (west, south, east, north) = (lon_min, lat_min, lon_max, lat_max)
    ox_bbox = (lon_min, lat_min, lon_max, lat_max)

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"  Fetching OSM network (attempt {attempt}/{MAX_RETRIES})...", end=' ', flush=True)

        # Set hard wall-clock alarm (Linux/Mac only)
        try:
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(FETCH_TIMEOUT_SECS)
        except AttributeError:
            pass  # Windows — no SIGALRM

        try:
            G = ox.graph_from_bbox(
                bbox=ox_bbox,
                network_type='drive',
                simplify=True
            )
            edges = ox.graph_to_gdfs(G, nodes=False)
            print(f"{len(edges)} edges.")
            return edges

        except _Timeout:
            print(f"TIMED OUT after {FETCH_TIMEOUT_SECS}s.")

        except Exception as e:
            err_str = str(e).lower()
            print(f"FAILED ({type(e).__name__}: {e})")

            # Retry on transient Overpass / HTTP errors
            is_transient = any(tok in err_str for tok in [
                '429', '503', '504', 'timeout', 'timed out',
                'too many requests', 'server load', 'overpass',
                'connection', 'read timed out',
            ])
            if not is_transient:
                print("  Non-transient error — skipping district.")
                return None

        finally:
            try:
                signal.alarm(0)
            except AttributeError:
                pass

        # Exponential backoff before retry
        if attempt < MAX_RETRIES:
            wait = DELAY_BETWEEN_SECS * (2 ** (attempt - 1))
            print(f"  Waiting {wait}s before retry...")
            time.sleep(wait)

    print(f"  All {MAX_RETRIES} attempts failed — skipping district.")
    return None


# ── Snap & aggregate ───────────────────────────────────────────────────────
def snap_and_aggregate(df_pts, edges):
    if df_pts.empty or edges is None or edges.empty:
        return {}

    edges_proj = edges.to_crs("EPSG:3763")
    mids   = edges_proj.geometry.interpolate(0.5, normalized=True)
    coords = np.array([(g.x, g.y) for g in mids])
    tree   = cKDTree(coords)

    pts      = gpd.GeoDataFrame(
        df_pts,
        geometry=gpd.points_from_xy(df_pts.lon, df_pts.lat),
        crs="EPSG:4326"
    )
    pts_proj = pts.to_crs("EPSG:3763")
    pt_xy    = np.array([(g.x, g.y) for g in pts_proj.geometry])

    dists, idxs = tree.query(pt_xy, k=1)
    valid = dists <= SNAP_THRESHOLD_M
    print(f"  {valid.sum()}/{len(df_pts)} points snapped.")

    df_pts = df_pts.copy()
    df_pts['edge_idx'] = np.where(valid, idxs, -1)
    matched = df_pts[df_pts['edge_idx'] >= 0]

    agg = {}
    for edge_idx, group in matched.groupby('edge_idx'):
        types = group['infraction_type'].tolist()
        years = [y for y in group['year'].tolist() if len(y) == 4 and y.isdigit()]
        agg[int(edge_idx)] = {
            'count':          len(group),
            'top_infraction': Counter(types).most_common(1)[0][0],
            'first_year':     min(years) if years else None,
            'last_year':      max(years) if years else None,
            'type_breakdown': dict(Counter(types))
        }
    return agg


def build_features(edges, agg):
    features = []
    edges_wgs = edges.to_crs("EPSG:4326") if edges.crs.to_epsg() != 4326 else edges
    for edge_idx, stats in agg.items():
        try:
            row  = edges_wgs.iloc[edge_idx]
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            name = row.get('name') if hasattr(row, 'get') else None
            if isinstance(name, float): name = None
            if isinstance(name, list):  name = name[0]
            features.append({
                "type": "Feature",
                "geometry": geom.__geo_interface__,
                "properties": {
                    "street_name":    name,
                    "count":          stats['count'],
                    "top_infraction": stats['top_infraction'],
                    "first_year":     stats['first_year'],
                    "last_year":      stats['last_year'],
                    "type_breakdown": json.dumps(stats['type_breakdown'])
                }
            })
        except Exception:
            continue
    return features


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--district', choices=list(DISTRICTS.keys()),
                        help='Run a single district only (for testing)')
    args = parser.parse_args()

    df_all = load_infractions()
    districts = {args.district: DISTRICTS[args.district]} if args.district else DISTRICTS

    all_features = []
    succeeded, skipped = 0, 0
    district_list = list(districts.items())

    for i, (name, bbox) in enumerate(district_list):
        print(f"\n── {name.upper()} ({i+1}/{len(district_list)}) ──")
        df_district = filter_to_bbox(df_all, bbox)
        print(f"  {len(df_district)} points in bbox.")

        if df_district.empty:
            print("  No data — skipping.")
            skipped += 1
            continue

        edges = fetch_network(name, bbox)
        if edges is None:
            skipped += 1
            continue

        agg      = snap_and_aggregate(df_district, edges)
        features = build_features(edges, agg)
        all_features.extend(features)
        print(f"  → {len(features)} segments with infractions.")
        succeeded += 1

        # Polite delay between districts to avoid Overpass rate-limiting
        if i < len(district_list) - 1:
            print(f"  Pausing {DELAY_BETWEEN_SECS}s before next district...")
            time.sleep(DELAY_BETWEEN_SECS)

    print(f"\n── DONE: {succeeded} districts succeeded, {skipped} skipped ──")
    print(f"   {len(all_features)} total street segments.")

    if succeeded == 0:
        print("⚠  No districts succeeded — writing empty collection.")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump({"type": "FeatureCollection", "features": all_features}, f, ensure_ascii=False)

    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f"✓ Saved → {OUT_PATH} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
