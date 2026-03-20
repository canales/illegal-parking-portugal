"""
street_match.py
─────────────────────────────────────────────────────────────────────────────
Snaps infraction points to the OSM road network and aggregates per street.
Produces data/streets_matched.geojson for the map's "By street" view.

Strategy:
  - Uses osmnx to fetch the road network from OpenStreetMap.
  - Runs city by city (Lisbon, Porto, Braga, etc.) to avoid Overpass
    API timeouts that would happen fetching all Portugal at once.
  - Snaps each point to nearest road edge within 50m.
  - Aggregates count, top_infraction, first_year, last_year per edge.
  - Merges all cities into a single output GeoJSON.

Runtime: ~10–20 min on GitHub Actions (network fetch dominates).

Usage:
    pip install osmnx geopandas scipy shapely numpy pandas
    python scripts/street_match.py

    # Single city for quick local testing:
    python scripts/street_match.py --city lisbon
"""

import os
import json
import argparse
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

# ── Cities to process ─────────────────────────────────────────────────────
# Each entry: name, (lat_min, lon_min, lat_max, lon_max)
# Covers ~95% of the data. Add more cities here as the dataset grows.
CITIES = {
    'lisbon':    (38.60, -9.50, 38.90, -8.90),
    'porto':     (41.00, -8.80, 41.30, -8.50),
    'braga':     (41.45, -8.55, 41.65, -8.35),
    'coimbra':   (40.15, -8.55, 40.25, -8.35),
    'aveiro':    (40.58, -8.70, 40.68, -8.55),
    'setubal':   (38.50, -8.95, 38.60, -8.85),
    'faro':      (37.00, -8.05, 37.10, -7.90),
    'cascais':   (38.68, -9.50, 38.76, -9.38),
    'sintra':    (38.76, -9.45, 38.86, -9.30),
    'amadora':   (38.73, -9.25, 38.78, -9.18),
    'matosinhos':(41.17, -8.74, 41.22, -8.66),
    'gaia':      (41.08, -8.65, 41.17, -8.55),
}

# Max snapping distance in metres
SNAP_THRESHOLD_M = 50

# osmnx config — be polite to the Overpass API
ox.settings.log_console = False
ox.settings.use_cache   = True   # caches network requests locally
ox.settings.timeout     = 180


def load_infractions():
    """Load all infraction points from GeoJSON."""
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
    print(f"Loaded {len(df)} infraction points total.")
    return df


def filter_to_bbox(df, bbox):
    lat_min, lon_min, lat_max, lon_max = bbox
    return df[
        (df.lat >= lat_min) & (df.lat <= lat_max) &
        (df.lon >= lon_min) & (df.lon <= lon_max)
    ].copy()


def fetch_network(city_name, bbox):
    """Fetch simplified OSM drive network for a bounding box."""
    lat_min, lon_min, lat_max, lon_max = bbox
    print(f"  Fetching OSM network for {city_name}...")
    try:
        G = ox.graph_from_bbox(
            bbox=(lat_max, lat_min, lon_max, lon_min),  # osmnx order: N, S, E, W
            network_type='drive',
            simplify=True
        )
        edges = ox.graph_to_gdfs(G, nodes=False)
        print(f"  → {len(edges)} road edges loaded.")
        return edges
    except Exception as e:
        print(f"  WARNING: Could not fetch network for {city_name}: {e}")
        return None


def build_kdtree(edges):
    """Build KD-tree from edge midpoints projected to metric CRS."""
    edges_proj = edges.to_crs("EPSG:3763")  # Portugal TM06
    mids = edges_proj.geometry.interpolate(0.5, normalized=True)
    coords = np.array([(g.x, g.y) for g in mids])
    tree = cKDTree(coords)
    return tree, edges_proj


def snap_and_aggregate(df_city, edges, city_name):
    """Snap points to nearest edge and aggregate stats per edge."""
    if df_city.empty:
        print(f"  No infraction points in {city_name}, skipping.")
        return {}

    tree, edges_proj = build_kdtree(edges)

    # Project points
    pts = gpd.GeoDataFrame(df_city, geometry=gpd.points_from_xy(df_city.lon, df_city.lat), crs="EPSG:4326")
    pts_proj = pts.to_crs("EPSG:3763")
    pt_coords = np.array([(g.x, g.y) for g in pts_proj.geometry])

    dists, idxs = tree.query(pt_coords, k=1)
    valid = dists <= SNAP_THRESHOLD_M

    snapped = valid.sum()
    print(f"  {snapped}/{len(df_city)} points snapped within {SNAP_THRESHOLD_M}m.")

    df_city = df_city.copy()
    df_city['edge_idx'] = np.where(valid, idxs, -1)
    matched = df_city[df_city['edge_idx'] >= 0]

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
    """Convert aggregated edge stats to GeoJSON features."""
    features = []
    edges_wgs = edges.to_crs("EPSG:4326") if edges.crs.to_epsg() != 4326 else edges

    for edge_idx, stats in agg.items():
        try:
            row  = edges_wgs.iloc[edge_idx]
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            name = row.get('name') if hasattr(row, 'get') else None
            if isinstance(name, float):
                name = None
            # osmnx sometimes returns a list of names for merged edges
            if isinstance(name, list):
                name = name[0]

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', choices=list(CITIES.keys()),
                        help='Process a single city only (faster for testing)')
    args = parser.parse_args()

    df_all = load_infractions()
    cities_to_run = {args.city: CITIES[args.city]} if args.city else CITIES

    all_features = []
    total_edges_processed = 0

    for city_name, bbox in cities_to_run.items():
        print(f"\n── {city_name.upper()} ──────────────────────────────")
        df_city = filter_to_bbox(df_all, bbox)
        print(f"  {len(df_city)} infraction points in bbox.")

        if df_city.empty:
            print("  Skipping — no data.")
            continue

        edges = fetch_network(city_name, bbox)
        if edges is None or edges.empty:
            continue

        agg = snap_and_aggregate(df_city, edges, city_name)
        features = build_features(edges, agg)
        all_features.extend(features)
        total_edges_processed += len(agg)
        print(f"  → {len(features)} street segments with infractions.")

    print(f"\n── TOTAL: {len(all_features)} segments across all cities ──")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump({"type": "FeatureCollection", "features": all_features}, f, ensure_ascii=False)

    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f"✓ Saved → {OUT_PATH} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
