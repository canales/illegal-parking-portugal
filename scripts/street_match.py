"""
street_match.py
─────────────────────────────────────────────────────────────────────────────
Snaps infraction points to the OSM road network and aggregates per street.
Produces data/streets_matched.geojson for the map's "Por rua" view.

Strategy:
  - Uses pyrosm to read a local Portugal OSM PBF (faster & more reliable
    than hitting the Overpass API for a whole country).
  - Downloads portugal-latest.osm.pbf from Geofabrik if not present.
  - Snaps each point to nearest road edge within 50m.
  - Aggregates count, top_infraction, first_year, last_year per edge.
  - Writes a lightweight GeoJSON (geometry + properties, no raw point dump).

Runtime: ~12–20 min locally for all Portugal. ~20 min on GitHub Actions.

Usage:
    pip install pyrosm geopandas scipy shapely
    python scripts/street_match.py

    # Lisbon-only fast run (for testing):
    python scripts/street_match.py --city lisbon
"""

import os
import json
import argparse
import urllib.request
from collections import Counter, defaultdict

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from scipy.spatial import cKDTree
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PBF_PATH  = os.path.join(ROOT, 'data', 'portugal-latest.osm.pbf')
IN_PATH   = os.path.join(ROOT, 'data', 'infractions.geojson')
OUT_PATH  = os.path.join(ROOT, 'data', 'streets_matched.geojson')

PBF_URL   = "https://download.geofabrik.de/europe/portugal-latest.osm.pbf"

# City bounding boxes for fast local testing
CITY_BBOXES = {
    'lisbon': (38.60, -9.50, 38.90, -8.90),   # (lat_min, lon_min, lat_max, lon_max)
    'porto':  (41.00, -8.80, 41.30, -8.40),
    'braga':  (41.45, -8.55, 41.65, -8.35),
}

# Max distance (metres) to snap a point to a road edge
SNAP_THRESHOLD_M = 50


def download_pbf():
    """Download PBF if not already present (~400MB)."""
    if os.path.exists(PBF_PATH):
        size_mb = os.path.getsize(PBF_PATH) / 1e6
        print(f"PBF already present ({size_mb:.0f} MB) — skipping download.")
        return
    os.makedirs(os.path.dirname(PBF_PATH), exist_ok=True)
    print(f"Downloading Portugal OSM PBF from Geofabrik (~400MB)...")
    print(f"  → {PBF_URL}")
    urllib.request.urlretrieve(PBF_URL, PBF_PATH, reporthook=_progress)
    print()


def _progress(count, block_size, total_size):
    pct = min(int(count * block_size * 100 / total_size), 100)
    print(f"\r  {pct}%", end='', flush=True)


def load_osm_network(bbox=None):
    """
    Load OSM drive network using pyrosm.
    bbox: (lat_min, lon_min, lat_max, lon_max) or None for all Portugal.
    """
    try:
        from pyrosm import OSM
    except ImportError:
        raise ImportError("Run: pip install pyrosm")

    print("Parsing OSM network...")
    osm = OSM(PBF_PATH, bounding_box=list(bbox) if bbox else None)
    # driving=True filters out footways, service roads etc.
    edges = osm.get_network(network_type="driving")
    print(f"  Loaded {len(edges)} road edges.")
    return edges


def load_infractions(bbox=None):
    """Load infraction GeoJSON, optionally filtering to bbox."""
    with open(IN_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rows = []
    for feat in data['features']:
        lon, lat = feat['geometry']['coordinates']
        p = feat['properties']
        if bbox:
            lat_min, lon_min, lat_max, lon_max = bbox
            if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
                continue
        year = p.get('data_data', '')[:4]
        rows.append({
            'lat': lat, 'lon': lon,
            'infraction_type': p.get('infraction_type', ''),
            'year': year
        })

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} infraction points{' (filtered to bbox)' if bbox else ''}.")
    return df


def build_edge_midpoints(edges: gpd.GeoDataFrame):
    """
    Build a KD-tree from edge midpoints for fast nearest-edge lookup.
    Returns (tree, edge_index_array).
    """
    print("Building spatial index on road edges...")
    # Project to a metric CRS for distance calculations
    edges_proj = edges.to_crs("EPSG:3763")  # Portugal TM06

    mids = edges_proj.geometry.interpolate(0.5, normalized=True)
    coords = np.array([(g.x, g.y) for g in mids])
    tree = cKDTree(coords)
    return tree, edges_proj, coords


def snap_points(df: pd.DataFrame, tree: cKDTree, edges_proj: gpd.GeoDataFrame):
    """
    Snap each infraction point to nearest road edge.
    Returns df with 'edge_idx' column (-1 if no edge within threshold).
    """
    print("Snapping points to nearest road edge...")

    # Project points to same CRS
    pts_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
    pts_proj = pts_gdf.to_crs("EPSG:3763")
    pt_coords = np.array([(g.x, g.y) for g in pts_proj.geometry])

    dists, idxs = tree.query(pt_coords, k=1)

    # Reject points farther than threshold
    valid = dists <= SNAP_THRESHOLD_M
    df = df.copy()
    df['edge_idx'] = np.where(valid, idxs, -1)
    df['snap_dist_m'] = dists

    snapped = valid.sum()
    print(f"  {snapped}/{len(df)} points snapped ({100*snapped/len(df):.1f}%). "
          f"{len(df)-snapped} beyond {SNAP_THRESHOLD_M}m threshold.")
    return df


def aggregate_per_edge(df: pd.DataFrame, edges_proj: gpd.GeoDataFrame):
    """Aggregate infraction stats per road edge."""
    print("Aggregating per street segment...")
    matched = df[df['edge_idx'] >= 0].copy()

    agg = {}
    for edge_idx, group in matched.groupby('edge_idx'):
        types = group['infraction_type'].tolist()
        years = [y for y in group['year'].tolist() if y.isdigit()]
        agg[edge_idx] = {
            'count':         len(group),
            'top_infraction': Counter(types).most_common(1)[0][0],
            'first_year':    min(years) if years else None,
            'last_year':     max(years) if years else None,
            'type_breakdown': dict(Counter(types))
        }

    return agg


def build_output_geojson(edges: gpd.GeoDataFrame, edges_proj: gpd.GeoDataFrame, agg: dict):
    """Build output GeoJSON — only edges that have ≥1 infraction."""
    print("Building output GeoJSON...")
    features = []

    for edge_idx, stats in agg.items():
        geom = edges.geometry.iloc[edge_idx]
        if geom is None or geom.is_empty:
            continue

        # Street name: try 'name' column from OSM
        name = edges.iloc[edge_idx].get('name', None) if hasattr(edges.iloc[edge_idx], 'get') else None
        if isinstance(name, float):  # NaN
            name = None

        features.append({
            "type": "Feature",
            "geometry": json.loads(gpd.GeoSeries([geom], crs="EPSG:4326").to_json())['features'][0]['geometry'],
            "properties": {
                "street_name":    name,
                "count":          stats['count'],
                "top_infraction": stats['top_infraction'],
                "first_year":     stats['first_year'],
                "last_year":      stats['last_year'],
                "type_breakdown": json.dumps(stats['type_breakdown'])
            }
        })

    print(f"  {len(features)} road segments with infractions.")
    return {"type": "FeatureCollection", "features": features}


def main():
    parser = argparse.ArgumentParser(description='Street matching for parking infractions')
    parser.add_argument('--city', choices=list(CITY_BBOXES.keys()),
                        help='Run for a single city bbox (faster, for testing)')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip PBF download check')
    args = parser.parse_args()

    bbox = CITY_BBOXES.get(args.city) if args.city else None
    if args.city:
        print(f"Running in CITY mode: {args.city} (bbox: {bbox})")
    else:
        print("Running in FULL PORTUGAL mode")

    if not args.skip_download:
        download_pbf()

    edges = load_osm_network(bbox=bbox)

    # Keep only the geometry and name columns we need
    keep_cols = ['geometry']
    for col in ['name', 'highway', 'maxspeed']:
        if col in edges.columns:
            keep_cols.append(col)
    edges = edges[keep_cols].copy()

    tree, edges_proj, _ = build_edge_midpoints(edges)
    df = load_infractions(bbox=bbox)
    df = snap_points(df, tree, edges_proj)
    agg = aggregate_per_edge(df, edges_proj)

    output = build_output_geojson(edges, edges_proj, agg)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False)

    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f"\n✓ Saved → {OUT_PATH} ({size_kb:.0f} KB)")
    print("Done! Load the map to see the 'Por rua' layer.")


if __name__ == "__main__":
    main()
