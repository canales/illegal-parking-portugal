"""
street_match.py
─────────────────────────────────────────────────────────────────────────────
Snaps infraction points to the OSM road network and aggregates per street.
Produces data/streets_matched.geojson for the map's "By street" view.

Architecture
────────────
1. GRID  — Portugal is divided into 0.1° × 0.1° cells (~10×10km each).
           Only cells that contain at least MIN_POINTS infraction reports
           are processed. This covers all of Portugal automatically without
           a manual city list, and skips rural cells with no data.

2. CACHE — Each cell's OSM road network is saved as a GeoPackage file in
           data/osm_cache/cell_{lat}_{lon}.gpkg after the first fetch.
           On subsequent runs the cache is loaded from disk — Overpass is
           never called again for that cell unless --refresh-cache is used.
           GitHub Actions preserves the cache folder between runs via the
           actions/cache step in update_data.yml.

First run:  ~315 cells × ~12s = ~60 min (one-time cost).
Later runs: ~2–4 min (snap only, no network calls).

Usage
─────
    pip install osmnx==2.0.1 geopandas shapely pandas pyogrio
    python scripts/street_match.py

    # Force re-fetch all cells from Overpass (e.g. after OSM updates)
    python scripts/street_match.py --refresh-cache

    # Single cell for quick local testing (provide lat/lon of cell origin)
    python scripts/street_match.py --cell 38.7 -9.1
"""

import os
import json
import time
import math
import signal
import argparse
from collections import Counter
from typing import Optional
from pathlib import Path

import osmnx as ox
import geopandas as gpd
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
IN_PATH    = ROOT / 'data' / 'infractions.geojson'
OUT_PATH   = ROOT / 'data' / 'streets_matched.geojson'
CACHE_DIR  = ROOT / 'data' / 'osm_cache'

# ── Grid config ────────────────────────────────────────────────────────────
CELL_SIZE  = 0.1    # degrees — ~10km × 10km at Portugal's latitude
MIN_POINTS = 5      # skip cells with fewer than this many infraction points

# Portugal bounding box (covers mainland + Azores + Madeira)
PT_LAT_MIN, PT_LAT_MAX = 29.0, 43.0
PT_LON_MIN, PT_LON_MAX = -32.0, -6.0

# ── Snap config ────────────────────────────────────────────────────────────
SNAP_THRESHOLD_M   = 50
FETCH_TIMEOUT_SECS = 60    # fail fast — 60s is plenty for a 0.1° cell
DELAY_BETWEEN_SECS = 15    # polite pause between requests
MAX_RETRIES        = 2     # one retry is enough; 3 retries × 60s = 3min wasted per cell

# ── osmnx 2.x settings ────────────────────────────────────────────────────
ox.settings.log_console       = False
ox.settings.use_cache         = False
ox.settings.requests_timeout  = 120
ox.settings.overpass_memory   = 536870912   # 512 MB server-side maxsize
# max_query_area_size left at default (2,500,000,000 m²) — our 0.1° cells
# are ~100 km², well below the threshold, so no subdivision occurs.
ox.settings.http_user_agent   = 'street_match.py/2.0 (illegal-parking-portugal; GitHub Actions)'


# ── Timeout helper ─────────────────────────────────────────────────────────
class _Timeout(Exception):
    pass

def _arm_timeout(secs):
    try:
        signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(_Timeout()))
        signal.alarm(secs)
    except AttributeError:
        pass  # Windows

def _disarm_timeout():
    try:
        signal.alarm(0)
    except AttributeError:
        pass


# ── Grid helpers ───────────────────────────────────────────────────────────
def cell_origin(lat: float, lon: float) -> tuple:
    """
    Return the bottom-left corner of the 0.1° cell containing (lat, lon).

    IMPORTANT: must use math.floor, not int().
    int() truncates toward zero, so int(-91.5) = -91.
    math.floor(-91.5) = -92, which is correct for negative longitudes.
    Without this, a point at lon=-9.19 gets assigned to cell -9.1
    (lon -9.1 to -9.0, east of Lisbon) instead of cell -9.2 (correct).
    The points and OSM network end up in completely different places.
    """
    return (
        round(math.floor(lat / CELL_SIZE) * CELL_SIZE, 2),
        round(math.floor(lon / CELL_SIZE) * CELL_SIZE, 2)
    )

def cell_bbox(clat: float, clon: float) -> tuple:
    """Return (lat_min, lon_min, lat_max, lon_max) for a cell origin."""
    return (clat, clon, round(clat + CELL_SIZE, 2), round(clon + CELL_SIZE, 2))

def cell_cache_path(clat: float, clon: float) -> Path:
    return CACHE_DIR / f"cell_{clat:.1f}_{clon:.1f}.gpkg"


# ── Data loading ───────────────────────────────────────────────────────────
def load_infractions() -> pd.DataFrame:
    with open(IN_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    rows = []
    for feat in data['features']:
        lon, lat = feat['geometry']['coordinates']
        if not (PT_LAT_MIN <= lat <= PT_LAT_MAX and PT_LON_MIN <= lon <= PT_LON_MAX):
            continue
        p = feat['properties']
        rows.append({
            'lat': lat, 'lon': lon,
            'infraction_type': p.get('infraction_type', ''),
            'year': p.get('data_data', '')[:4]
        })
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} infraction points (within Portugal bbox).")
    return df


def build_grid(df: pd.DataFrame) -> dict:
    """
    Assign each point to a 0.1° grid cell.
    Returns {(clat, clon): sub-DataFrame} for cells with >= MIN_POINTS.
    """
    df = df.copy()
    df['clat'] = df['lat'].apply(lambda v: round(math.floor(v / CELL_SIZE) * CELL_SIZE, 2))
    df['clon'] = df['lon'].apply(lambda v: round(math.floor(v / CELL_SIZE) * CELL_SIZE, 2))

    cells = {}
    for (clat, clon), group in df.groupby(['clat', 'clon']):
        if len(group) >= MIN_POINTS:
            cells[(clat, clon)] = group
    print(f"Grid: {len(cells)} cells with ≥{MIN_POINTS} points "
          f"(skipped {df.groupby(['clat','clon']).ngroups - len(cells)} sparse cells).")
    return cells


# ── OSM network — fetch or load from cache ─────────────────────────────────
def get_network(clat: float, clon: float, refresh: bool) -> Optional[gpd.GeoDataFrame]:
    cache_path = cell_cache_path(clat, clon)

    # Cache hit
    if cache_path.exists() and not refresh:
        try:
            edges = gpd.read_file(cache_path, layer='edges')
            # Restore CRS — GeoPackage preserves it but let's be explicit
            if edges.crs is None:
                edges = edges.set_crs("EPSG:4326")
            return edges
        except Exception as e:
            print(f"    Cache read failed ({e}), re-fetching...")

    # Cache miss — fetch from Overpass
    lat_min, lon_min, lat_max, lon_max = cell_bbox(clat, clon)
    ox_bbox = (lon_min, lat_min, lon_max, lat_max)  # osmnx 2.x: (W, S, E, N)

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"    Overpass fetch (attempt {attempt}/{MAX_RETRIES})...", end=' ', flush=True)
        _arm_timeout(FETCH_TIMEOUT_SECS)
        try:
            G     = ox.graph_from_bbox(bbox=ox_bbox, network_type='drive', simplify=True)
            edges = ox.graph_to_gdfs(G, nodes=False)

            # Deduplicate bidirectional edges.
            # OSM/osmnx represents two-way roads as two directed edges — one per
            # direction of travel — with identical geometry but reversed node order.
            # This causes the same physical street segment to appear twice in the
            # GeoDataFrame, doubling infraction counts when reports are snapped to it.
            # We deduplicate by treating each edge as an unordered pair of its
            # start and end coordinates (a frozenset), keeping only one direction.
            edges = edges.reset_index(drop=True)
            edges['_edge_key'] = edges.geometry.apply(
                lambda g: frozenset([g.coords[0], g.coords[-1]])
            )
            edges = edges.drop_duplicates(subset='_edge_key').drop(columns='_edge_key')
            edges = edges.reset_index(drop=True)
            print(f"{len(edges)} edges (after dedup).")

            # Save to cache
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            edges_save = edges.to_crs("EPSG:4326").copy()
            # Keep only the columns we need — reduces file size significantly
            keep = ['geometry'] + [c for c in ['name','highway'] if c in edges_save.columns]
            edges_save[keep].to_file(str(cache_path), driver='GPKG', layer='edges')
            print(f"    Cached → {cache_path.name}")
            return edges

        except _Timeout:
            print(f"TIMED OUT after {FETCH_TIMEOUT_SECS}s.")
        except Exception as e:
            err = str(e).lower()
            print(f"FAILED ({type(e).__name__}: {e})")
            transient = any(t in err for t in ['429','503','504','timeout','too many','connection'])
            if not transient:
                _disarm_timeout()
                return None
        finally:
            _disarm_timeout()

        if attempt < MAX_RETRIES:
            wait = DELAY_BETWEEN_SECS * (2 ** (attempt - 1))
            print(f"    Retrying in {wait}s...")
            time.sleep(wait)

    print(f"    All {MAX_RETRIES} attempts failed — skipping cell.")
    return None


# ── Snap & aggregate ───────────────────────────────────────────────────────
def utm_epsg(clon: float) -> str:
    """
    Return the EPSG code for the UTM zone covering a given longitude.
    Works for all of Portugal including Azores (zone 26) and Madeira (zone 28).
    EPSG:3763 only covers mainland — this is the correct replacement.
    """
    zone = int((clon + 180) / 6) + 1
    return f"EPSG:326{zone:02d}"   # Northern Hemisphere (all of Portugal)


def snap_and_aggregate(df_cell: pd.DataFrame, edges: gpd.GeoDataFrame, clon: float):
    """
    Snap infraction points to nearest road edge using sjoin_nearest.

    WHY sjoin_nearest instead of a KDTree on midpoints:
    The previous approach built a KDTree from edge midpoints and measured
    distance from each point to the nearest midpoint. A point 10m from the
    *end* of a 500m street segment is ~260m from its midpoint — well above
    the 50m threshold, so it was rejected. sjoin_nearest measures distance
    to the actual line geometry, so any point within 50m of any part of any
    edge gets matched correctly.

    WHY per-cell UTM instead of EPSG:3763:
    EPSG:3763 (Portugal TM06) is only valid for mainland Portugal. For Azores
    cells (~-25° lon) and Madeira cells (~-17° lon) it produces wildly wrong
    metric coordinates, making all distances appear huge. We compute the
    correct UTM zone from the cell's longitude instead.
    """
    epsg = utm_epsg(clon)

    # Reset index so edge positions are 0, 1, 2… (iloc-safe for build_features)
    edges_proj = edges.reset_index(drop=True).to_crs(epsg)

    pts = gpd.GeoDataFrame(
        df_cell.copy(),
        geometry=gpd.points_from_xy(df_cell.lon, df_cell.lat),
        crs="EPSG:4326"
    ).to_crs(epsg)

    # sjoin_nearest: for each point find the nearest edge within SNAP_THRESHOLD_M.
    # When a point is equidistant from two edges (e.g. at an intersection),
    # sjoin_nearest returns one row per match. We deduplicate on the original
    # point index (not on attribute values) so each report counts exactly once
    # while legitimate duplicate-coordinate reports are preserved.
    pts = pts.reset_index(drop=True)   # ensure clean 0-based index
    joined = gpd.sjoin_nearest(
        pts,
        edges_proj[['geometry']],
        how='left',
        max_distance=SNAP_THRESHOLD_M,
        distance_col='snap_dist_m'
    )

    # Keep only the single closest edge per original point row
    joined = joined.sort_values('snap_dist_m').loc[
        ~joined.index.duplicated(keep='first')
    ]

    snapped = joined[joined['index_right'].notna()].copy()
    snapped['edge_idx'] = snapped['index_right'].astype(int)

    agg = {}
    for edge_idx, group in snapped.groupby('edge_idx'):
        types = group['infraction_type'].tolist()
        years = [y for y in group['year'].tolist() if len(y) == 4 and y.isdigit()]
        agg[int(edge_idx)] = {
            'count':          len(group),
            'top_infraction': Counter(types).most_common(1)[0][0],
            'first_year':     min(years) if years else None,
            'last_year':      max(years) if years else None,
            'type_breakdown': dict(Counter(types))
        }

    return agg, len(snapped), edges_proj  # return edges_proj so build_features uses same index


def load_municipios() -> 'gpd.GeoDataFrame | None':
    """Load municipality boundaries for spatial join. Returns None if missing."""
    munis_path = ROOT / 'data' / 'municipios.geojson'
    if not munis_path.exists():
        print("⚠  municipios.geojson not found — street municipio tags will be skipped.")
        return None
    try:
        munis = gpd.read_file(str(munis_path))[['geometry', 'NAME_2']].copy()
        munis = munis.rename(columns={'NAME_2': 'municipio'})
        return munis.to_crs("EPSG:4326")
    except Exception as e:
        print(f"⚠  Could not load municipios.geojson ({e}) — skipping municipio tags.")
        return None


def build_features(edges_proj: gpd.GeoDataFrame, agg: dict, munis: 'gpd.GeoDataFrame | None') -> list:
    # edges_proj already has a reset integer index from snap_and_aggregate.
    # Convert back to WGS84 for the output GeoJSON.
    edges_wgs = edges_proj.to_crs("EPSG:4326")

    # Spatial join: tag each edge with its municipality using midpoint
    muni_by_edge = {}
    if munis is not None:
        try:
            mids = edges_wgs.copy()
            mids['geometry'] = edges_wgs.geometry.interpolate(0.5, normalized=True)
            joined = gpd.sjoin(mids[['geometry']], munis, how='left', predicate='within')
            muni_by_edge = joined['municipio'].to_dict()
        except Exception as e:
            print(f"    ⚠  municipio join failed ({e})")

    features  = []
    for edge_idx, stats in agg.items():
        try:
            row  = edges_wgs.iloc[edge_idx]
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            name = row.get('name') if hasattr(row, 'get') else None
            if isinstance(name, float): name = None
            if isinstance(name, list):  name = name[0]
            muni = muni_by_edge.get(edge_idx)
            if not isinstance(muni, str): muni = None
            features.append({
                "type": "Feature",
                "geometry": geom.__geo_interface__,
                "properties": {
                    "street_name":    name,
                    "municipio":      muni,
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
    parser = argparse.ArgumentParser(description='Portugal grid street matching with OSM cache')
    parser.add_argument('--refresh-cache', action='store_true',
                        help='Re-fetch all cells from Overpass, ignoring the cache')
    parser.add_argument('--cell', nargs=2, type=float, metavar=('LAT', 'LON'),
                        help='Process a single cell origin only, e.g. --cell 38.7 -9.1')
    args = parser.parse_args()

    df_all = load_infractions()
    grid   = build_grid(df_all)
    munis  = load_municipios()

    # Filter to a single test cell if requested
    if args.cell:
        clat, clon = round(args.cell[0], 2), round(args.cell[1], 2)
        if (clat, clon) not in grid:
            print(f"No data in cell ({clat}, {clon}) — or fewer than {MIN_POINTS} points.")
            return
        grid = {(clat, clon): grid[(clat, clon)]}
        print(f"Single-cell mode: ({clat}, {clon})")

    if args.refresh_cache:
        print("⚠  Cache refresh mode — all cells will be re-fetched from Overpass.")

    all_features = []
    succeeded = skipped = cached_hits = 0
    cells_list = sorted(grid.items())  # sorted for deterministic order

    for i, ((clat, clon), df_cell) in enumerate(cells_list):
        cache_path = cell_cache_path(clat, clon)
        from_cache = cache_path.exists() and not args.refresh_cache

        print(f"\n── Cell ({clat:.1f}, {clon:.1f})  [{i+1}/{len(cells_list)}]  "
              f"{'📦 cache' if from_cache else '🌐 fetch'}  "
              f"({len(df_cell)} pts) ──")

        edges = get_network(clat, clon, refresh=args.refresh_cache)

        if edges is None:
            skipped += 1
            continue

        agg, snapped, edges_proj = snap_and_aggregate(df_cell, edges, clon)
        features = build_features(edges_proj, agg, munis)
        all_features.extend(features)
        print(f"  {snapped}/{len(df_cell)} snapped → {len(features)} segments")
        succeeded += 1
        if from_cache:
            cached_hits += 1

        # Polite delay only when actually hitting Overpass
        if not from_cache and i < len(cells_list) - 1:
            print(f"  Pausing {DELAY_BETWEEN_SECS}s...")
            time.sleep(DELAY_BETWEEN_SECS)

    print(f"\n{'═'*60}")
    print(f"  Cells processed:  {succeeded}  ({cached_hits} from cache, "
          f"{succeeded - cached_hits} from Overpass)")
    print(f"  Cells skipped:    {skipped}")
    print(f"  Street segments:  {len(all_features)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump({"type": "FeatureCollection", "features": all_features}, f, ensure_ascii=False)

    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"✓ Saved → {OUT_PATH} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
