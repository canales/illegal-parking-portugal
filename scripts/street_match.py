"""
street_match.py
─────────────────────────────────────────────────────────────────────────────
Snaps infraction points to the OSM road network and aggregates per street.
Produces data/streets_matched.geojson for the map's "By street" view.

Architecture
────────────
1. PBF    - The road network is extracted from a Geofabrik .osm.pbf file
            (downloaded fresh each run by the workflow, ~700 MB for Portugal).
            This replaces all Overpass API calls: no rate limits, no mirrors,
            no timeouts, no cache eviction issues.

2. GRID   - Portugal is divided into 0.1 x 0.1 degree cells (~10x10km each).
            Only cells that contain at least MIN_POINTS infraction reports
            are processed.

3. SNAP   - Each cell's infraction points are snapped to the nearest road
            edge using sjoin_nearest, then aggregated per street segment.

Runtime: ~5-10 min on GitHub Actions (PBF download: ~4s, processing: ~5min).

Usage
─────
    pip install pyrosm geopandas shapely pandas pyogrio
    python scripts/street_match.py --pbf data/portugal-latest.osm.pbf

    # Single cell for quick local testing (provide lat/lon of cell origin)
    python scripts/street_match.py --pbf data/portugal-latest.osm.pbf --cell 38.7 -9.1
"""

import re
import json
import math
import argparse
from collections import Counter
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

_DEFAULT_IN   = ROOT / 'data' / 'infractions.geojson'
_DEFAULT_OUT  = ROOT / 'data' / 'streets_matched.geojson'
ROAD_LENGTHS_PATH = ROOT / 'data' / 'road_network_lengths.json'

# ── Grid config ────────────────────────────────────────────────────────────
CELL_SIZE  = 0.1    # degrees, ~10km x 10km at Portugal's latitude
MIN_POINTS = 1      # skip cells with no infraction points at all

# Portugal bounding box (covers mainland + Azores + Madeira)
PT_LAT_MIN, PT_LAT_MAX = 29.0, 43.0
PT_LON_MIN, PT_LON_MAX = -32.0, -6.0

# ── Snap config ────────────────────────────────────────────────────────────
SNAP_THRESHOLD_M = 100  # GPS drift + wide avenues need generous tolerance

# Highway types to EXCLUDE (same logic as the old Overpass custom_filter).
# We keep everything except pure pedestrian infrastructure and construction.
HIGHWAY_EXCLUDE = {
    'footway', 'path', 'steps', 'corridor',
    'elevator', 'escalator', 'construction',
}


# ── Grid helpers ───────────────────────────────────────────────────────────
def cell_origin(lat: float, lon: float) -> tuple:
    """
    Return the bottom-left corner of the 0.1 degree cell containing (lat, lon).

    IMPORTANT: must use math.floor, not int().
    int() truncates toward zero, so int(-91.5) = -91.
    math.floor(-91.5) = -92, which is correct for negative longitudes.
    Without this, a point at lon=-9.19 gets assigned to cell -9.1
    (lon -9.1 to -9.0, east of Lisbon) instead of cell -9.2 (correct).
    """
    return (
        round(math.floor(lat / CELL_SIZE) * CELL_SIZE, 2),
        round(math.floor(lon / CELL_SIZE) * CELL_SIZE, 2)
    )


def cell_bbox(clat: float, clon: float) -> tuple:
    """Return (lat_min, lon_min, lat_max, lon_max) for a cell origin."""
    return (clat, clon, round(clat + CELL_SIZE, 2), round(clon + CELL_SIZE, 2))


# ── Data loading ───────────────────────────────────────────────────────────
def load_infractions(in_path: Path) -> pd.DataFrame:
    with open(in_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    rows = []
    for feat in data['features']:
        lon, lat = feat['geometry']['coordinates']
        if not (PT_LAT_MIN <= lat <= PT_LAT_MAX and PT_LON_MIN <= lon <= PT_LON_MAX):
            continue
        p = feat['properties']
        date_str = p.get('occurredAt') or p.get('data_data', '')
        rows.append({
            'lat': lat, 'lon': lon,
            'infraction_type': p.get('infraction_type', ''),
            'year':  date_str[:4],
            'month': date_str[:7],
        })
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} infraction points (within Portugal bbox).")
    return df


def build_grid(df: pd.DataFrame) -> dict:
    """
    Assign each point to a 0.1 degree grid cell.
    Returns {(clat, clon): sub-DataFrame} for cells with >= MIN_POINTS.
    """
    df = df.copy()
    df['clat'] = df['lat'].apply(lambda v: round(math.floor(v / CELL_SIZE) * CELL_SIZE, 2))
    df['clon'] = df['lon'].apply(lambda v: round(math.floor(v / CELL_SIZE) * CELL_SIZE, 2))

    cells = {}
    for (clat, clon), group in df.groupby(['clat', 'clon']):
        if len(group) >= MIN_POINTS:
            cells[(clat, clon)] = group
    print(f"Grid: {len(cells)} cells with >={MIN_POINTS} points "
          f"(skipped {df.groupby(['clat','clon']).ngroups - len(cells)} sparse cells).")
    return cells


# ── OSM road network from PBF ─────────────────────────────────────────────
def load_roads_from_pbf(pbf_path: Path) -> gpd.GeoDataFrame:
    """
    Read the full road network from a Geofabrik .osm.pbf file using pyrosm.
    Returns a GeoDataFrame of road edges in EPSG:4326 with 'name' and
    'highway' columns, filtered and deduplicated.
    """
    from pyrosm import OSM

    print(f"Loading road network from {pbf_path.name}...")
    osm = OSM(str(pbf_path))

    # get_network('driving+service') includes service roads where illegal
    # parking is common. We then manually filter out pure pedestrian types.
    roads = osm.get_network(network_type='driving+service')
    print(f"  Raw edges from PBF: {len(roads)}")

    # Filter out highway types we don't want
    if 'highway' in roads.columns:
        roads = roads[~roads['highway'].isin(HIGHWAY_EXCLUDE)].copy()

    # Filter out private/no access roads
    if 'access' in roads.columns:
        roads = roads[~roads['access'].isin(['private', 'no'])].copy()

    # Keep only the columns we need (reduces memory significantly)
    keep_cols = ['geometry']
    for col in ['name', 'highway']:
        if col in roads.columns:
            keep_cols.append(col)
    roads = roads[keep_cols].copy()

    # Ensure CRS
    if roads.crs is None:
        roads = roads.set_crs("EPSG:4326")
    elif roads.crs.to_epsg() != 4326:
        roads = roads.to_crs("EPSG:4326")

    # Deduplicate bidirectional edges (if any).
    # pyrosm doesn't typically produce directional duplicates like osmnx,
    # but we check anyway. We use start/end coords for LineStrings and
    # fall back to object id for MultiLineStrings (which are rare).
    def _edge_key(g):
        try:
            return frozenset([g.coords[0], g.coords[-1]])
        except (NotImplementedError, IndexError):
            return id(g)

    roads = roads.reset_index(drop=True)
    roads['_edge_key'] = roads.geometry.apply(_edge_key)
    before = len(roads)
    roads = roads.drop_duplicates(subset='_edge_key').drop(columns='_edge_key')
    roads = roads.reset_index(drop=True)
    print(f"  After filtering + dedup: {len(roads)} edges (removed {before - len(roads)} duplicates)")

    # Build spatial index (used by clip operations later)
    roads.sindex
    return roads


def clip_roads_to_cell(all_roads: gpd.GeoDataFrame, clat: float, clon: float) -> gpd.GeoDataFrame:
    """
    Extract road edges that intersect a given grid cell.
    Uses the spatial index for fast bbox filtering.
    """
    lat_min, lon_min, lat_max, lon_max = cell_bbox(clat, clon)
    cell_box = box(lon_min, lat_min, lon_max, lat_max)

    # Use spatial index for fast candidate selection
    candidates_idx = list(all_roads.sindex.intersection(cell_box.bounds))
    if not candidates_idx:
        return gpd.GeoDataFrame(columns=all_roads.columns, crs=all_roads.crs)

    candidates = all_roads.iloc[candidates_idx]
    mask = candidates.intersects(cell_box)
    result = candidates[mask].copy().reset_index(drop=True)
    return result


# ── Snap & aggregate ───────────────────────────────────────────────────────
def utm_epsg(clon: float) -> str:
    """
    Return the EPSG code for the UTM zone covering a given longitude.
    Works for all of Portugal including Azores (zone 26) and Madeira (zone 28).
    EPSG:3763 only covers mainland.
    """
    zone = int((clon + 180) / 6) + 1
    return f"EPSG:326{zone:02d}"


def snap_and_aggregate(df_cell: pd.DataFrame, edges: gpd.GeoDataFrame, clon: float):
    """
    Snap infraction points to nearest road edge using sjoin_nearest.

    WHY sjoin_nearest instead of a KDTree on midpoints:
    The midpoint approach rejects points near the ends of long street segments
    because they're far from the midpoint. sjoin_nearest measures distance
    to the actual line geometry, so any point within threshold of any part of
    any edge gets matched correctly.

    WHY per-cell UTM instead of EPSG:3763:
    EPSG:3763 (Portugal TM06) is only valid for mainland Portugal. For Azores
    (~-25 deg lon) and Madeira (~-17 deg lon) it produces wrong metric
    coordinates. We compute the correct UTM zone from the cell's longitude.
    """
    epsg = utm_epsg(clon)

    edges_proj = edges.reset_index(drop=True).to_crs(epsg)

    pts = gpd.GeoDataFrame(
        df_cell.copy(),
        geometry=gpd.points_from_xy(df_cell.lon, df_cell.lat),
        crs="EPSG:4326"
    ).to_crs(epsg)

    pts = pts.reset_index(drop=True)
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
        types  = group['infraction_type'].tolist()
        years  = [y for y in group['year'].tolist() if len(y) == 4 and y.isdigit()]
        months = [m for m in group['month'].tolist() if len(m) == 7]
        agg[int(edge_idx)] = {
            'count':             len(group),
            'top_infraction':    Counter(types).most_common(1)[0][0],
            'first_year':        min(years) if years else None,
            'last_year':         max(years) if years else None,
            'type_breakdown':    dict(Counter(types)),
            'monthly_breakdown': dict(Counter(months))
        }

    return agg, len(snapped), edges_proj


def accumulate_road_lengths(edges_proj: gpd.GeoDataFrame,
                            munis: 'gpd.GeoDataFrame | None',
                            road_km_acc: dict) -> None:
    """
    Accumulate total road network length (km) per municipality from ALL edges
    in the cell (not just matched ones). Gives the correct denominator for
    per-km-road rankings.
    """
    if munis is None:
        return
    try:
        edges_proj = edges_proj.reset_index(drop=True).copy()
        edges_proj['_len_m'] = edges_proj.geometry.length

        mids = edges_proj.copy()
        mids['geometry'] = edges_proj.geometry.interpolate(0.5, normalized=True)
        mids_wgs = mids[['geometry', '_len_m']].to_crs('EPSG:4326')

        joined = gpd.sjoin(mids_wgs, munis[['geometry', 'municipio']],
                           how='left', predicate='within')

        for _, row in joined.iterrows():
            muni = row.get('municipio')
            if not isinstance(muni, str):
                continue
            road_km_acc[muni] = road_km_acc.get(muni, 0) + row['_len_m'] / 1000

    except Exception as e:
        print(f"    Warning: road length accumulation failed ({e})")


def load_municipios() -> 'gpd.GeoDataFrame | None':
    """Load municipality boundaries for spatial join. Returns None if missing."""
    munis_path = ROOT / 'data' / 'municipios.geojson'
    if not munis_path.exists():
        print("Warning: municipios.geojson not found. Street municipio tags will be skipped.")
        return None
    try:
        munis = gpd.read_file(str(munis_path))[['geometry', 'NAME_2']].copy()
        munis['NAME_2'] = munis['NAME_2'].apply(
            lambda n: re.sub(r'(?<=[a-záàâãéèêíóôõúç])([A-ZÁÀÂÃÉÈÊÍÓÔÕÚÇ])', r' \1', n) if isinstance(n, str) else n
        )
        munis = munis.rename(columns={'NAME_2': 'municipio'})
        return munis.to_crs("EPSG:4326")
    except Exception as e:
        print(f"Warning: Could not load municipios.geojson ({e}). Skipping municipio tags.")
        return None


def build_features(edges_proj: gpd.GeoDataFrame, agg: dict, munis: 'gpd.GeoDataFrame | None') -> list:
    edges_wgs = edges_proj.to_crs("EPSG:4326")

    muni_by_edge = {}
    if munis is not None:
        try:
            mids = edges_proj.copy()
            mids['geometry'] = edges_proj.geometry.interpolate(0.5, normalized=True)
            mids = mids.to_crs("EPSG:4326")
            joined = gpd.sjoin(mids[['geometry']], munis, how='left', predicate='within')
            muni_by_edge = joined['municipio'].to_dict()
        except Exception as e:
            print(f"    Warning: municipio join failed ({e})")

    features = []
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
                    "street_name":       name,
                    "municipio":         muni,
                    "count":             stats['count'],
                    "top_infraction":    stats['top_infraction'],
                    "first_year":        stats['first_year'],
                    "last_year":         stats['last_year'],
                    "type_breakdown":    json.dumps(stats['type_breakdown']),
                    "monthly_breakdown": json.dumps(stats['monthly_breakdown'])
                }
            })
        except Exception:
            continue
    return features


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Snap infraction points to OSM road network (from Geofabrik PBF)')
    parser.add_argument('--pbf', required=True,
                        help='Path to the Geofabrik .osm.pbf file (e.g. data/portugal-latest.osm.pbf)')
    parser.add_argument('--cell', nargs=2, type=float, metavar=('LAT', 'LON'),
                        help='Process a single cell origin only, e.g. --cell 38.7 -9.1')
    parser.add_argument('--input',  default=str(_DEFAULT_IN),
                        help='Input infractions GeoJSON (default: data/infractions.geojson)')
    parser.add_argument('--output', default=str(_DEFAULT_OUT),
                        help='Output streets GeoJSON (default: data/streets_matched.geojson)')
    args = parser.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output)
    pbf_path = Path(args.pbf)

    if not pbf_path.exists():
        print(f"Error: PBF file not found: {pbf_path}")
        return

    # Load everything up front
    df_all    = load_infractions(in_path)
    grid      = build_grid(df_all)
    munis     = load_municipios()
    all_roads = load_roads_from_pbf(pbf_path)

    # Filter to a single test cell if requested
    if args.cell:
        clat, clon = round(args.cell[0], 2), round(args.cell[1], 2)
        if (clat, clon) not in grid:
            print(f"No data in cell ({clat}, {clon}), or fewer than {MIN_POINTS} points.")
            return
        grid = {(clat, clon): grid[(clat, clon)]}
        print(f"Single-cell mode: ({clat}, {clon})")

    # Process cells
    all_features = []
    road_km_acc  = {}
    succeeded = skipped = 0
    cells_list = sorted(grid.items())

    for i, ((clat, clon), df_cell) in enumerate(cells_list):
        print(f"\n-- Cell ({clat:.1f}, {clon:.1f})  [{i+1}/{len(cells_list)}]  "
              f"({len(df_cell)} pts) --")

        edges = clip_roads_to_cell(all_roads, clat, clon)
        if edges.empty:
            print("  No road edges in cell. Skipping.")
            skipped += 1
            continue

        print(f"  {len(edges)} road edges in cell.")

        agg, snapped, edges_proj = snap_and_aggregate(df_cell, edges, clon)
        features = build_features(edges_proj, agg, munis)
        all_features.extend(features)

        accumulate_road_lengths(edges_proj, munis, road_km_acc)

        print(f"  {snapped}/{len(df_cell)} snapped, {len(features)} segments")
        succeeded += 1

    print(f"\n{'='*60}")
    print(f"  Cells processed: {succeeded}")
    print(f"  Cells skipped:   {skipped}")
    print(f"  Street segments: {len(all_features)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({"type": "FeatureCollection", "features": all_features}, f, ensure_ascii=False)

    size_kb = out_path.stat().st_size / 1024
    print(f"Saved {out_path} ({size_kb:.0f} KB)")

    # Save road network lengths only for the main infractions run
    if in_path == _DEFAULT_IN:
        road_km_rounded = {m: round(km, 3) for m, km in road_km_acc.items()}
        with open(ROAD_LENGTHS_PATH, 'w', encoding='utf-8') as f:
            json.dump(road_km_rounded, f, ensure_ascii=False, indent=2)
        print(f"Saved road lengths: {ROAD_LENGTHS_PATH} ({len(road_km_rounded)} municipalities)")


if __name__ == "__main__":
    main()
