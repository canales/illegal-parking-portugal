"""
seed_imbecis.py — one-time import of historical Imbecis reports from a JSONL dump.

Usage:
    python scripts/seed_imbecis.py path/to/confirmed-reports.jsonl

Output:
    data/infractions_imbecis.geojson
"""

import json
import os
import sys
from datetime import datetime, timezone
from typing import Optional

try:
    import geopandas as gpd
    from shapely.geometry import Point
    HAS_GPD = True
except ImportError:
    HAS_GPD = False

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MUNIS_PATH = os.path.join(ROOT, 'data', 'municipios.geojson')
OUT_PATH   = os.path.join(ROOT, 'data', 'infractions_imbecis.geojson')


def assign_municipios(features: list) -> list:
    """Spatial join: tag every feature with its municipality name from our own GeoJSON.
    Ignores Imbecis-provided municipality to ensure naming consistency."""
    if not HAS_GPD:
        print("⚠  geopandas not installed — skipping municipality assignment.")
        return features
    if not os.path.exists(MUNIS_PATH):
        print("⚠  municipios.geojson not found — skipping municipality assignment.")
        return features

    import pandas as pd
    munis = gpd.read_file(MUNIS_PATH)[['geometry', 'NAME_2']].copy()
    munis = munis.rename(columns={'NAME_2': 'municipio'})
    munis = munis.to_crs(epsg=4326)

    pts = gpd.GeoDataFrame(
        [{'idx': i, 'geometry': Point(f['geometry']['coordinates'])}
         for i, f in enumerate(features)],
        crs='EPSG:4326'
    )

    joined     = gpd.sjoin(pts, munis, how='left', predicate='within')
    muni_by_idx = joined.set_index('idx')['municipio'].to_dict()

    assigned = unassigned = 0
    for i, feat in enumerate(features):
        m = muni_by_idx.get(i)
        if pd.notna(m) and m:
            feat['properties']['municipio'] = m
            assigned += 1
        else:
            feat['properties']['municipio'] = None
            unassigned += 1

    print(f"  {assigned} points assigned, {unassigned} outside all municipality polygons.")
    return features


def record_to_feature(rec: dict) -> Optional[dict]:
    """Convert a single JSONL record to a GeoJSON Feature."""
    loc = rec.get('location', {})
    lat = loc.get('latitude')
    lon = loc.get('longitude')

    if lat is None or lon is None:
        return None

    # Validate coordinates are within Portugal's bounding box (approx)
    if not (-9.6 <= lon <= -6.1 and 36.8 <= lat <= 42.2):
        return None

    occurred_at = rec.get('occurredAt') or rec.get('createdAt')

    return {
        'type': 'Feature',
        'geometry': {
            'type':        'Point',
            'coordinates': [round(lon, 6), round(lat, 6)]
        },
        'properties': {
            'id':              rec['id'],
            'occurredAt':      occurred_at,
            'infraction_type': 'imbecis',
            'source':          'imbecis',
            'municipio':       None,  # filled by spatial join
            'picture':         rec.get('picture'),
        }
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/seed_imbecis.py path/to/confirmed-reports.jsonl")
        sys.exit(1)

    jsonl_path = sys.argv[1]
    if not os.path.exists(jsonl_path):
        print(f"Error: file not found: {jsonl_path}")
        sys.exit(1)

    print(f"Reading {jsonl_path}...")
    records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  ⚠ Skipping malformed line: {e}")

    print(f"  {len(records)} records read.")

    # Convert to GeoJSON features
    features = []
    skipped  = 0
    for rec in records:
        feat = record_to_feature(rec)
        if feat:
            features.append(feat)
        else:
            skipped += 1

    print(f"  {len(features)} valid features, {skipped} skipped (missing/invalid coords).")

    # Spatial join for consistent municipio naming
    print("Assigning municipalities via spatial join...")
    features = assign_municipios(features)

    # Sort by occurredAt ascending for consistent output
    features.sort(key=lambda f: f['properties'].get('occurredAt') or '')

    # Write output
    now = datetime.now(timezone.utc).isoformat()
    geojson = {
        'type':     'FeatureCollection',
        'features': features,
        'metadata': {
            'source':       'imbecis',
            'total':        len(features),
            'seeded_at':    now,
            'date_range': {
                'earliest': features[0]['properties']['occurredAt'] if features else None,
                'latest':   features[-1]['properties']['occurredAt'] if features else None,
            }
        }
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, ensure_ascii=False, separators=(',', ':'))

    print(f"\n✓ Saved {len(features)} features → {OUT_PATH}")
    print(f"  Date range: {geojson['metadata']['date_range']['earliest'][:10]} → "
          f"{geojson['metadata']['date_range']['latest'][:10]}")


if __name__ == '__main__':
    main()
