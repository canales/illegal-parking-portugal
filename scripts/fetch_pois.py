"""
fetch_pois.py
─────────────────────────────────────────────────────────────────────────────
Fetches Points of Interest relevant to vulnerable road users from the
Overpass API (OpenStreetMap) and saves them to data/pois.geojson.

Uses a grid-based approach identical to street_match.py — splits Portugal
into 0.5° × 0.5° cells (~55 cells) and queries each one individually.
This avoids Overpass timeouts and rate limiting that occur with a single
full-country query.

Categories fetched:
  - Schools          (amenity=school)
  - Hospitals        (amenity=hospital)
  - Kindergartens    (amenity=kindergarten)
  - Care homes       (social_facility=nursing_home + assisted_living, merged)

The file is cached and only re-fetched if older than MAX_AGE_DAYS.

Usage:
    python scripts/fetch_pois.py
"""

import json
import os
import time
import urllib.request
import urllib.parse
from datetime import datetime, timezone, timedelta
from typing import Optional

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_PATH = os.path.join(ROOT, 'data', 'pois.geojson')

# ── Config ─────────────────────────────────────────────────────────────────
OVERPASS_URLS = [
    'https://overpass.kumi.systems/api/interpreter',  # mirror — less busy
    'https://overpass-api.de/api/interpreter',         # main server fallback
]
MAX_AGE_DAYS = 7    # re-fetch if file is older than this
TIMEOUT      = 60   # per-cell query timeout (seconds) — small cells are fast
CELL_SIZE    = 0.5  # degrees — ~55km cells, ~55 cells cover Portugal
SLEEP_S      = 2    # seconds between cells — polite to the API

# Portugal mainland bounding box (south, west, north, east)
PT_SOUTH, PT_WEST, PT_NORTH, PT_EAST = 36.9, -9.5, 42.2, -6.2

# POI type definitions
POI_TYPES = {
    'school':       {'label_en': 'School',       'label_pt': 'Escola',            'color': '#4363D8'},
    'hospital':     {'label_en': 'Hospital',     'label_pt': 'Hospital',           'color': '#E6194B'},
    'kindergarten': {'label_en': 'Kindergarten', 'label_pt': 'Jardim de infância', 'color': '#F58231'},
    'care_home':    {'label_en': 'Care home',    'label_pt': 'Lar / residência',   'color': '#3CB44B'},
}


# ── Grid ───────────────────────────────────────────────────────────────────
def generate_cells(south, west, north, east, cell_size):
    """Generate (s, w, n, e) bounding boxes covering the given area."""
    cells = []
    lat = south
    while lat < north:
        lon = west
        cell_n = min(round(lat + cell_size, 4), north)
        while lon < east:
            cell_e = min(round(lon + cell_size, 4), east)
            cells.append((round(lat, 4), round(lon, 4), cell_n, cell_e))
            lon = round(lon + cell_size, 4)
        lat = round(lat + cell_size, 4)
    return cells


# ── Overpass ───────────────────────────────────────────────────────────────
def build_query(s, w, n, e):
    bbox = f'{s},{w},{n},{e}'
    return f"""[out:json][timeout:{TIMEOUT}];
(
  node["amenity"="school"]["name"]({bbox});
  way["amenity"="school"]["name"]({bbox});
  node["amenity"="hospital"]["name"]({bbox});
  way["amenity"="hospital"]["name"]({bbox});
  node["amenity"="kindergarten"]["name"]({bbox});
  way["amenity"="kindergarten"]["name"]({bbox});
  node["amenity"="social_facility"]["social_facility"="nursing_home"]["name"]({bbox});
  way["amenity"="social_facility"]["social_facility"="nursing_home"]["name"]({bbox});
  node["amenity"="social_facility"]["social_facility"="assisted_living"]["name"]({bbox});
  way["amenity"="social_facility"]["social_facility"="assisted_living"]["name"]({bbox});
);
out center;"""


def fetch_cell(query: str, cell_num: int, total: int) -> list:
    """POST a single cell query to Overpass, trying both endpoints."""
    data    = urllib.parse.urlencode({'data': query}).encode('utf-8')
    headers = {
        'User-Agent':   'estacionamento-abusivo-map/1.0',
        'Content-Type': 'application/x-www-form-urlencoded',
    }
    for url in OVERPASS_URLS:
        for attempt in range(2):
            try:
                req = urllib.request.Request(url, data=data, headers=headers)
                with urllib.request.urlopen(req, timeout=TIMEOUT + 15) as resp:
                    return json.loads(resp.read().decode('utf-8')).get('elements', [])
            except Exception as e:
                wait = 10 + 10 * attempt
                print(f'\n    ✗ {url} failed: {e} — waiting {wait}s')
                time.sleep(wait)
    print(f'\n    ✗ Cell {cell_num}/{total} failed on all endpoints — skipping')
    return []


# ── Classification ─────────────────────────────────────────────────────────
def classify_element(el: dict) -> Optional[str]:
    tags    = el.get('tags', {})
    amenity = tags.get('amenity', '')
    sf      = tags.get('social_facility', '')
    if amenity == 'school':       return 'school'
    if amenity == 'hospital':     return 'hospital'
    if amenity == 'kindergarten': return 'kindergarten'
    if amenity == 'social_facility' and sf in ('nursing_home', 'assisted_living'):
        return 'care_home'
    return None


def element_to_feature(el: dict, poi_type: str) -> Optional[dict]:
    tags = el.get('tags', {})
    if el['type'] == 'node':
        lat, lon = el.get('lat'), el.get('lon')
    elif el['type'] == 'way':
        center = el.get('center', {})
        lat, lon = center.get('lat'), center.get('lon')
    else:
        return None
    if lat is None or lon is None:
        return None
    info = POI_TYPES[poi_type]
    return {
        'type': 'Feature',
        'geometry': {'type': 'Point', 'coordinates': [round(lon, 6), round(lat, 6)]},
        'properties': {
            'osm_id':   el.get('id'),
            'osm_type': el['type'],
            'poi_type': poi_type,
            'name':     tags.get('name', ''),
            'label_en': info['label_en'],
            'label_pt': info['label_pt'],
            'color':    info['color'],
            'address':  tags.get('addr:street', ''),
            'operator': tags.get('operator', ''),
        }
    }


# ── Cache check ────────────────────────────────────────────────────────────
def is_cache_fresh() -> bool:
    if not os.path.exists(SAVE_PATH):
        return False
    mtime = datetime.fromtimestamp(os.path.getmtime(SAVE_PATH), tz=timezone.utc)
    return datetime.now(tz=timezone.utc) - mtime < timedelta(days=MAX_AGE_DAYS)


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print(f'[{datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}] fetch_pois.py starting...')

    if is_cache_fresh():
        print(f'Cache is fresh (< {MAX_AGE_DAYS} days old). Skipping fetch.')
        return

    cells = generate_cells(PT_SOUTH, PT_WEST, PT_NORTH, PT_EAST, CELL_SIZE)
    print(f'Grid: {len(cells)} cells at {CELL_SIZE}° resolution')

    seen_ids = set()   # (type, id) — deduplication across cell boundaries
    features = []
    by_type  = {k: 0 for k in POI_TYPES}
    skipped  = 0

    for i, (s, w, n, e) in enumerate(cells, 1):
        print(f'  Cell {i:2d}/{len(cells)}: ({s},{w})→({n},{e})', end=' ', flush=True)
        elements  = fetch_cell(build_query(s, w, n, e), i, len(cells))
        cell_new  = 0

        for el in elements:
            osm_id = (el['type'], el.get('id'))
            if osm_id in seen_ids:
                continue
            seen_ids.add(osm_id)

            poi_type = classify_element(el)
            if not poi_type:
                skipped += 1
                continue
            feat = element_to_feature(el, poi_type)
            if feat:
                features.append(feat)
                by_type[poi_type] += 1
                cell_new += 1
            else:
                skipped += 1

        print(f'→ {cell_new} POIs (total so far: {len(features)})')
        if i < len(cells):
            time.sleep(SLEEP_S)

    geojson = {
        'type':     'FeatureCollection',
        'features': features,
        'metadata': {
            'fetched_at': datetime.now(tz=timezone.utc).isoformat(),
            'total':      len(features),
            'by_type':    by_type,
            'source':     'OpenStreetMap via Overpass API',
        }
    }

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    with open(SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, ensure_ascii=False, separators=(',', ':'))

    print(f'\nDone. Saved {len(features)} POIs → {SAVE_PATH}')
    print(f'  Skipped: {skipped} (unclassified or missing coordinates)')
    print('  By type:')
    for k, n in by_type.items():
        print(f'    {k:<20} {n}')


if __name__ == '__main__':
    main()
