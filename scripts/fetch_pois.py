"""
fetch_pois.py
─────────────────────────────────────────────────────────────────────────────
Fetches Points of Interest relevant to vulnerable road users from the
Overpass API (OpenStreetMap) and saves them to data/pois.geojson.

Categories fetched:
  - Schools          (amenity=school)
  - Hospitals        (amenity=hospital)
  - Kindergartens    (amenity=kindergarten)
  - Nursing homes    (social_facility=nursing_home)
  - Assisted living  (social_facility=assisted_living)

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
    'https://overpass.kumi.systems/api/interpreter',  # less busy mirror
    'https://overpass-api.de/api/interpreter',         # main server
]
MAX_AGE_DAYS  = 7
TIMEOUT       = 120

# Portugal mainland bounding box (south, west, north, east)
BBOX = '36.9,-9.5,42.2,-6.2'

# POI type definitions
POI_TYPES = {
    'school':       {'label_en': 'School',      'label_pt': 'Escola',            'color': '#4363D8'},
    'hospital':     {'label_en': 'Hospital',    'label_pt': 'Hospital',           'color': '#E6194B'},
    'kindergarten': {'label_en': 'Kindergarten','label_pt': 'Jardim de infância', 'color': '#F58231'},
    'care_home':    {'label_en': 'Care home',   'label_pt': 'Lar / residência',   'color': '#3CB44B'},
}

# ── Overpass query ─────────────────────────────────────────────────────────
QUERY = f"""
[out:json][timeout:{TIMEOUT}];
(
  node["amenity"="school"]["name"]({BBOX});
  way["amenity"="school"]["name"]({BBOX});
  node["amenity"="hospital"]["name"]({BBOX});
  way["amenity"="hospital"]["name"]({BBOX});
  node["amenity"="kindergarten"]["name"]({BBOX});
  way["amenity"="kindergarten"]["name"]({BBOX});
  node["amenity"="social_facility"]["social_facility"="nursing_home"]["name"]({BBOX});
  way["amenity"="social_facility"]["social_facility"="nursing_home"]["name"]({BBOX});
  node["amenity"="social_facility"]["social_facility"="assisted_living"]["name"]({BBOX});
  way["amenity"="social_facility"]["social_facility"="assisted_living"]["name"]({BBOX});
);
out center;
""".strip()


def is_cache_fresh() -> bool:
    """Return True if the cached file exists and is younger than MAX_AGE_DAYS."""
    if not os.path.exists(SAVE_PATH):
        return False
    mtime = datetime.fromtimestamp(os.path.getmtime(SAVE_PATH), tz=timezone.utc)
    age   = datetime.now(tz=timezone.utc) - mtime
    return age < timedelta(days=MAX_AGE_DAYS)


def fetch_overpass(query: str) -> dict:
    """POST query to Overpass API, trying multiple mirrors with retries."""
    data = urllib.parse.urlencode({'data': query}).encode('utf-8')
    headers = {
        'User-Agent':   'estacionamento-abusivo-map/1.0',
        'Content-Type': 'application/x-www-form-urlencoded',
    }
    attempt = 0
    for url in OVERPASS_URLS:
        for retry in range(2):  # 2 tries per endpoint
            attempt += 1
            try:
                print(f'  Attempt {attempt}: {url}')
                req = urllib.request.Request(url, data=data, headers=headers)
                with urllib.request.urlopen(req, timeout=TIMEOUT + 30) as resp:
                    return json.loads(resp.read().decode('utf-8'))
            except Exception as e:
                print(f'  Failed: {e}')
                wait = 30 * attempt
                print(f'  Waiting {wait}s before next attempt...')
                time.sleep(wait)
    raise RuntimeError('All Overpass API endpoints failed.')


def classify_element(el: dict) -> Optional[str]:
    """Return the POI type key for an OSM element, or None if unrecognised."""
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
    """Convert an OSM element to a GeoJSON feature."""
    tags = el.get('tags', {})

    # Nodes have lat/lon directly; ways have a computed center
    if el['type'] == 'node':
        lat, lon = el.get('lat'), el.get('lon')
    elif el['type'] == 'way':
        center = el.get('center', {})
        lat, lon = center.get('lat'), center.get('lon')
    else:
        return None

    if lat is None or lon is None:
        return None

    type_info = POI_TYPES[poi_type]

    return {
        'type': 'Feature',
        'geometry': {
            'type': 'Point',
            'coordinates': [round(lon, 6), round(lat, 6)]
        },
        'properties': {
            'osm_id':     el.get('id'),
            'osm_type':   el['type'],
            'poi_type':   poi_type,
            'name':       tags.get('name', ''),
            'label_en':   type_info['label_en'],
            'label_pt':   type_info['label_pt'],
            'color':      type_info['color'],
            'address':    tags.get('addr:street', ''),
            'operator':   tags.get('operator', ''),
        }
    }


def main():
    print(f'[{datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}] fetch_pois.py starting...')

    if is_cache_fresh():
        print(f'Cache is fresh (< {MAX_AGE_DAYS} days old). Skipping fetch.')
        return

    print('Fetching POI data from Overpass API...')
    raw = fetch_overpass(QUERY)

    elements  = raw.get('elements', [])
    features  = []
    skipped   = 0
    by_type   = {k: 0 for k in POI_TYPES}

    for el in elements:
        poi_type = classify_element(el)
        if not poi_type:
            skipped += 1
            continue
        feat = element_to_feature(el, poi_type)
        if feat:
            features.append(feat)
            by_type[poi_type] += 1
        else:
            skipped += 1

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

    print(f'\nSaved {len(features)} POIs → {SAVE_PATH}')
    print(f'  Skipped: {skipped} (unclassified or missing coordinates)')
    print('  By type:')
    for k, n in by_type.items():
        print(f'    {k:<20} {n}')


if __name__ == '__main__':
    main()
