"""
fetch_imbecis.py — weekly incremental fetch of new Imbecis confirmed reports.

Reads the latest occurredAt from the existing infractions_imbecis.geojson,
fetches only newer reports from the API, appends and deduplicates by id,
then runs a spatial join to assign consistent municipio names.

Usage (called by GitHub Actions weekly):
    python scripts/fetch_imbecis.py
"""

import json
import os
import sys
import time
import urllib.request
import urllib.parse
from datetime import datetime, timedelta, timezone
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

API_BASE   = 'https://api.imbecis.app'
ENDPOINT   = '/reports/confirmed'
PAGE_SIZE  = 50      # max allowed by API
SLEEP_S    = 1       # polite pause between pages
RETRIES    = 3       # attempts per page before giving up
RETRY_WAIT_S = 3     # grows with each attempt

# A report is published only after enough users confirm it, so one that happened BEFORE the newest report we
# already hold can still arrive later. Asking only for "newer than the latest occurredAt" would skip it for good.
# So we ask for a window before the latest one; reports already held are skipped by id, so this is safe.
OVERLAP_DAYS = 30


# ── Spatial join ────────────────────────────────────────────────────────────

def assign_municipios(features: list) -> list:
    """Spatial join using our own municipios.geojson for naming consistency."""
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

    joined      = gpd.sjoin(pts, munis, how='left', predicate='within')
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

    print(f"  {assigned} assigned, {unassigned} outside municipality polygons.")
    return features


# ── Record conversion ────────────────────────────────────────────────────────

def record_to_feature(rec: dict) -> Optional[dict]:
    loc = rec.get('location', {})
    lat = loc.get('latitude')
    lon = loc.get('longitude')
    if lat is None or lon is None:
        return None
    # Validate coordinates cover all Portuguese territory including islands:
    # Mainland: ~36.8–42.2°N, ~-9.6–-6.1°W
    # Madeira:  ~32.6–33.1°N, ~-17.3–-16.3°W
    # Azores:   ~36.9–39.7°N, ~-31.3–-25.0°W
    mainland = (-9.6 <= lon <= -6.1 and 36.8 <= lat <= 42.2)
    madeira  = (-17.4 <= lon <= -16.2 and 32.5 <= lat <= 33.2)
    azores   = (-31.4 <= lon <= -24.9 and 36.8 <= lat <= 39.8)
    if not (mainland or madeira or azores):
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


# ── API fetch ────────────────────────────────────────────────────────────────

def overlap_start(latest: Optional[str]) -> Optional[str]:
    """The day OVERLAP_DAYS before the latest report we hold, written the way the API wrote it."""
    if not latest:
        return None
    try:
        day = datetime.strptime(latest[:10], '%Y-%m-%d').date() - timedelta(days=OVERLAP_DAYS)
    except ValueError:
        return latest
    return day.isoformat() + latest[10:]


def get_page(url: str) -> Optional[dict]:
    """One page of the API, retried a few times. None if it never worked."""
    for attempt in range(1, RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={
                'Accept':     'application/json',
                'User-Agent': 'illegal-parking-portugal-map/1.0 (https://canales.github.io/illegal-parking-portugal)',
            })
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            if data.get('success'):
                return data
            print(f"  ⚠ API returned an error (attempt {attempt}/{RETRIES}): {data}")
        except Exception as e:
            print(f"  ⚠ Request failed (attempt {attempt}/{RETRIES}): {e}")
        if attempt < RETRIES:
            time.sleep(RETRY_WAIT_S * attempt)
    return None


def fetch_since(from_time: Optional[str]):
    """Fetch all confirmed reports newer than from_time. Returns (raw records, finished_ok)."""
    all_records = []
    page        = 1

    while True:
        params = {'page': page, 'pageSize': PAGE_SIZE}
        if from_time:
            params['from_time'] = from_time

        url = f"{API_BASE}{ENDPOINT}?{urllib.parse.urlencode(params)}"
        print(f"  Fetching page {page}: {url}")

        data = get_page(url)
        if data is None:
            return all_records, False

        payload    = data.get('payload', [])
        meta       = data.get('meta', {})
        total_pages = meta.get('totalPages', 1)

        all_records.extend(payload)
        print(f"    Got {len(payload)} records (page {page}/{total_pages}, total so far: {len(all_records)})")

        if page >= total_pages:
            break

        page += 1
        time.sleep(SLEEP_S)

    return all_records, True


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"[fetch_imbecis.py] starting — {datetime.now(timezone.utc).isoformat()}")

    # Load existing data
    existing_features = []
    existing_ids      = set()
    last_occurred_at  = None

    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        existing_features = existing.get('features', [])
        existing_ids      = {f['properties']['id'] for f in existing_features}

        # Find the latest occurredAt to use as from_time
        dates = [
            f['properties'].get('occurredAt')
            for f in existing_features
            if f['properties'].get('occurredAt')
        ]
        if dates:
            last_occurred_at = max(dates)

        print(f"  Existing records: {len(existing_features)}")
        print(f"  Latest report we hold: {last_occurred_at or 'none'}")
    else:
        print("  No existing file — fetching all records.")

    # Fetch new records from API
    print("\nFetching from Imbecis API...")
    since = overlap_start(last_occurred_at)
    print(f"  Fetching since:   {since or 'beginning'}  ({OVERLAP_DAYS}-day overlap, duplicates are skipped by id)")
    raw_records, finished_ok = fetch_since(since)
    print(f"  {len(raw_records)} records returned by API.")

    # Convert and deduplicate
    new_features = []
    dupes        = 0
    invalid      = 0

    for rec in raw_records:
        if rec['id'] in existing_ids:
            dupes += 1
            continue
        feat = record_to_feature(rec)
        if feat:
            new_features.append(feat)
            existing_ids.add(rec['id'])
        else:
            invalid += 1

    print(f"  {len(new_features)} new, {dupes} duplicates skipped, {invalid} invalid coords.")

    if not new_features:
        if not finished_ok:
            print("\n✗ The API could not be read, so nothing could be updated.")
            sys.exit(1)           # a non-zero exit makes the weekly run show as failed, instead of looking fine
        print("\nNo new records — nothing to update.")
        return

    # Spatial join only on new features (existing already have municipio)
    print("\nAssigning municipalities to new records...")
    new_features = assign_municipios(new_features)

    # Merge and sort
    all_features = existing_features + new_features
    all_features.sort(key=lambda f: f['properties'].get('occurredAt') or '')

    # Write output
    now = datetime.now(timezone.utc).isoformat()
    dates = [f['properties'].get('occurredAt') for f in all_features if f['properties'].get('occurredAt')]

    geojson = {
        'type':     'FeatureCollection',
        'features': all_features,
        'metadata': {
            'source':       'imbecis',
            'total':        len(all_features),
            'updated_at':   now,
            'new_this_run': len(new_features),
            'date_range': {
                'earliest': min(dates) if dates else None,
                'latest':   max(dates) if dates else None,
            }
        }
    }

    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, ensure_ascii=False, separators=(',', ':'))

    print(f"\n✓ Saved {len(all_features)} total records → {OUT_PATH}")
    print(f"  +{len(new_features)} new this run")
    print(f"  Date range: {min(dates)[:10]} → {max(dates)[:10]}")
    if not finished_ok:
        print("\n✗ The API stopped answering part-way: what arrived was saved, but the update is incomplete.")
        sys.exit(1)


if __name__ == '__main__':
    main()
