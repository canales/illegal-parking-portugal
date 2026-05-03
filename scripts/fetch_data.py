"""
fetch_data.py
─────────────────────────────────────────────────────────────────────────────
Fetches all infraction records from api.denuncia-estacionamento.app
and saves a clean GeoJSON to data/infractions.geojson.

Also performs a spatial join against data/municipios.geojson to tag
every point with its municipality name in the `municipio` property.
This is done for ALL records every run (re-assignment is idempotent
and takes ~5s with GeoPandas).

Usage:
    pip install geopandas pyogrio shapely
    python scripts/fetch_data.py
"""

import json
import os
import urllib.request
import urllib.parse
from datetime import datetime, timezone

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_PATH  = os.path.join(ROOT, 'data', 'infractions.geojson')
MUNIS_PATH = os.path.join(ROOT, 'data', 'municipios.geojson')

# ── API ────────────────────────────────────────────────────────────────────
API_BASE_URL       = "https://api.denuncia-estacionamento.app/penalties/"
PENALTIES_LIST_URL = "https://api.denuncia-estacionamento.app/penalties_list"

# ── Portugal bounding box ──────────────────────────────────────────────────
LAT_MIN, LAT_MAX = 29.0, 43.0
LON_MIN, LON_MAX = -32.0, -6.0


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={'User-Agent': 'estacionamento-abusivo-map/1.0'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode('utf-8'))


def is_valid_coord(lat: float, lon: float) -> bool:
    return LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX


def load_existing():
    if not os.path.exists(SAVE_PATH):
        print("No existing file found — starting fresh.")
        return [], set()
    with open(SAVE_PATH, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            features = data.get("features", [])
            seen = set()
            for feat in features:
                p = feat["properties"]
                # Prefer id-based dedup; fall back to composite key for old records
                if p.get("id"):
                    seen.add(p["id"])
                else:
                    coords = feat["geometry"]["coordinates"]
                    seen.add(f"{coords[0]}_{coords[1]}_{p.get('occurredAt', '')}")
            print(f"Loaded {len(features)} existing records.")
            return features, seen
        except Exception as e:
            print(f"Could not parse existing file ({e}) — starting fresh.")
            return [], set()


def build_occurred_at(data_data: str, data_hora: str) -> str:
    """Combine data_data (date) and data_hora (time) into a single ISO 8601 string.

    data_data is stored as midnight UTC e.g. '2020-06-23T22:00:00.000Z'
    data_hora is the local time e.g. '19:31:00'
    We use the date part of data_data and attach data_hora to form a naive
    local datetime string: '2020-06-23T19:31:00'
    """
    if not data_data:
        return ''
    date_part = data_data[:10]   # 'YYYY-MM-DD'
    if data_hora and len(data_hora) >= 5:
        return f"{date_part}T{data_hora}"
    return f"{date_part}T00:00:00"


def assign_municipios(features: list) -> list:
    """
    Spatial join: tag every feature with its municipality name.
    Uses GeoPandas sjoin for speed (~5s for 16k points).
    Falls back gracefully if municipios.geojson is missing.
    """
    if not os.path.exists(MUNIS_PATH):
        print("⚠  municipios.geojson not found — skipping municipality assignment.")
        print("   Commit data/municipios.geojson to enable city rankings.")
        return features

    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError:
        print("⚠  geopandas not installed — skipping municipality assignment.")
        return features

    print("Assigning municipalities via spatial join...")

    import re
    munis = gpd.read_file(MUNIS_PATH)[['geometry', 'NAME_2']].copy()
    # GADM NAME_2 field has no spaces (e.g. "MarinhaGrande").
    # Insert a space before each capital letter that follows a lowercase letter.
    munis['NAME_2'] = munis['NAME_2'].apply(
        lambda n: re.sub(r'(?<=[a-záàâãéèêíóôõúç])([A-ZÁÀÂÃÉÈÊÍÓÔÕÚÇ])', r' \1', n) if isinstance(n, str) else n
    )
    munis = munis.rename(columns={'NAME_2': 'municipio'})
    munis = munis.to_crs("EPSG:4326")

    # Build points GeoDataFrame
    lons = [f["geometry"]["coordinates"][0] for f in features]
    lats = [f["geometry"]["coordinates"][1] for f in features]
    pts  = gpd.GeoDataFrame(
        {'idx': range(len(features))},
        geometry=[Point(lon, lat) for lon, lat in zip(lons, lats)],
        crs="EPSG:4326"
    )

    joined = gpd.sjoin(pts, munis, how='left', predicate='within')

    # Write municipio back into features
    muni_by_idx = joined.set_index('idx')['municipio'].to_dict()
    assigned = unassigned = 0
    for i, feat in enumerate(features):
        m = muni_by_idx.get(i)
        if isinstance(m, str) and m:
            feat["properties"]["municipio"] = m
            assigned += 1
        else:
            feat["properties"]["municipio"] = None
            unassigned += 1

    print(f"  {assigned} points assigned, {unassigned} outside all municipality polygons.")
    return features


def main():
    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC] Starting data fetch...")

    features, seen_keys = load_existing()

    # 1. Get penalty types
    print("Fetching penalty type list...")
    try:
        penalties_resp = fetch_json(PENALTIES_LIST_URL)
    except Exception as e:
        print(f"ERROR fetching penalties list: {e}")
        return

    penalty_keys = [p["id"] for p in penalties_resp.get("data", []) if "id" in p]
    print(f"Found {len(penalty_keys)} infraction types: {', '.join(penalty_keys)}")

    # 2. Fetch each type
    new_count = 0
    skipped_geo = 0

    for penalty_id in penalty_keys:
        url = API_BASE_URL + urllib.parse.quote(penalty_id)
        try:
            resp = fetch_json(url)
        except urllib.error.HTTPError:
            continue
        except Exception as e:
            print(f"  WARNING: could not fetch {penalty_id}: {e}")
            continue

        for item in resp.get("data", []):
            attrs = item.get("attributes", {})
            lat = attrs.get("data_coord_latit")
            lon = attrs.get("data_coord_long")
            if lat is None or lon is None:
                continue
            try:
                lat_f, lon_f = float(lat), float(lon)
            except ValueError:
                continue

            if not is_valid_coord(lat_f, lon_f):
                skipped_geo += 1
                continue

            record_id = item.get("id", "")
            date_val  = attrs.get("data_data", "")
            hora_val  = attrs.get("data_hora", "")
            occurred  = build_occurred_at(date_val, hora_val)

            # Deduplicate: prefer id if available, else composite key for legacy
            dedup_key = record_id if record_id else f"{lon_f}_{lat_f}_{date_val}"

            if dedup_key not in seen_keys:
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon_f, lat_f]},
                    "properties": {
                        "id":              record_id,
                        "infraction_type": penalty_id,
                        "occurredAt":      occurred,
                        "autoridade":      attrs.get("autoridade", "Desconhecida"),
                        "source":          "denuncia_estacionamento",
                    }
                })
                seen_keys.add(dedup_key)
                new_count += 1

        print(f"  {penalty_id}: done ({new_count} new so far)")

    print(f"\nFinished. {new_count} new records added. {skipped_geo} out-of-Portugal coords skipped.")

    # 3. Spatial join — assign municipio to ALL records every run
    features = assign_municipios(features)

    # 4. Save infractions
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    with open(SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump({"type": "FeatureCollection", "features": features}, f,
                  ensure_ascii=False)
    print(f"Saved {len(features)} total records → {SAVE_PATH}")

    # 5. Save stats.json — picked up by the map to show the update badge
    stats_path = os.path.join(ROOT, 'data', 'stats.json')
    stats = {
        "new_count":    new_count,
        "total":        len(features),
        "updated_date": datetime.now(timezone.utc).strftime('%Y-%m-%d')
    }
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f)
    print(f"Saved stats → {stats_path}")


if __name__ == "__main__":
    main()
