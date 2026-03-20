"""
fetch_data.py
─────────────────────────────────────────────────────────────────────────────
Fetches all infraction records from api.denuncia-estacionamento.app
and saves a clean GeoJSON to data/infractions.geojson.

Runs as a standalone script (no QGIS dependency).
Called by GitHub Actions weekly, or run locally anytime.

Usage:
    python scripts/fetch_data.py
"""

import json
import os
import urllib.request
import urllib.parse
from datetime import datetime, timezone

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_PATH = os.path.join(ROOT, 'data', 'infractions.geojson')

# ── API ────────────────────────────────────────────────────────────────────
API_BASE_URL      = "https://api.denuncia-estacionamento.app/penalties/"
PENALTIES_LIST_URL = "https://api.denuncia-estacionamento.app/penalties_list"

# ── Portugal bounding box — filter out bad GPS coordinates ─────────────────
# Mainland + Madeira + Azores covered by generous bbox
LAT_MIN, LAT_MAX = 29.0, 43.0
LON_MIN, LON_MAX = -32.0, -6.0


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={'User-Agent': 'estacionamento-abusivo-map/1.0'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode('utf-8'))


def is_valid_coord(lat: float, lon: float) -> bool:
    """Reject coordinates outside Portugal (catches bad GPS readings)."""
    return LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX


def load_existing() -> tuple[list, set]:
    """Load existing GeoJSON, return (features list, seen_keys set)."""
    if not os.path.exists(SAVE_PATH):
        print("No existing file found — starting fresh.")
        return [], set()
    with open(SAVE_PATH, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            features = data.get("features", [])
            seen = set()
            for feat in features:
                coords = feat["geometry"]["coordinates"]
                date  = feat["properties"].get("data_data", "")
                seen.add(f"{coords[0]}_{coords[1]}_{date}")
            print(f"Loaded {len(features)} existing records.")
            return features, seen
        except Exception as e:
            print(f"Could not parse existing file ({e}) — starting fresh.")
            return [], set()


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

            # Filter out-of-Portugal GPS noise
            if not is_valid_coord(lat_f, lon_f):
                skipped_geo += 1
                continue

            date_val = attrs.get("data_data", "")
            key = f"{lon_f}_{lat_f}_{date_val}"

            if key not in seen_keys:
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon_f, lat_f]},
                    "properties": {
                        "infraction_type": penalty_id,
                        "data_data":       date_val,
                        "data_hora":       attrs.get("data_hora", ""),
                        "autoridade":      attrs.get("autoridade", "Desconhecida")
                    }
                })
                seen_keys.add(key)
                new_count += 1

        print(f"  {penalty_id}: done ({new_count} new so far)")

    print(f"\nFinished. {new_count} new records added. {skipped_geo} out-of-Portugal coords skipped.")

    if new_count > 0 or not os.path.exists(SAVE_PATH):
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        with open(SAVE_PATH, 'w', encoding='utf-8') as f:
            json.dump({"type": "FeatureCollection", "features": features}, f,
                      ensure_ascii=False)
        print(f"Saved {len(features)} total records → {SAVE_PATH}")
    else:
        print("No new records. File unchanged.")


if __name__ == "__main__":
    main()
