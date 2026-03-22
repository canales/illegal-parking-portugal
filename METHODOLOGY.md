# Methodology

## Illegal Parking in Portugal — Map & Data Pipeline

This document explains how data is collected, processed, and displayed in the
[Illegal Parking in Portugal](https://canales.github.io/illegal-parking-portugal) map.

---

## 1. Data Source and Collection

Reports are submitted by users through the
[Denúncia Estacionamento](https://denuncia-estacionamento.app) app. Each record
contains the following fields:

| Field | Description |
|---|---|
| `lat` / `lon` | GPS coordinates at time of report |
| `infraction_type` | One of 47 infraction categories (e.g. `passeios`, `garagem`) |
| `autoridade` | Police authority the report was sent to |
| `data_data` | Date of the report (YYYY-MM-DD) |
| `data_hora` | Time of the report |

Data is fetched weekly from the public API at `https://api.denuncia-estacionamento.app`
using an incremental strategy: only records not already present in the local dataset
are appended, identified by a composite key of `(longitude, latitude, date)`.

---

## 2. Geographic Filtering

A bounding box covering mainland Portugal, Madeira, and the Azores is applied
to every incoming record:

```
LAT: 29.0 – 43.0
LON: -32.0 – -6.0
```

Records outside this box (typically GPS errors placing reports in other
countries) are discarded. Approximately 53 such records have been filtered
to date.

---

## 3. Infraction Type Taxonomy

The dataset contains 47 infraction types as defined by the Denúncia
Estacionamento app. The list expanded over time as the app added new
categories, so some types have fewer historical records than others.
The full list is fetched dynamically from the API's `/penalties_list`
endpoint at each update run.

---

## 4. Grid-Based Street Matching

Portugal's territory is divided into a 0.1° × 0.1° grid (approximately
10 × 10 km per cell at Portugal's latitude). Only grid cells containing
at least 5 infraction reports are processed — cells with fewer reports
are skipped to avoid unnecessary Overpass API queries for areas with
negligible data.

For each qualifying cell, the OSM road network is fetched from the
Overpass API using `osmnx` and saved as a GeoPackage file in
`data/osm_cache/`. Subsequent runs load from this cache, so Overpass
is only queried once per cell. The cache is preserved between GitHub
Actions runs using `actions/cache` and is refreshed monthly.

---

## 5. Coordinate Projection and UTM Zones

Metric distance calculations (required for snapping and clustering) use
the UTM projection appropriate to each grid cell's longitude:

- **Mainland Portugal** (lon ≈ -9° to -6°): UTM zone 29N (EPSG:32629)
- **Madeira** (lon ≈ -17°): UTM zone 28N (EPSG:32628)
- **Azores** (lon ≈ -25° to -32°): UTM zone 26N (EPSG:32626)

The previously used EPSG:3763 (Portugal TM06) is only valid for the
mainland. Applying it to the islands produces severely distorted metric
coordinates, which caused near-zero snap rates before this was corrected.

---

## 6. Snap Threshold

Each infraction point is snapped to its nearest road segment using
`geopandas.sjoin_nearest` with a maximum distance of **50 metres**.

The previous approach (KDTree on edge midpoints) was replaced because it
measured distance to the *midpoint* of each segment rather than to the
nearest point on the line geometry. A report 10m from the end of a 500m
street segment would be ~260m from its midpoint and incorrectly rejected.
`sjoin_nearest` measures distance to the actual line geometry and handles
this correctly.

When a point is equidistant from two edges (e.g. at an intersection),
`sjoin_nearest` returns one row per match. Duplicates are resolved by
retaining only the closest match per original point index, ensuring each
report is counted exactly once.

---

## 7. Municipality Assignment

Municipality names are assigned to both infraction points and street
segments using a spatial join against GADM level-2 boundaries for
Portugal (`data/municipios.geojson`).

The join uses `geopandas.sjoin` with the `within` predicate: each point
or segment midpoint is tested against the 308 municipality polygons, and
the matching `NAME_2` field is written as the `municipio` property.
Points outside all polygons (typically offshore or bad GPS) receive
`municipio: null` and are excluded from city and street rankings.

---

## 8. Street Deduplication — Union-Find Clustering

### The Problem

OSM stores streets as collections of short segments between intersections.
A single street like Av. da Liberdade is represented as dozens of separate
features. Naive grouping by `street_name` alone would incorrectly merge
unrelated streets that happen to share a name across different
neighbourhoods (e.g. "Rua de Santo António" appears in both Belém and
Mouraria in Lisbon).

### The Solution

Streets are deduplicated using **Union-Find (Disjoint Set Union) spatial
clustering** within each municipality:

1. All segments sharing the same `street_name` and `municipio` are
   identified as candidates.
2. For each pair of candidate segments, the Haversine distance between
   their midpoints is computed.
3. Two segments are merged into the same cluster if their midpoints are
   within **400 metres** of each other — not just the first segment in
   the group, but any segment already in the cluster. This
   *nearest-neighbour chaining* correctly handles long streets where
   consecutive segments are 50–150m apart but first-to-last distance
   exceeds the threshold.
4. Clusters are aggregated: report counts and type breakdowns are summed,
   and a bounding box is computed from all segment coordinates.

### Disambiguation

When multiple clusters share the same name within a municipality (i.e.
genuinely separate streets with the same name), they are distinguished
with a numeric suffix: `Rua de Santo António`, `Rua de Santo António (2)`,
etc.

### Threshold Rationale

400m was chosen to be generous enough to bridge gaps at intersections or
where OSM is missing intermediate segments, while tight enough to keep
genuinely separate streets in different clusters. Typical Lisbon block
spacing is 50–150m, so 400m provides ample tolerance without merging
streets in different neighbourhoods.

---

## 9. City Rankings — Total vs Per Km of Road

**Total** is the raw count of infraction reports per municipality for the
selected time period and type filter.

**Per km of road** normalises the total by the municipality's road network
length, calculated by summing the Haversine length of all matched street
segments whose midpoint falls within the municipality. This metric
controls for city size: a municipality with 100km of road and 500 reports
scores 5.0, while one with 10km of road and 100 reports scores 10.0 —
revealing disproportionate concentrations independent of geographic extent.

---

## 10. Viewport-Reactive Statistics

The *Top Infractions* panel and the *City Rankings* panel update
dynamically as the user pans or zooms the map.

Rather than using MapLibre's `queryRenderedFeatures` (which has a hard
tile-rendering cap that causes undercounting at any zoom level), the
panels filter the in-memory source data array against the map's current
bounding box using `map.getBounds()`. This approach is always exact
regardless of zoom level — zooming to a single neighbourhood shows only
that neighbourhood's infractions, while zooming out to all of Portugal
shows the full national picture.

---

## 11. Known Limitations

- **GPS accuracy**: User-submitted coordinates have typical accuracy of
  10–50m. Some reports may be snapped to an adjacent street rather than
  the exact location of the infraction.

- **OSM coverage**: Rural areas may have incomplete road network data in
  OpenStreetMap, leading to lower snap rates in those cells.

- **Reporting bias**: Reports reflect where app users are active, not
  necessarily where illegal parking is most prevalent. Urban areas with
  more app users will appear more prominent in the data.

- **Authority field**: The `autoridade` field reflects where the user
  chose to direct their report, not always the municipality of the
  infraction. For this reason, municipality assignment uses GPS-based
  spatial join rather than authority name parsing.

- **Unnamed streets**: OSM segments without a `name` property are matched
  and counted in the heatmap and points layers but excluded from the
  Street Rankings panel, as they cannot be meaningfully ranked or
  identified.

---

## 12. Data Update Cadence

The full data pipeline runs automatically every **Monday at 06:00 UTC**
via GitHub Actions, and can be triggered manually at any time.

Each run:
1. Fetches new infraction records incrementally from the API
2. Re-runs the municipality spatial join across all records
3. Re-runs street matching for all grid cells (using cache where available)
4. Commits updated `data/infractions.geojson` and `data/streets_matched.geojson`
   back to the repository, triggering a GitHub Pages redeploy

The OSM network cache is preserved between runs. A cell is only
re-fetched from Overpass if the cache file is absent or if `--refresh-cache`
is passed explicitly.

---

*Last updated: March 2026*
*Map: [canales.github.io/illegal-parking-portugal](https://canales.github.io/illegal-parking-portugal)*
*Data source: [denuncia-estacionamento.app](https://denuncia-estacionamento.app)*
