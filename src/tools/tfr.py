"""Live FAA Temporary Flight Restriction (TFR) lookup & geospatial tool."""

import math
import time

import requests

TFR_LIST_URL = "https://tfr.faa.gov/tfrapi/exportTfrList"
TFR_WFS_URL = (
    "https://tfr.faa.gov/geoserver/TFR/ows"
    "?service=WFS&version=1.1.0&request=GetFeature"
    "&typeName=TFR:V_TFR_LOC&outputFormat=application/json&maxFeatures=300"
)
_CACHE_TTL_SECONDS = 300
_list_cache: tuple[float, list[dict]] | None = None
_geo_cache: tuple[float, list[dict]] | None = None


def fetch_tfr_list() -> list[dict]:
    """Fetch active/scheduled TFRs from ``tfr.faa.gov`` with a 5-minute cache."""
    global _list_cache
    now = time.monotonic()
    if _list_cache is not None and (now - _list_cache[0]) < _CACHE_TTL_SECONDS:
        return _list_cache[1]

    resp = requests.get(
        TFR_LIST_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=15
    )
    resp.raise_for_status()
    data = resp.json()
    tfrs = data if isinstance(data, list) else []
    _list_cache = (now, tfrs)
    return tfrs


def fetch_tfr_geojson() -> list[dict]:
    """Fetch TFR polygon features from FAA GeoServer WFS with a 5-minute cache."""
    global _geo_cache
    now = time.monotonic()
    if _geo_cache is not None and (now - _geo_cache[0]) < _CACHE_TTL_SECONDS:
        return _geo_cache[1]

    resp = requests.get(
        TFR_WFS_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=20
    )
    resp.raise_for_status()
    features = resp.json().get("features", [])
    _geo_cache = (now, features)
    return features


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in statute miles between two WGS84 points."""
    r_miles = 3958.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r_miles * math.asin(math.sqrt(a))


def point_in_ring(lat: float, lon: float, ring: list[list[float]]) -> bool:
    """Ray-casting check for ``(lat, lon)`` inside a GeoJSON ``[lon, lat]`` ring."""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if ((yi > lat) != (yj > lat)) and (
            lon < (xj - xi) * (lat - yi) / ((yj - yi) or 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def _feature_distance(lat: float, lon: float, geom: dict) -> tuple[bool, float]:
    """Return ``(is_inside, min_distance_miles)`` for a Polygon or MultiPolygon."""
    gtype = geom.get("type", "")
    coords = geom.get("coordinates", [])
    polygons = [coords] if gtype == "Polygon" else coords if gtype == "MultiPolygon" else []

    min_dist = float("inf")
    for poly in polygons:
        if not poly:
            continue
        outer = poly[0]
        if point_in_ring(lat, lon, outer):
            return True, 0.0
        for pt in outer:
            d = haversine_miles(lat, lon, pt[1], pt[0])
            if d < min_dist:
                min_dist = d
    return False, min_dist


def check_tfrs_near_point(
    lat: float, lon: float, radius_miles: float = 30.0
) -> str:
    """Check TFR GeoJSON polygons inside or within ``radius_miles`` of ``(lat, lon)``."""
    try:
        features = fetch_tfr_geojson()
    except requests.RequestException as exc:
        return f"Could not reach FAA TFR GeoJSON feed ({exc}). Check https://tfr.faa.gov/."

    nearby: list[tuple[bool, float, dict]] = []
    for feat in features:
        geom = feat.get("geometry") or {}
        inside, dist = _feature_distance(lat, lon, geom)
        if inside or dist <= radius_miles:
            nearby.append((inside, dist, feat.get("properties") or {}))

    nearby.sort(key=lambda item: (not item[0], item[1]))
    if not nearby:
        return (
            f"No active TFR polygons found within {radius_miles:.0f} miles of "
            f"({lat:.4f}, {lon:.4f}) across {len(features)} mapped TFRs. "
            "Confirm official NOTAMs & LAANC before flight."
        )

    lines = [
        f"Found {len(nearby)} TFR(s) within {radius_miles:.0f} miles of ({lat:.4f}, {lon:.4f}):"
    ]
    for inside, dist, props in nearby[:10]:
        notam = props.get("NOTAM_KEY", "?")
        status = "INSIDE TFR" if inside else f"{dist:.1f} mi away"
        lines.append(
            f"- [{status}] NOTAM {notam} | {props.get('LEGAL', '')} | "
            f"{props.get('TITLE', '')}"
        )
    lines.append("Note: tfr.faa.gov is informational; confirm official NOTAMs/LAANC before flight.")
    return "\n".join(lines)


def check_active_tfrs(
    state: str = "",
    keyword: str = "",
    lat: float | None = None,
    lon: float | None = None,
    radius_miles: float = 30.0,
) -> str:
    """Check live FAA TFRs by GPS coordinates (preferred) or by state/keyword."""
    if lat is not None and lon is not None:
        return check_tfrs_near_point(lat=lat, lon=lon, radius_miles=radius_miles)

    try:
        tfrs = fetch_tfr_list()
    except requests.RequestException as exc:
        return f"Could not reach tfr.faa.gov ({exc}). Verify NOTAMs at https://tfr.faa.gov/."

    state_clean = state.strip().upper()
    kw_clean = keyword.strip().lower()

    matches = []
    for item in tfrs:
        item_state = str(item.get("state", "")).upper()
        desc = str(item.get("description", ""))
        tfr_type = str(item.get("type", ""))
        if state_clean and item_state != state_clean:
            continue
        if kw_clean and kw_clean not in desc.lower() and kw_clean not in tfr_type.lower():
            continue
        matches.append(item)

    if not matches:
        scope = f"state={state_clean or 'ALL'}" + (f", keyword={keyword}" if kw_clean else "")
        return (
            f"No active or scheduled TFRs matched ({scope}) on tfr.faa.gov "
            f"(checked {len(tfrs)} total TFRs). Always confirm official NOTAMs/LAANC before flight."
        )

    lines = [f"Found {len(matches)} TFR(s) on tfr.faa.gov (total active/scheduled: {len(tfrs)}):"]
    for m in matches[:10]:
        notam = m.get("notam_id", "?")
        detail_id = str(notam).replace("/", "_")
        url = f"https://tfr.faa.gov/tfr3/?page=detail_{detail_id}"
        lines.append(
            f"- NOTAM {notam} | {m.get('type', '')} | {m.get('state', '')} "
            f"({m.get('facility', '')}) | {m.get('description', '')} | {url}"
        )
    if len(matches) > 10:
        lines.append(f"...and {len(matches) - 10} more at https://tfr.faa.gov/")
    lines.append("Note: tfr.faa.gov is informational; confirm official NOTAMs/LAANC before flight.")
    return "\n".join(lines)
