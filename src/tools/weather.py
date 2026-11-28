"""Live aviation weather (METAR) lookup via AviationWeather.gov."""

import requests

from src.tools.tfr import haversine_miles

METAR_URL = "https://aviationweather.gov/api/data/metar"


def _format_metar(m: dict, dist_mi: float | None = None) -> str:
    icao = m.get("icaoId", "?")
    name = m.get("name", "")
    visib = m.get("visib", "?")
    wdir = m.get("wdir", "?")
    wspd = m.get("wspd", "?")
    wgst = m.get("wgst")
    flt_cat = m.get("fltCat", "?")
    raw = m.get("rawOb", "")
    clouds = m.get("clouds") or []
    cloud_str = (
        ", ".join(f"{c.get('cover', '')} {c.get('base', '')}ft" for c in clouds)
        or "Clear/None reported"
    )
    gust_str = f" gusting {wgst} kt" if wgst else ""
    dist_str = f" ({dist_mi:.1f} mi away)" if dist_mi is not None else ""
    return (
        f"- {icao} ({name}){dist_str} | Category: {flt_cat} | Visibility: {visib} sm | "
        f"Wind: {wdir}° at {wspd} kt{gust_str} | Clouds: {cloud_str}\n  Raw: {raw}"
    )


def check_aviation_weather(
    station_id: str = "", lat: float | None = None, lon: float | None = None
) -> str:
    """Fetch current METAR weather by 4-letter ICAO code or nearest to ``(lat, lon)``."""
    params: dict[str, str] = {"format": "json"}
    if station_id.strip():
        params["ids"] = station_id.strip().upper()
    elif lat is not None and lon is not None:
        params["bbox"] = f"{lat - 0.5:.3f},{lon - 0.5:.3f},{lat + 0.5:.3f},{lon + 0.5:.3f}"
    else:
        return "Provide a 4-letter airport ICAO code (e.g., KBOS) or GPS coordinates."

    try:
        resp = requests.get(
            METAR_URL, params=params, headers={"User-Agent": "Part107FieldOps/1.0"}, timeout=30
        )
        if resp.status_code == 204 or not resp.text.strip():
            return "No recent METAR stations found for that location."
        resp.raise_for_status()
        stations = resp.json()
    except requests.RequestException as exc:
        return f"Could not reach aviationweather.gov ({exc}). Check local METARs manually."

    if not stations:
        return "No METAR data returned for that location."

    if lat is not None and lon is not None and not station_id.strip():
        with_dist = []
        for s in stations:
            slat, slon = s.get("lat"), s.get("lon")
            d = haversine_miles(lat, lon, float(slat), float(slon)) if slat and slon else 999.0
            with_dist.append((d, s))
        with_dist.sort(key=lambda x: x[0])
        lines = ["Nearest live METAR weather reports (compare vs. 14 CFR § 107.51(c)-(d)):"]
        for d, s in with_dist[:3]:
            lines.append(_format_metar(s, dist_mi=d))
        return "\n".join(lines)

    lines = ["Live METAR weather report (compare vs. 14 CFR § 107.51(c)-(d)):"]
    for s in stations[:3]:
        lines.append(_format_metar(s))
    return "\n".join(lines)
