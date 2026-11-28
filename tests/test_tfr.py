"""Offline unit tests for the live FAA TFR tool and geospatial checks."""

from src.tools import tfr


def test_check_active_tfrs_filters_state_and_keyword(monkeypatch) -> None:
    sample = [
        {
            "notam_id": "6/9880",
            "type": "SECURITY",
            "facility": "ZHU",
            "state": "TX",
            "description": "Killeen, TX, Nov 30-Dec 4",
        },
        {
            "notam_id": "6/1111",
            "type": "HAZARDS",
            "facility": "ZBW",
            "state": "MA",
            "description": "Boston, MA temporary hazard",
        },
    ]
    monkeypatch.setattr(tfr, "fetch_tfr_list", lambda: sample)

    ma_out = tfr.check_active_tfrs(state="ma")
    assert "6/1111" in ma_out
    assert "6/9880" not in ma_out

    none_out = tfr.check_active_tfrs(state="CA")
    assert "No active or scheduled TFRs matched" in none_out


def test_check_tfrs_near_point_inside_polygon(monkeypatch) -> None:
    features = [
        {
            "properties": {
                "NOTAM_KEY": "5/6914-1-FDC-F",
                "LEGAL": "SECURITY",
                "TITLE": "South Padre Island, TX",
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[-97.2, 26.0], [-97.1, 26.0], [-97.1, 26.2], [-97.2, 26.2], [-97.2, 26.0]]
                ],
            },
        }
    ]
    monkeypatch.setattr(tfr, "fetch_tfr_geojson", lambda: features)

    inside_out = tfr.check_active_tfrs(lat=26.11, lon=-97.15)
    assert "INSIDE TFR" in inside_out
    assert "5/6914-1-FDC-F" in inside_out

    far_out = tfr.check_active_tfrs(lat=42.0, lon=-70.6, radius_miles=30.0)
    assert "No active TFR polygons found" in far_out
