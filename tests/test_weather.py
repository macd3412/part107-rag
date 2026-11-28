"""Offline unit tests for live aviation weather formatting."""

from src.tools.weather import check_aviation_weather


def test_check_aviation_weather_by_station(monkeypatch) -> None:
    class FakeResp:
        status_code = 200
        text = "[{}]"

        def raise_for_status(self) -> None:
            return None

        def json(self) -> list[dict]:
            return [
                {
                    "icaoId": "KBOS",
                    "name": "Boston Logan",
                    "visib": "10+",
                    "wdir": 270,
                    "wspd": 12,
                    "fltCat": "VFR",
                    "clouds": [{"cover": "FEW", "base": 4500}],
                    "rawOb": "KBOS 281454Z 27012KT 10SM FEW045",
                }
            ]

    monkeypatch.setattr("src.tools.weather.requests.get", lambda *a, **kw: FakeResp())
    out = check_aviation_weather(station_id="kbos")
    assert "KBOS" in out
    assert "10+ sm" in out
    assert "VFR" in out
