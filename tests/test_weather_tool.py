"""Unit tests for `meeko.tools.weather`.

Stubs out `httpx.AsyncClient` so the suite stays offline; the WeatherClient
itself is exercised end-to-end (forecast fetch + format).
"""

from typing import Any

import httpx
import pytest

from meeko.tools import weather as weather_mod
from meeko.tools.weather import (
    WeatherClient,
    get_tool_definitions,
    handle,
)


def test_tool_definition_shape():
    defs = get_tool_definitions()
    assert len(defs) == 1
    d = defs[0]
    assert d["name"] == "get_weather"
    assert "description" in d and isinstance(d["description"], str)
    schema = d["input_schema"]
    assert schema["type"] == "object"
    props = schema["properties"]
    assert set(props) == {"latitude", "longitude", "place_label"}
    # All three args are optional individually; the handler enforces
    # all-or-nothing at runtime.
    assert not schema.get("required")


class _StubResponse:
    def __init__(self, payload: dict | None = None, status_code: int = 200) -> None:
        self._payload = payload or {}
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "boom",
                request=None,
                response=None,  # type: ignore[arg-type]
            )

    def json(self) -> dict:
        return self._payload


class _StubClient:
    """Mimics `httpx.AsyncClient` as an async context manager.

    `responder` maps URL → list of payloads (consumed in order on each
    matching call); an `Exception` payload is raised instead of returned.
    """

    def __init__(self, responder: dict[str, list[Any]]):
        self._responder = responder
        self.calls: list[tuple[str, dict]] = []

    async def __aenter__(self) -> _StubClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def get(self, url: str, params: dict | None = None) -> _StubResponse:
        self.calls.append((url, dict(params or {})))
        queue = self._responder.get(url)
        if not queue:
            raise AssertionError(f"Unexpected GET {url}")
        payload = queue.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return _StubResponse(payload)


_FORECAST = "https://api.open-meteo.com/v1/forecast"


def _forecast_payload() -> dict:
    return {
        "current": {
            "temperature_2m": 54.3,
            "weather_code": 2,
            "wind_speed_10m": 8.1,
            "wind_direction_10m": 315,  # NW
        },
        "daily": {
            "temperature_2m_max": [61.0, 58.0],
            "temperature_2m_min": [48.0, 49.0],
            "precipitation_probability_max": [10, 80],
            "weather_code": [2, 61],
        },
    }


def _install_stub_client(monkeypatch, stub: _StubClient) -> None:
    monkeypatch.setattr(
        weather_mod.httpx,
        "AsyncClient",
        lambda *a, **kw: stub,
    )


_HOME_LAT = 40.7484
_HOME_LON = -73.9857


def _configure_home(client: WeatherClient, units: str = "imperial") -> None:
    client.configure(latitude=_HOME_LAT, longitude=_HOME_LON, units=units)


def _configure_no_home(client: WeatherClient, units: str = "imperial") -> None:
    client.configure(latitude=None, longitude=None, units=units)


@pytest.mark.asyncio
async def test_handle_uses_home_coords_when_args_omitted(monkeypatch):
    stub = _StubClient({_FORECAST: [_forecast_payload()]})
    _install_stub_client(monkeypatch, stub)

    client = WeatherClient()
    _configure_home(client)
    weather_mod.weather_client = client

    result = await handle("get_weather", {})

    forecast_call = next(c for c in stub.calls if c[0] == _FORECAST)
    assert forecast_call[1]["latitude"] == _HOME_LAT
    assert forecast_call[1]["longitude"] == _HOME_LON

    assert "Home:" in result
    assert "54°F" in result
    assert "partly cloudy" in result
    assert "8 mph" in result and "from NW" in result
    assert "Today" in result and "Tomorrow" in result
    assert "10%" in result
    assert "light rain" in result


@pytest.mark.asyncio
async def test_handle_uses_supplied_coords_when_provided(monkeypatch):
    stub = _StubClient({_FORECAST: [_forecast_payload()]})
    _install_stub_client(monkeypatch, stub)

    client = WeatherClient()
    _configure_home(client)
    weather_mod.weather_client = client

    result = await handle(
        "get_weather",
        {
            "latitude": 43.6591,
            "longitude": -70.2568,
            "place_label": "Portland, Maine",
        },
    )

    # Forecast must use the supplied coords, not the home coords.
    forecast_call = next(c for c in stub.calls if c[0] == _FORECAST)
    assert forecast_call[1]["latitude"] == 43.6591
    assert forecast_call[1]["longitude"] == -70.2568
    assert "Portland, Maine" in result


@pytest.mark.asyncio
async def test_handle_partial_coord_args_returns_correction(monkeypatch):
    # No HTTP stub — should short-circuit before any forecast call.
    client = WeatherClient()
    _configure_home(client)
    weather_mod.weather_client = client

    result = await handle("get_weather", {"latitude": 43.0})
    assert "supply latitude, longitude" in result.lower()


@pytest.mark.asyncio
async def test_handle_out_of_range_coords_returns_friendly_message(monkeypatch):
    client = WeatherClient()
    _configure_home(client)
    weather_mod.weather_client = client

    result = await handle(
        "get_weather",
        {"latitude": 999.0, "longitude": 0.0, "place_label": "Nowhere"},
    )
    assert "out of range" in result.lower()


@pytest.mark.asyncio
async def test_handle_http_error_returns_friendly_message(monkeypatch):
    stub = _StubClient({_FORECAST: [httpx.ConnectError("nope")]})
    _install_stub_client(monkeypatch, stub)

    client = WeatherClient()
    _configure_home(client)
    weather_mod.weather_client = client

    result = await handle("get_weather", {})
    assert "couldn't reach" in result.lower()


@pytest.mark.asyncio
async def test_handle_no_home_and_no_arg_returns_config_message(monkeypatch):
    client = WeatherClient()
    _configure_no_home(client)
    weather_mod.weather_client = client

    result = await handle("get_weather", {})
    assert "no home location" in result.lower()


@pytest.mark.asyncio
async def test_metric_units_change_forecast_params_and_symbols(monkeypatch):
    stub = _StubClient({_FORECAST: [_forecast_payload()]})
    _install_stub_client(monkeypatch, stub)

    client = WeatherClient()
    _configure_home(client, units="metric")
    weather_mod.weather_client = client

    result = await handle("get_weather", {})

    forecast_call = next(c for c in stub.calls if c[0] == _FORECAST)
    assert forecast_call[1]["temperature_unit"] == "celsius"
    assert forecast_call[1]["wind_speed_unit"] == "kmh"
    assert "°C" in result and "km/h" in result
    assert "°F" not in result
