"""Unit tests for `meeko.tools.weather`.

Stubs out `httpx.AsyncClient` so the suite stays offline; the WeatherClient
itself is exercised end-to-end (forecast fetch + format).
"""

from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx
import pytest

from meeko.tools import weather as weather_mod
from meeko.tools.weather import (
    _HOURLY_HOURS,
    WeatherClient,
    _now_hour_local,
    _now_local,
    _precip_window,
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
    assert set(props) == {"latitude", "longitude", "place_label", "date", "hourly"}
    # All args are optional individually; the handler enforces the
    # all-or-nothing place rule and date validation at runtime.
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


def _today_iso() -> str:
    return datetime.now().astimezone().date().isoformat()


def _hourly_times(day_iso: str) -> list[str]:
    return [f"{day_iso}T{h:02d}:00" for h in range(24)]


def _forecast_payload(
    *,
    day_iso: str | None = None,
    prob_max: int = 80,
    rain_hours: tuple[int, ...] = (14, 15, 16),
) -> dict:
    """One-day payload (start_date == end_date), so every array has length 1
    for `daily` and 24 for `hourly`."""
    day_iso = day_iso or _today_iso()
    probs = [90 if h in rain_hours else 0 for h in range(24)]
    return {
        "utc_offset_seconds": 0,
        "current": {
            "temperature_2m": 54.3,
            "relative_humidity_2m": 47.0,
            "weather_code": 2,
        },
        "daily": {
            "time": [day_iso],
            "temperature_2m_max": [61.0],
            "temperature_2m_min": [48.0],
            "precipitation_probability_max": [prob_max],
            "weather_code": [61],
        },
        "hourly": {
            "time": _hourly_times(day_iso),
            "precipitation_probability": probs,
            "precipitation": [0.1 if h in rain_hours else 0.0 for h in range(24)],
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
async def test_handle_uses_home_coords_and_bounds_to_today(monkeypatch):
    # No rain in the immediate hours so the timing clause doesn't depend on
    # the wall clock; we assert it separately below.
    stub = _StubClient({_FORECAST: [_forecast_payload(rain_hours=())]})
    _install_stub_client(monkeypatch, stub)

    client = WeatherClient()
    _configure_home(client)
    weather_mod.weather_client = client

    result = await handle("get_weather", {})

    forecast_call = next(c for c in stub.calls if c[0] == _FORECAST)
    assert forecast_call[1]["latitude"] == _HOME_LAT
    assert forecast_call[1]["longitude"] == _HOME_LON
    # Bounded to a single day.
    today = _today_iso()
    assert forecast_call[1]["start_date"] == today
    assert forecast_call[1]["end_date"] == today

    assert "Home, today" in result
    assert today in result
    # Today shows current temp + humidity and the live weather code (partly
    # cloudy = code 2).
    assert "54°F" in result
    assert "humidity 47%" in result
    assert "partly cloudy" in result
    assert "High 61°F, low 48°F" in result
    assert "80% chance of precipitation" in result
    # Wind is gone.
    assert "mph" not in result and "from" not in result.lower()


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

    forecast_call = next(c for c in stub.calls if c[0] == _FORECAST)
    assert forecast_call[1]["latitude"] == 43.6591
    assert forecast_call[1]["longitude"] == -70.2568
    assert "Portland, Maine" in result


@pytest.mark.asyncio
async def test_future_day_selection_and_no_current_conditions(monkeypatch):
    three_out = (datetime.now().astimezone().date() + timedelta(days=3)).isoformat()
    stub = _StubClient({_FORECAST: [_forecast_payload(day_iso=three_out)]})
    _install_stub_client(monkeypatch, stub)

    client = WeatherClient()
    _configure_home(client)
    weather_mod.weather_client = client

    result = await handle("get_weather", {"date": three_out})

    forecast_call = next(c for c in stub.calls if c[0] == _FORECAST)
    assert forecast_call[1]["start_date"] == three_out
    assert forecast_call[1]["end_date"] == three_out

    # Future day: weekday label, daily weather code (rain=61), high/low,
    # precip — but NO current temp / humidity.
    assert three_out in result
    assert "light rain" in result
    assert "High 61°F, low 48°F" in result
    assert "humidity" not in result
    assert "Currently" not in result


def test_precip_window_filters_by_supplied_now_hour():
    # 24 hourly rows, rain 14:00–16:00. With no cutoff the window renders;
    # a cutoff past those hours filters them out entirely — independent of
    # the machine's wall clock.
    wet = (14, 15, 16)
    hourly = {
        "time": _hourly_times("2026-06-02"),
        "precipitation_probability": [90 if h in wet else 0 for h in range(24)],
        "precipitation": [0.1 if h in wet else 0.0 for h in range(24)],
    }
    assert _precip_window(hourly, now_hour=None) == "around 2–4 PM"
    assert _precip_window(hourly, now_hour=12) == "around 2–4 PM"
    # Current hour is kept (14 is not < 14); 17 onward drops the window.
    assert _precip_window(hourly, now_hour=14) == "around 2–4 PM"
    assert _precip_window(hourly, now_hour=17) == ""


def test_now_hour_local_uses_api_offset_not_server_tz():
    # Future day → no cutoff.
    assert _now_hour_local({"utc_offset_seconds": 3600}, is_today=False) is None
    # Offset chosen so the target-local time is exactly midnight → hour 0,
    # regardless of where (or when) this test runs.
    now = datetime.now(UTC)
    secs_since_utc_midnight = now.hour * 3600 + now.minute * 60 + now.second
    payload = {"utc_offset_seconds": -secs_since_utc_midnight}
    assert _now_hour_local(payload, is_today=True) == 0
    # Missing offset falls back to a sane 0–23 hour rather than raising.
    fallback = _now_hour_local({}, is_today=True)
    assert isinstance(fallback, int) and 0 <= fallback <= 23


@pytest.mark.asyncio
async def test_precip_timing_window_rendered(monkeypatch):
    # A future day so "now" filtering doesn't trim the morning window.
    day = (datetime.now().astimezone().date() + timedelta(days=2)).isoformat()
    stub = _StubClient(
        {_FORECAST: [_forecast_payload(day_iso=day, rain_hours=(14, 15, 16))]}
    )
    _install_stub_client(monkeypatch, stub)

    client = WeatherClient()
    _configure_home(client)
    weather_mod.weather_client = client

    result = await handle("get_weather", {"date": day})

    assert "precipitation likely around 2–4 PM" in result


@pytest.mark.asyncio
async def test_low_precip_chance_omits_timing(monkeypatch):
    day = (datetime.now().astimezone().date() + timedelta(days=2)).isoformat()
    stub = _StubClient(
        {_FORECAST: [_forecast_payload(day_iso=day, prob_max=10, rain_hours=())]}
    )
    _install_stub_client(monkeypatch, stub)

    client = WeatherClient()
    _configure_home(client)
    weather_mod.weather_client = client

    result = await handle("get_weather", {"date": day})

    assert "10% chance of precipitation" in result
    assert "precipitation likely" not in result


@pytest.mark.asyncio
async def test_out_of_range_date_returns_message_without_fetch(monkeypatch):
    # No stub installed: if get_weather tried to fetch, the missing
    # AsyncClient would surface. It must short-circuit before that.
    far = (datetime.now().astimezone().date() + timedelta(days=30)).isoformat()

    client = WeatherClient()
    _configure_home(client)
    weather_mod.weather_client = client

    result = await handle("get_weather", {"date": far})
    assert "two weeks" in result.lower()


@pytest.mark.asyncio
async def test_past_date_returns_message_without_fetch(monkeypatch):
    past = (datetime.now().astimezone().date() - timedelta(days=1)).isoformat()

    client = WeatherClient()
    _configure_home(client)
    weather_mod.weather_client = client

    result = await handle("get_weather", {"date": past})
    assert "look ahead" in result.lower()


@pytest.mark.asyncio
async def test_unparseable_date_returns_message_without_fetch(monkeypatch):
    client = WeatherClient()
    _configure_home(client)
    weather_mod.weather_client = client

    result = await handle("get_weather", {"date": "next thursday"})
    assert "date" in result.lower()


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
    stub = _StubClient({_FORECAST: [_forecast_payload(rain_hours=())]})
    _install_stub_client(monkeypatch, stub)

    client = WeatherClient()
    _configure_home(client, units="metric")
    weather_mod.weather_client = client

    result = await handle("get_weather", {})

    forecast_call = next(c for c in stub.calls if c[0] == _FORECAST)
    assert forecast_call[1]["temperature_unit"] == "celsius"
    assert forecast_call[1]["precipitation_unit"] == "mm"
    assert "°C" in result
    assert "°F" not in result


# --- Hourly mode -------------------------------------------------------


def _midnight_offset() -> tuple[int, date]:
    """A `utc_offset_seconds` that puts the forecast location at exactly
    midnight, plus the local date that implies. Lets the hourly tests assert an
    exact first row regardless of where (or when) they run."""
    now = datetime.now(UTC)
    secs_since_utc_midnight = now.hour * 3600 + now.minute * 60 + now.second
    return -secs_since_utc_midnight, now.replace(tzinfo=None).date()


def _hourly_payload(
    *,
    start_date: date,
    days: int = 4,
    offset_seconds: int = 0,
    temp: float = 55.4,
    code: int = 61,
    prob: int | None = 40,
) -> dict:
    """A multi-day hourly payload shaped like Open-Meteo's, starting at 00:00
    local on `start_date`."""
    times: list[str] = []
    for d in range(days):
        day_iso = (start_date + timedelta(days=d)).isoformat()
        times.extend(_hourly_times(day_iso))
    hourly: dict[str, Any] = {
        "time": times,
        "temperature_2m": [temp] * len(times),
        "weather_code": [code] * len(times),
    }
    if prob is not None:
        hourly["precipitation_probability"] = [prob] * len(times)
    return {
        "utc_offset_seconds": offset_seconds,
        "current": {
            "temperature_2m": 54.3,
            "relative_humidity_2m": 47.0,
            "weather_code": 2,
        },
        "hourly": hourly,
    }


def _hourly_rows(result: str) -> list[str]:
    return result.split("\n")[1:]


@pytest.mark.asyncio
async def test_hourly_returns_48_rows_from_the_current_hour(monkeypatch):
    offset, local_today = _midnight_offset()
    payload = _hourly_payload(
        start_date=local_today - timedelta(days=1), offset_seconds=offset
    )
    stub = _StubClient({_FORECAST: [payload]})
    _install_stub_client(monkeypatch, stub)

    client = WeatherClient()
    _configure_home(client)
    weather_mod.weather_client = client

    result = await handle("get_weather", {"hourly": True})

    call = next(c for c in stub.calls if c[0] == _FORECAST)
    assert call[1]["latitude"] == _HOME_LAT
    assert "temperature_2m" in call[1]["hourly"]
    assert "weather_code" in call[1]["hourly"]
    # Hourly mode has no use for the daily block.
    assert "daily" not in call[1]
    today = datetime.now().astimezone().date()
    assert call[1]["start_date"] == (today - timedelta(days=1)).isoformat()
    assert call[1]["end_date"] == (today + timedelta(days=2)).isoformat()

    assert result.startswith("Home, next 48 hours. Currently 54°F, partly cloudy:")
    rows = _hourly_rows(result)
    assert len(rows) == _HOURLY_HOURS
    # Location-local midnight → the first row is that hour, on today's date.
    assert rows[0] == f"{local_today.strftime('%a')} 12 AM  55°F  light rain  40%"


@pytest.mark.asyncio
async def test_hourly_rows_cross_midnight_with_day_labels(monkeypatch):
    offset, local_today = _midnight_offset()
    payload = _hourly_payload(
        start_date=local_today - timedelta(days=1), offset_seconds=offset
    )
    stub = _StubClient({_FORECAST: [payload]})
    _install_stub_client(monkeypatch, stub)

    client = WeatherClient()
    _configure_home(client)
    weather_mod.weather_client = client

    rows = _hourly_rows(await handle("get_weather", {"hourly": True}))

    tomorrow = local_today + timedelta(days=1)
    # Rows 0–23 are today, 24–47 tomorrow — starting from local midnight.
    assert rows[23].startswith(f"{local_today.strftime('%a')} 11 PM")
    assert rows[24].startswith(f"{tomorrow.strftime('%a')} 12 AM")
    assert rows[47].startswith(f"{tomorrow.strftime('%a')} 11 PM")


@pytest.mark.asyncio
async def test_hourly_ignores_date_argument(monkeypatch):
    offset, local_today = _midnight_offset()
    payload = _hourly_payload(
        start_date=local_today - timedelta(days=1), offset_seconds=offset
    )
    stub = _StubClient({_FORECAST: [payload]})
    _install_stub_client(monkeypatch, stub)

    client = WeatherClient()
    _configure_home(client)
    weather_mod.weather_client = client

    # A date far outside the daily path's 14-day window must not short-circuit
    # the hourly request.
    far = (datetime.now().astimezone().date() + timedelta(days=30)).isoformat()
    result = await handle("get_weather", {"hourly": True, "date": far})

    assert "two weeks" not in result
    assert len(_hourly_rows(result)) == _HOURLY_HOURS


@pytest.mark.asyncio
async def test_hourly_uses_supplied_coords_and_label(monkeypatch):
    offset, local_today = _midnight_offset()
    payload = _hourly_payload(
        start_date=local_today - timedelta(days=1), offset_seconds=offset
    )
    stub = _StubClient({_FORECAST: [payload]})
    _install_stub_client(monkeypatch, stub)

    client = WeatherClient()
    _configure_home(client)
    weather_mod.weather_client = client

    result = await handle(
        "get_weather",
        {
            "hourly": True,
            "latitude": 43.6591,
            "longitude": -70.2568,
            "place_label": "Portland, Maine",
        },
    )

    call = next(c for c in stub.calls if c[0] == _FORECAST)
    assert call[1]["latitude"] == 43.6591
    assert result.startswith("Portland, Maine, next 48 hours")


@pytest.mark.asyncio
async def test_hourly_omits_probability_when_absent(monkeypatch):
    offset, local_today = _midnight_offset()
    payload = _hourly_payload(
        start_date=local_today - timedelta(days=1),
        offset_seconds=offset,
        prob=None,
    )
    stub = _StubClient({_FORECAST: [payload]})
    _install_stub_client(monkeypatch, stub)

    client = WeatherClient()
    _configure_home(client)
    weather_mod.weather_client = client

    rows = _hourly_rows(await handle("get_weather", {"hourly": True}))
    assert rows[0].endswith("light rain")
    assert "%" not in rows[0]


@pytest.mark.asyncio
async def test_hourly_empty_block_returns_friendly_message(monkeypatch):
    stub = _StubClient({_FORECAST: [{"utc_offset_seconds": 0, "hourly": {}}]})
    _install_stub_client(monkeypatch, stub)

    client = WeatherClient()
    _configure_home(client)
    weather_mod.weather_client = client

    result = await handle("get_weather", {"hourly": True})
    assert "couldn't get an hourly forecast" in result


@pytest.mark.asyncio
async def test_hourly_http_error_returns_friendly_message(monkeypatch):
    stub = _StubClient({_FORECAST: [httpx.ConnectError("nope")]})
    _install_stub_client(monkeypatch, stub)

    client = WeatherClient()
    _configure_home(client)
    weather_mod.weather_client = client

    result = await handle("get_weather", {"hourly": True})
    assert "couldn't reach" in result.lower()


@pytest.mark.asyncio
async def test_hourly_metric_units(monkeypatch):
    offset, local_today = _midnight_offset()
    payload = _hourly_payload(
        start_date=local_today - timedelta(days=1), offset_seconds=offset
    )
    stub = _StubClient({_FORECAST: [payload]})
    _install_stub_client(monkeypatch, stub)

    client = WeatherClient()
    _configure_home(client, units="metric")
    weather_mod.weather_client = client

    result = await handle("get_weather", {"hourly": True})

    call = next(c for c in stub.calls if c[0] == _FORECAST)
    assert call[1]["temperature_unit"] == "celsius"
    assert "°C" in result
    assert "°F" not in result


def test_now_local_uses_api_offset_not_server_tz():
    offset, local_today = _midnight_offset()
    local = _now_local({"utc_offset_seconds": offset})
    assert local.tzinfo is None
    assert local.date() == local_today
    assert local.hour == 0
    # Missing offset falls back to this machine's local time rather than raising.
    assert _now_local({}).tzinfo is None
