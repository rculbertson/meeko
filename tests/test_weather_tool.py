"""Unit tests for `meeko.tools.weather`.

Stubs out `httpx.AsyncClient` so the suite stays offline; the WeatherClient
itself is exercised end-to-end (forecast fetch + format).
"""

from datetime import UTC, date, datetime, time, timedelta
from typing import Any

import httpx
import pytest

from meeko.tools import weather as weather_mod
from meeko.tools.weather import (
    _HOURLY_HOURS,
    WeatherClient,
    _FriendlyError,
    _hourly_series,
    _now_hour_local,
    _now_local,
    _precip_window,
    _resolve_date,
    _wet_runs,
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
    matching call); an `Exception` payload is raised instead of returned, and
    a callable one is invoked with the request params to build the payload
    (so a test can return exactly the window the code asked for).
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
        if callable(payload):
            payload = payload(dict(params or {}))
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


async def _get_weather(
    monkeypatch,
    payload: Any = None,
    args: dict | None = None,
    *,
    home: bool = True,
    units: str = "imperial",
) -> tuple[str, dict | None]:
    """Run the `get_weather` tool once and return `(result, forecast_params)`.

    `payload` is one `_StubClient` response (dict, exception, or callable).
    With `payload=None` no stub is installed, for the paths that must
    short-circuit before any fetch; `forecast_params` is then None.
    """
    stub = None
    if payload is not None:
        stub = _StubClient({_FORECAST: [payload]})
        _install_stub_client(monkeypatch, stub)

    client = WeatherClient()
    client.configure(
        latitude=_HOME_LAT if home else None,
        longitude=_HOME_LON if home else None,
        units=units,
    )
    weather_mod.weather_client = client

    result = await handle("get_weather", args or {})
    params = stub.calls[0][1] if stub and stub.calls else None
    return result, params


def _days_out_iso(days: int) -> str:
    return (datetime.now().astimezone().date() + timedelta(days=days)).isoformat()


@pytest.mark.asyncio
async def test_handle_uses_home_coords_and_bounds_to_today(monkeypatch):
    # No rain in the immediate hours so the timing clause doesn't depend on
    # the wall clock; we assert it separately below.
    result, params = await _get_weather(monkeypatch, _forecast_payload(rain_hours=()))

    # Home coords, bounded to a single day.
    today = _today_iso()
    expected_params = {
        "latitude": _HOME_LAT,
        "longitude": _HOME_LON,
        "start_date": today,
        "end_date": today,
    }
    assert expected_params.items() <= params.items()

    # Today shows current temp + humidity and the live weather code (partly
    # cloudy = code 2).
    for expected in (
        "Home, today",
        today,
        "54°F",
        "humidity 47%",
        "partly cloudy",
        "High 61°F, low 48°F",
        "80% chance of precipitation",
    ):
        assert expected in result
    # Wind is gone.
    assert "mph" not in result
    assert "from" not in result.lower()


@pytest.mark.asyncio
async def test_handle_uses_supplied_coords_when_provided(monkeypatch):
    result, params = await _get_weather(
        monkeypatch,
        _forecast_payload(),
        {
            "latitude": 43.6591,
            "longitude": -70.2568,
            "place_label": "Portland, Maine",
        },
    )

    assert params["latitude"] == 43.6591
    assert params["longitude"] == -70.2568
    assert "Portland, Maine" in result


@pytest.mark.asyncio
async def test_future_day_selection_and_no_current_conditions(monkeypatch):
    three_out = _days_out_iso(3)
    result, params = await _get_weather(
        monkeypatch, _forecast_payload(day_iso=three_out), {"date": three_out}
    )

    assert params["start_date"] == three_out
    assert params["end_date"] == three_out

    # Future day: weekday label, daily weather code (rain=61), high/low,
    # precip — but NO current temp / humidity.
    assert three_out in result
    assert "light rain" in result
    assert "High 61°F, low 48°F" in result
    assert "humidity" not in result
    assert "Currently" not in result


def test_precip_window_filters_by_supplied_now_hour():
    # 24 hourly rows, rain 14:00-16:00. With no cutoff the window renders;
    # a cutoff past those hours filters them out entirely — independent of
    # the machine's wall clock.
    wet = (14, 15, 16)
    hourly = {
        "time": _hourly_times("2026-06-02"),
        "precipitation_probability": [90 if h in wet else 0 for h in range(24)],
        "precipitation": [0.1 if h in wet else 0.0 for h in range(24)],
    }
    assert _precip_window(hourly, now_hour=None) == "around 2-4 PM"
    assert _precip_window(hourly, now_hour=12) == "around 2-4 PM"
    # Current hour is kept (14 is not < 14); 17 onward drops the window.
    assert _precip_window(hourly, now_hour=14) == "around 2-4 PM"
    assert _precip_window(hourly, now_hour=17) == ""


def test_precip_window_falls_back_to_amount_when_probability_missing():
    # No probability for 9-10 AM, but a measurable amount → wet. Elsewhere a
    # missing probability with zero/missing amount stays dry.
    wet = (9, 10)
    hourly = {
        "time": _hourly_times("2026-06-02"),
        "precipitation_probability": [None if h in wet else 0 for h in range(24)],
        "precipitation": [0.2 if h in wet else 0.0 for h in range(24)],
    }
    assert _precip_window(hourly, now_hour=None) == "around 9-10 AM"


def test_precip_window_unparseable_timestamp_does_not_split_run():
    # A garbled 15:00 row is skipped, not treated as dry, so 14:00 and 16:00
    # still form one window.
    times = _hourly_times("2026-06-02")
    times[15] = "not-a-time"
    wet = (14, 15, 16)
    hourly = {
        "time": times,
        "precipitation_probability": [90 if h in wet else 0 for h in range(24)],
    }
    assert _precip_window(hourly, now_hour=None) == "around 2-4 PM"


def test_hourly_series_skips_bad_timestamps_and_pads_short_arrays():
    hourly = {
        "time": ["2026-06-02T00:00", None, "garbage", "2026-06-02T03:00"],
        "a": [1, 2, 3, 4],
        "b": [10],  # shorter than `time`
    }
    rows = list(_hourly_series(hourly, "a", "b", "missing"))
    assert rows == [
        (datetime(2026, 6, 2, 0), (1, 10, None)),
        (datetime(2026, 6, 2, 3), (4, None, None)),
    ]
    assert list(_hourly_series({}, "a")) == []


def test_wet_runs_groups_consecutive_wet_hours():
    hours = [datetime(2026, 6, 2, h) for h in range(6)]
    flags = [True, True, False, False, True, True]  # trailing run must flush
    runs = _wet_runs(zip(hours, flags, strict=True))
    assert runs == [hours[0:2], hours[4:6]]
    assert _wet_runs([]) == []
    assert _wet_runs([(hours[0], False)]) == []


def test_resolve_date_bounds():
    today = date(2026, 6, 2)
    assert _resolve_date(None, today) == today
    assert _resolve_date("2026-06-02", today) == today
    # today + 13 is the last forecastable day; +14 is refused.
    assert _resolve_date("2026-06-15", today) == date(2026, 6, 15)
    with pytest.raises(_FriendlyError, match="two weeks"):
        _resolve_date("2026-06-16", today)
    with pytest.raises(_FriendlyError, match="look ahead"):
        _resolve_date("2026-06-01", today)
    with pytest.raises(_FriendlyError, match="date"):
        _resolve_date("next thursday", today)


def test_now_hour_local_uses_api_offset_not_server_tz():
    # Future day → no cutoff.
    assert _now_hour_local({"utc_offset_seconds": 3600}, is_today=False) is None
    # Offset chosen so the target-local time is exactly midnight → hour 0,
    # regardless of where (or when) this test runs.
    now = datetime.now(UTC)
    secs_since_utc_midnight = now.hour * 3600 + now.minute * 60 + now.second
    payload = {"utc_offset_seconds": -secs_since_utc_midnight}
    assert _now_hour_local(payload, is_today=True) == 0
    # Missing offset falls back to a sane 0-23 hour rather than raising.
    fallback = _now_hour_local({}, is_today=True)
    assert isinstance(fallback, int) and 0 <= fallback <= 23


@pytest.mark.asyncio
async def test_precip_timing_window_rendered(monkeypatch):
    # A future day so "now" filtering doesn't trim the morning window.
    day = _days_out_iso(2)
    result, _ = await _get_weather(
        monkeypatch,
        _forecast_payload(day_iso=day, rain_hours=(14, 15, 16)),
        {"date": day},
    )

    assert "precipitation likely around 2-4 PM" in result


@pytest.mark.asyncio
async def test_low_precip_chance_omits_timing(monkeypatch):
    day = _days_out_iso(2)
    result, _ = await _get_weather(
        monkeypatch,
        _forecast_payload(day_iso=day, prob_max=10, rain_hours=()),
        {"date": day},
    )

    assert "10% chance of precipitation" in result
    assert "precipitation likely" not in result


@pytest.mark.asyncio
async def test_out_of_range_date_returns_message_without_fetch(monkeypatch):
    # No stub installed: if get_weather tried to fetch, the missing
    # AsyncClient would surface. It must short-circuit before that.
    result, _ = await _get_weather(monkeypatch, args={"date": _days_out_iso(30)})
    assert "two weeks" in result.lower()


@pytest.mark.asyncio
async def test_past_date_returns_message_without_fetch(monkeypatch):
    result, _ = await _get_weather(monkeypatch, args={"date": _days_out_iso(-1)})
    assert "look ahead" in result.lower()


@pytest.mark.asyncio
async def test_unparseable_date_returns_message_without_fetch(monkeypatch):
    result, _ = await _get_weather(monkeypatch, args={"date": "next thursday"})
    assert "date" in result.lower()


@pytest.mark.asyncio
async def test_handle_partial_coord_args_returns_correction(monkeypatch):
    # No HTTP stub — should short-circuit before any forecast call.
    result, _ = await _get_weather(monkeypatch, args={"latitude": 43.0})
    assert "supply latitude, longitude" in result.lower()


@pytest.mark.asyncio
async def test_handle_out_of_range_coords_returns_friendly_message(monkeypatch):
    result, _ = await _get_weather(
        monkeypatch,
        args={"latitude": 999.0, "longitude": 0.0, "place_label": "Nowhere"},
    )
    assert "out of range" in result.lower()


@pytest.mark.asyncio
async def test_handle_http_error_returns_friendly_message(monkeypatch):
    result, _ = await _get_weather(monkeypatch, httpx.ConnectError("nope"))
    assert "couldn't reach" in result.lower()


@pytest.mark.asyncio
async def test_handle_no_home_and_no_arg_returns_config_message(monkeypatch):
    result, _ = await _get_weather(monkeypatch, home=False)
    assert "no home location" in result.lower()


@pytest.mark.asyncio
async def test_metric_units_change_forecast_params_and_symbols(monkeypatch):
    result, params = await _get_weather(
        monkeypatch, _forecast_payload(rain_hours=()), units="metric"
    )

    assert params["temperature_unit"] == "celsius"
    assert params["precipitation_unit"] == "mm"
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


def _midnight_hourly_payload(**kwargs: Any) -> tuple[dict, date]:
    """An hourly payload for a location at local midnight right now, starting
    the day before (as the request window does), plus that location's date."""
    offset, local_today = _midnight_offset()
    payload = _hourly_payload(
        start_date=local_today - timedelta(days=1), offset_seconds=offset, **kwargs
    )
    return payload, local_today


@pytest.mark.asyncio
async def test_hourly_returns_48_rows_from_the_current_hour(monkeypatch):
    payload, local_today = _midnight_hourly_payload()
    result, params = await _get_weather(monkeypatch, payload, {"hourly": True})

    assert params["latitude"] == _HOME_LAT
    assert "temperature_2m" in params["hourly"]
    assert "weather_code" in params["hourly"]
    # Hourly mode has no use for the daily block.
    assert "daily" not in params
    assert params["start_date"] == _days_out_iso(-1)
    assert params["end_date"] == _days_out_iso(3)

    assert result.startswith("Home, next 48 hours. Currently 54°F, partly cloudy:")
    rows = _hourly_rows(result)
    assert len(rows) == _HOURLY_HOURS
    # Location-local midnight → the first row is that hour, on today's date.
    assert rows[0] == f"{local_today.strftime('%a')} 12 AM  55°F  light rain  40%"


@pytest.mark.asyncio
async def test_hourly_rows_cross_midnight_with_day_labels(monkeypatch):
    payload, local_today = _midnight_hourly_payload()
    result, _ = await _get_weather(monkeypatch, payload, {"hourly": True})
    rows = _hourly_rows(result)

    tomorrow = local_today + timedelta(days=1)
    # Rows 0-23 are today, 24-47 tomorrow — starting from local midnight.
    assert rows[23].startswith(f"{local_today.strftime('%a')} 11 PM")
    assert rows[24].startswith(f"{tomorrow.strftime('%a')} 12 AM")
    assert rows[47].startswith(f"{tomorrow.strftime('%a')} 11 PM")


@pytest.mark.asyncio
async def test_hourly_ignores_date_argument(monkeypatch):
    payload, _ = _midnight_hourly_payload()
    # A date far outside the daily path's 14-day window must not short-circuit
    # the hourly request.
    result, _ = await _get_weather(
        monkeypatch, payload, {"hourly": True, "date": _days_out_iso(30)}
    )

    assert "two weeks" not in result
    assert len(_hourly_rows(result)) == _HOURLY_HOURS


@pytest.mark.asyncio
async def test_hourly_uses_supplied_coords_and_label(monkeypatch):
    payload, _ = _midnight_hourly_payload()
    result, params = await _get_weather(
        monkeypatch,
        payload,
        {
            "hourly": True,
            "latitude": 43.6591,
            "longitude": -70.2568,
            "place_label": "Portland, Maine",
        },
    )

    assert params["latitude"] == 43.6591
    assert result.startswith("Portland, Maine, next 48 hours")


@pytest.mark.asyncio
async def test_hourly_omits_probability_when_absent(monkeypatch):
    payload, _ = _midnight_hourly_payload(prob=None)
    result, _ = await _get_weather(monkeypatch, payload, {"hourly": True})

    rows = _hourly_rows(result)
    assert rows[0].endswith("light rain")
    assert "%" not in rows[0]


@pytest.mark.asyncio
async def test_hourly_empty_block_returns_friendly_message(monkeypatch):
    result, _ = await _get_weather(
        monkeypatch, {"utc_offset_seconds": 0, "hourly": {}}, {"hourly": True}
    )
    assert "couldn't get an hourly forecast" in result


@pytest.mark.asyncio
async def test_hourly_http_error_returns_friendly_message(monkeypatch):
    result, _ = await _get_weather(
        monkeypatch, httpx.ConnectError("nope"), {"hourly": True}
    )
    assert "couldn't reach" in result.lower()


@pytest.mark.asyncio
async def test_hourly_metric_units(monkeypatch):
    payload, _ = _midnight_hourly_payload()
    result, params = await _get_weather(
        monkeypatch, payload, {"hourly": True}, units="metric"
    )

    assert params["temperature_unit"] == "celsius"
    assert "°C" in result
    assert "°F" not in result


@pytest.mark.asyncio
async def test_hourly_full_window_when_location_is_a_day_ahead(monkeypatch):
    """A location whose local date is ahead of this machine's still gets 48
    rows — the request window is padded a day at each end, not just the near
    one."""
    today = datetime.now().astimezone().date()
    # Put the forecast location at 9 AM on this machine's *tomorrow* (roughly
    # a Pi in US/Eastern asking about Tokyo).
    # Half past the hour, so sub-second drift between computing the offset and
    # the code re-reading the clock can't shift which hour the window starts on.
    target = datetime.combine(today + timedelta(days=1), time(9, 30))
    offset = int((target - datetime.now(UTC).replace(tzinfo=None)).total_seconds())

    def _payload_for_window(params: dict) -> dict:
        start = date.fromisoformat(params["start_date"])
        end = date.fromisoformat(params["end_date"])
        return _hourly_payload(
            start_date=start,
            days=(end - start).days + 1,
            offset_seconds=offset,
        )

    result, params = await _get_weather(
        monkeypatch, _payload_for_window, {"hourly": True}
    )

    assert params["end_date"] == (today + timedelta(days=3)).isoformat()

    rows = _hourly_rows(result)
    assert len(rows) == _HOURLY_HOURS
    assert rows[0].startswith(f"{target.strftime('%a')} 9 AM")


def test_now_local_uses_api_offset_not_server_tz():
    offset, local_today = _midnight_offset()
    local = _now_local({"utc_offset_seconds": offset})
    assert local.tzinfo is None
    assert local.date() == local_today
    assert local.hour == 0
    # Missing offset falls back to this machine's local time rather than raising.
    assert _now_local({}).tzinfo is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_weather_client_daily_and_hourly():
    client = WeatherClient()
    result = await client.get_weather(
        latitude=37.7749,
        longitude=-122.4194,
        place_label="San Francisco",
    )
    assert "San Francisco" in result
    assert "°F" in result
    assert "High" in result

    hourly_result = await client.get_weather(
        latitude=37.7749,
        longitude=-122.4194,
        place_label="San Francisco",
        hourly=True,
    )
    assert "San Francisco, next" in hourly_result
    assert "°F" in hourly_result
