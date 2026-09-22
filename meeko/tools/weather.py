"""Weather tool for the voice assistant.

Provides `get_weather`, which fetches conditions from Open-Meteo (free, no
API key). The user's home location is configured as explicit
`[location] latitude` / `longitude` in `meeko.toml`. For ad-hoc queries
("weather in Tokyo"), Sonnet supplies `latitude`, `longitude`, and
`place_label` directly from its own geographic knowledge — no geocoder
involved.

Two modes, selected by the `hourly` flag:

* **Daily** (default) — current conditions plus the forecast for a single
  requested day. Defaults to today; Sonnet (which is told today's date every
  turn) may pass a `date` to ask about any day up to 14 days out. The query is
  bounded to that single day with Open-Meteo's `start_date`/`end_date`, so the
  hourly block stays small even though the horizon is two weeks. Hourly
  precipitation is scanned to say *when* precipitation is expected.
* **Hourly** (`hourly=true`) — one compact row per hour for the next 48 hours
  starting at the current hour, for "what's it doing this afternoon?" style
  questions. `date` is ignored in this mode.
"""

import logging
from collections.abc import Awaitable, Iterable, Iterator
from datetime import UTC, date, datetime, timedelta
from itertools import islice
from typing import Any

import httpx

from meeko.tools.dispatch import ToolDefinition

logger = logging.getLogger("meeko")

_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
# Generous timeout — a cold DNS lookup + TLS handshake from a Pi can
# eat several seconds on its own; we'd rather wait than fail and have
# Sonnet apologize on a flaky network.
_HTTP_TIMEOUT = 10.0

# How far ahead we'll forecast. Open-Meteo's free forecast reaches further,
# but two weeks is plenty for conversation and keeps relative-day references
# unambiguous.
_MAX_FORECAST_DAYS = 13  # today + 13 = 14 days inclusive

# How many hourly rows the `hourly=true` mode returns, starting at the current
# hour. 48 rows of ~30 characters is a small tool result — Sonnet condenses it
# for speech rather than reading it out.
_HOURLY_HOURS = 48

# An hour counts as "wet" when its precipitation probability is at least this
# (percent). When probability is missing we fall back to any measurable amount.
_PRECIP_PROB_THRESHOLD = 50

# WMO weather code → short phrase. Open-Meteo's `weather_code` follows
# WMO 4677 (truncated). https://open-meteo.com/en/docs#api-documentation
_WMO_PHRASES: dict[int, str] = {
    0: "clear",
    1: "mostly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "foggy",
    48: "freezing fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "light freezing drizzle",
    57: "freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light rain showers",
    81: "rain showers",
    82: "violent rain showers",
    85: "light snow showers",
    86: "snow showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "thunderstorm with heavy hail",
}


def _wmo_phrase(code: int | None) -> str:
    if code is None:
        return "unknown conditions"
    return _WMO_PHRASES.get(int(code), f"weather code {code}")


def _fmt_hour(dt: datetime, *, with_ampm: bool = True) -> str:
    """A bare 12-hour clock label, e.g. '2 PM' or (with_ampm=False) '2'."""
    label = dt.strftime("%-I")
    if with_ampm:
        label += " " + dt.strftime("%p")
    return label


def _fmt_run(run: list[datetime]) -> str:
    """Render one contiguous run of wet hours as a spoken time range."""
    first, last = run[0], run[-1]
    if first.hour == last.hour:
        return f"around {_fmt_hour(first)}"
    # Drop the meridiem from the start when both ends share it ("2-4 PM"),
    # keep it when they straddle noon/midnight ("11 AM-1 PM").
    same_ampm = first.strftime("%p") == last.strftime("%p")
    start = _fmt_hour(first, with_ampm=not same_ampm)
    end = _fmt_hour(last)
    return f"around {start}-{end}"


def _now_local(forecast: dict[str, Any]) -> datetime:
    """ "Now" as a naive datetime in the *forecast location's* timezone.

    Open-Meteo is queried with `timezone=auto`, so its hourly timestamps are in
    the target location's local time — which may differ from this machine's. We
    rebuild "now" there from the `utc_offset_seconds` the API returns, falling
    back to this machine's local time if it's missing.
    """
    offset = forecast.get("utc_offset_seconds")
    if isinstance(offset, (int, float)):
        return (datetime.now(UTC) + timedelta(seconds=offset)).replace(tzinfo=None)
    return datetime.now().astimezone().replace(tzinfo=None)


def _now_hour_local(forecast: dict[str, Any], is_today: bool) -> int | None:
    """The current hour (0-23) at the forecast location, or None for a future
    day (where every hour of that day is still ahead)."""
    if not is_today:
        return None
    return _now_local(forecast).hour


def _hourly_series(
    hourly: dict[str, Any], *keys: str
) -> Iterator[tuple[datetime, tuple[Any, ...]]]:
    """Walk Open-Meteo's parallel hourly arrays row by row.

    Yields `(timestamp, values)` with one value per key, in order. Rows whose
    timestamp doesn't parse are skipped; a value array shorter than `time`
    yields `None` past its end rather than raising.
    """
    times = hourly.get("time") or []
    series = [hourly.get(k) or [] for k in keys]
    for i, t in enumerate(times):
        try:
            dt = datetime.fromisoformat(t)
        except TypeError, ValueError:
            continue
        yield dt, tuple(s[i] if i < len(s) else None for s in series)


def _is_wet(prob: Any, amt: Any) -> bool:
    """Probability decides when present; otherwise any measurable amount."""
    if prob is not None:
        return prob >= _PRECIP_PROB_THRESHOLD
    return amt is not None and amt > 0


def _wet_runs(hours: Iterable[tuple[datetime, bool]]) -> list[list[datetime]]:
    """Group consecutive wet hours into runs; a dry hour ends a run."""
    runs: list[list[datetime]] = []
    run: list[datetime] = []
    for dt, wet in hours:
        if wet:
            run.append(dt)
        elif run:
            runs.append(run)
            run = []
    if run:
        runs.append(run)
    return runs


def _precip_window(hourly: dict[str, Any], *, now_hour: int | None) -> str:
    """Scan one day of hourly data and describe when precipitation is likely.

    Returns a phrase like "around 2-4 PM" or "around 8-9 AM and 4-6 PM", or
    "" when no hour crosses the threshold. When `now_hour` is set (the day is
    today, *in the forecast location's timezone*), hours already past are
    skipped so we don't report rain that supposedly happened this morning.
    Pass `None` for a future day, where every hour is still ahead.
    """
    hours = (
        (dt, _is_wet(prob, amt))
        for dt, (prob, amt) in _hourly_series(
            hourly, "precipitation_probability", "precipitation"
        )
        if now_hour is None or dt.hour >= now_hour
    )
    return " and ".join(_fmt_run(r) for r in _wet_runs(hours))


class _FriendlyError(Exception):
    """A friendly message to hand back to Sonnet in place of a forecast.

    `get_weather` catches it and returns the message, so the tool never raises
    and the tool-use turn stays intact.
    """


def _resolve_date(date_str: str | None, today: date) -> date:
    """The requested forecast day: today by default, else up to
    `_MAX_FORECAST_DAYS` ahead."""
    if date_str is None:
        return today
    try:
        req_date = date.fromisoformat(date_str)
    except TypeError, ValueError:
        raise _FriendlyError(
            "I didn't understand that date. Try asking for a specific day."
        ) from None
    delta = (req_date - today).days
    if delta < 0:
        raise _FriendlyError(
            "I can only look ahead, not back — try today or a day to come."
        )
    if delta > _MAX_FORECAST_DAYS:
        raise _FriendlyError("I can only forecast about two weeks ahead.")
    return req_date


async def _fetch_guarded(kind: str, fetch: Awaitable[dict[str, Any]]) -> dict[str, Any]:
    """Await an Open-Meteo fetch, turning network/parse failures into a
    logged warning and a `_FriendlyError`."""
    try:
        return await fetch
    except (httpx.HTTPError, ValueError) as exc:
        # The message and traceback of an httpx error embed the request
        # URL, whose query string carries the coordinates — the home
        # location, or wherever the user asked about. Only the exception
        # type and (for HTTP errors) the status code are safe at INFO.
        response = getattr(exc, "response", None)
        logger.warning(
            "Weather %s fetch failed (%s%s)",
            kind,
            type(exc).__name__,
            f", HTTP {response.status_code}" if response is not None else "",
        )
        logger.debug("Weather %s fetch failure detail", kind, exc_info=True)
        raise _FriendlyError(
            "Sorry, I couldn't reach the weather service right now."
        ) from None


class WeatherClient:
    """Fetches forecasts from Open-Meteo.

    Configured at startup in `meeko/main.py` via `configure()`. The home
    location is supplied as explicit lat/lon; ad-hoc per-call locations
    arrive as caller-supplied (lat, lon, label) tuples — Sonnet produces
    them from its own knowledge, no geocoder needed.
    """

    def __init__(self) -> None:
        self._latitude: float | None = None
        self._longitude: float | None = None
        self._units: str = "imperial"

    def configure(
        self,
        *,
        latitude: float | None,
        longitude: float | None,
        units: str,
    ) -> None:
        self._latitude = latitude
        self._longitude = longitude
        self._units = units

    @property
    def _unit_params(self) -> dict[str, str]:
        if self._units == "metric":
            return {
                "temperature_unit": "celsius",
                "wind_speed_unit": "kmh",
                "precipitation_unit": "mm",
            }
        return {
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "inch",
        }

    @property
    def _temp_symbol(self) -> str:
        return "°C" if self._units == "metric" else "°F"

    async def get_weather(
        self,
        latitude: float | None,
        longitude: float | None,
        place_label: str | None,
        date_str: str | None = None,
        hourly: bool = False,
    ) -> str:
        try:
            lat, lon, label = self._resolve_location(latitude, longitude, place_label)
            today = datetime.now().astimezone().date()

            if hourly:
                # The window is always "the next 48 hours from now" — any
                # `date` Sonnet happened to pass alongside is ignored.
                forecast = await _fetch_guarded("hourly", self._fetch_hourly(lat, lon))
                return _format_hourly(label, forecast, self._temp_symbol)

            req_date = _resolve_date(date_str, today)
            forecast = await _fetch_guarded(
                "forecast", self._fetch_forecast(lat, lon, req_date)
            )
            return _format_forecast(label, forecast, req_date, today, self._temp_symbol)
        except _FriendlyError as exc:
            return str(exc)

    def _resolve_location(
        self,
        latitude: float | None,
        longitude: float | None,
        place_label: str | None,
    ) -> tuple[float, float, str]:
        """Caller-supplied place if all three args are given, home if none."""
        supplied = (latitude, longitude, place_label)
        if supplied == (None, None, None):
            return self._home_location()
        if latitude is None or longitude is None or place_label is None:
            raise _FriendlyError(
                "To look up a specific place, please supply latitude, "
                "longitude, and a place label all together."
            )
        if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
            raise _FriendlyError("Those coordinates look out of range.")
        return latitude, longitude, place_label or "there"

    def _home_location(self) -> tuple[float, float, str]:
        if self._latitude is None or self._longitude is None:
            raise _FriendlyError(
                "No home location is configured. Try asking for a specific city."
            )
        return self._latitude, self._longitude, "Home"

    async def _fetch_forecast(
        self, lat: float, lon: float, req_date: date
    ) -> dict[str, Any]:
        iso = req_date.isoformat()
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,weather_code",
            "daily": (
                "temperature_2m_max,temperature_2m_min,"
                "precipitation_probability_max,weather_code"
            ),
            "hourly": "precipitation_probability,precipitation",
            "start_date": iso,
            "end_date": iso,
            "timezone": "auto",
            **self._unit_params,
        }
        return await self._get(params)

    async def _fetch_hourly(self, lat: float, lon: float) -> dict[str, Any]:
        # Open-Meteo's hourly arrays always start at 00:00 local of `start_date`,
        # so we ask for a window and slice from "now" at the target location.
        # 48 hours from the current hour reaches into the third calendar day, and
        # the target's local date can be a day either side of this machine's (UTC
        # offsets span -12 to +14) — so we pad a spare day at *both* ends. Behind
        # us, their "now" would otherwise precede the array; ahead of us, the
        # array would run out before 48 rows.
        today = datetime.now().astimezone().date()
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,weather_code",
            "hourly": "temperature_2m,precipitation_probability,weather_code",
            "start_date": (today - timedelta(days=1)).isoformat(),
            "end_date": (today + timedelta(days=3)).isoformat(),
            "timezone": "auto",
            **self._unit_params,
        }
        return await self._get(params)

    async def _get(self, params: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            resp = await client.get(_FORECAST_URL, params=params)
            resp.raise_for_status()
            return resp.json()


def _round(value: Any) -> str:
    try:
        return str(round(float(value)))
    except TypeError, ValueError:
        return "?"


def _day_label(req_date: date, today: date) -> str:
    delta = (req_date - today).days
    if delta == 0:
        return "today"
    if delta == 1:
        return "tomorrow"
    return req_date.strftime("%A")


def _first(seq: Any) -> Any:
    """First element of a daily array, or None when empty/missing."""
    if isinstance(seq, list) and seq:
        return seq[0]
    return None


def _format_forecast(
    place: str,
    forecast: dict[str, Any],
    req_date: date,
    today: date,
    temp_sym: str,
) -> str:
    is_today = req_date == today

    current = forecast.get("current") or {}
    daily = forecast.get("daily") or {}
    hourly = forecast.get("hourly") or {}

    # Conditions phrase: today uses the live code, a future day its daily code.
    if is_today:
        phrase = _wmo_phrase(current.get("weather_code"))
    else:
        phrase = _wmo_phrase(_first(daily.get("weather_code")))

    label = _day_label(req_date, today)
    parts = [f"{place}, {label} ({req_date.isoformat()}): {phrase}."]

    if is_today:
        cur_temp = _round(current.get("temperature_2m"))
        humidity = _round(current.get("relative_humidity_2m"))
        parts.append(f"Currently {cur_temp}{temp_sym}, humidity {humidity}%.")

    hi = _round(_first(daily.get("temperature_2m_max")))
    lo = _round(_first(daily.get("temperature_2m_min")))
    parts.append(f"High {hi}{temp_sym}, low {lo}{temp_sym}.")

    prob = _first(daily.get("precipitation_probability_max"))
    if prob is not None:
        precip = f"{int(prob)}% chance of precipitation"
        if prob >= _PRECIP_PROB_THRESHOLD:
            now_hour = _now_hour_local(forecast, is_today)
            window = _precip_window(hourly, now_hour=now_hour)
            if window:
                precip += f", precipitation likely {window}"
        parts.append(precip + ".")

    return " ".join(parts)


def _format_hourly(place: str, forecast: dict[str, Any], temp_sym: str) -> str:
    """Render up to `_HOURLY_HOURS` rows, one per hour, starting at the current
    hour at the forecast location."""
    hourly = forecast.get("hourly") or {}

    # Truncate to the hour so the row covering the current hour is included.
    cutoff = _now_local(forecast).replace(minute=0, second=0, microsecond=0)

    series = _hourly_series(
        hourly, "temperature_2m", "weather_code", "precipitation_probability"
    )
    upcoming = ((dt, vals) for dt, vals in series if dt >= cutoff)
    rows = [
        _hourly_row(dt, *vals, temp_sym=temp_sym)
        for dt, vals in islice(upcoming, _HOURLY_HOURS)
    ]

    if not rows:
        return "Sorry, I couldn't get an hourly forecast for there right now."

    current = forecast.get("current") or {}
    header = _hourly_header(place, current, len(rows), temp_sym)
    return header + ":\n" + "\n".join(rows)


def _hourly_row(dt: datetime, temp: Any, code: Any, prob: Any, *, temp_sym: str) -> str:
    """One hourly line, e.g. 'Mon 3 PM  54°F  rain  90%'."""
    row = f"{dt.strftime('%a')} {_fmt_hour(dt)}  {_round(temp)}{temp_sym}  "
    row += _wmo_phrase(code)
    if prob is not None:
        row += f"  {int(prob)}%"
    return row


def _hourly_header(
    place: str, current: dict[str, Any], n_rows: int, temp_sym: str
) -> str:
    header = f"{place}, next {n_rows} hours"
    cur_temp = current.get("temperature_2m")
    if cur_temp is not None:
        header += (
            f". Currently {_round(cur_temp)}{temp_sym}, "
            f"{_wmo_phrase(current.get('weather_code'))}"
        )
    return header


# Singleton instance. meeko/main.py calls `configure()` on startup.
weather_client = WeatherClient()


def get_tool_definitions() -> list[ToolDefinition]:
    return [
        {
            "name": "get_weather",
            "description": (
                "Get weather conditions and the forecast for a single day, "
                "or an hour-by-hour forecast for the next 48 hours. "
                "Set `hourly` to true when the question is about part of a "
                "day — 'this afternoon', 'tonight', 'when I leave at 7 "
                "tomorrow' — and summarize the rows rather than reading "
                "them all out. Leave `hourly` off for whole-day questions. "
                "Defaults to today; to ask about a future day, pass `date` "
                "as YYYY-MM-DD — you know today's date, so resolve relative "
                "references like 'tomorrow' or 'Tuesday' yourself. Any day "
                "from today up to 14 days out is supported. "
                "To look up a place other than the user's home, supply "
                "`latitude` and `longitude` in decimal degrees from your "
                "own knowledge of world geography — DO NOT call "
                "web_search to find them. Also supply `place_label`, a "
                "short human-readable name for the location (e.g. "
                "'Portland, Maine') that will appear in the spoken "
                "response. Omit all three to use the user's configured "
                "home location."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "latitude": {
                        "type": "number",
                        "description": "Decimal degrees, -90 to 90.",
                    },
                    "longitude": {
                        "type": "number",
                        "description": "Decimal degrees, -180 to 180.",
                    },
                    "place_label": {
                        "type": "string",
                        "description": (
                            "Short name used in the spoken response, "
                            "e.g. 'Tokyo' or 'Portland, Maine'."
                        ),
                    },
                    "date": {
                        "type": "string",
                        "description": (
                            "Day to forecast, as YYYY-MM-DD. Omit for "
                            "today. Must be today or within the next 14 days. "
                            "Ignored when `hourly` is true."
                        ),
                    },
                    "hourly": {
                        "type": "boolean",
                        "description": (
                            "Return one row per hour for the next 48 hours "
                            "instead of a single-day summary."
                        ),
                    },
                },
            },
        },
    ]


async def handle(
    fn_name: str,
    args: dict,
    *,
    client: WeatherClient | None = None,
) -> str:
    """Handle a weather-related function call."""
    cl = client or weather_client
    if fn_name == "get_weather":
        return await cl.get_weather(
            latitude=args.get("latitude"),
            longitude=args.get("longitude"),
            place_label=args.get("place_label"),
            date_str=args.get("date"),
            hourly=bool(args.get("hourly")),
        )
    return f"Unknown weather function: {fn_name}"
