"""Weather tool for the voice assistant.

Provides `get_weather`, which fetches current conditions plus the forecast
for a single requested day from Open-Meteo (free, no API key). The user's
home location is configured as explicit `[location] latitude` / `longitude`
in `meeko.toml`. For ad-hoc queries ("weather in Tokyo"), Sonnet supplies
`latitude`, `longitude`, and `place_label` directly from its own geographic
knowledge — no geocoder involved.

The reply defaults to **today**; Sonnet (which is told today's date every
turn) may pass a `date` to ask about any day up to 14 days out. The query is
bounded to that single day with Open-Meteo's `start_date`/`end_date`, so the
hourly block stays small even though the horizon is two weeks. Hourly
precipitation is scanned to say *when* precipitation is expected.
"""

import logging
from datetime import UTC, date, datetime, timedelta
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
    # Drop the meridiem from the start when both ends share it ("2–4 PM"),
    # keep it when they straddle noon/midnight ("11 AM–1 PM").
    same_ampm = first.strftime("%p") == last.strftime("%p")
    start = _fmt_hour(first, with_ampm=not same_ampm)
    end = _fmt_hour(last)
    return f"around {start}–{end}"


def _now_hour_local(forecast: dict[str, Any], is_today: bool) -> int | None:
    """The current hour (0–23) in the *forecast location's* timezone, or None
    for a future day. Open-Meteo is queried with `timezone=auto`, so its hourly
    timestamps are in the target location's local time — which may differ from
    this machine's. We rebuild "now" there from the `utc_offset_seconds` the API
    returns, falling back to this machine's local hour if it's missing."""
    if not is_today:
        return None
    offset = forecast.get("utc_offset_seconds")
    if isinstance(offset, (int, float)):
        local = datetime.now(UTC) + timedelta(seconds=offset)
        return local.hour
    return datetime.now().astimezone().hour


def _precip_window(hourly: dict[str, Any], *, now_hour: int | None) -> str:
    """Scan one day of hourly data and describe when precipitation is likely.

    Returns a phrase like "around 2–4 PM" or "around 8–9 AM and 4–6 PM", or
    "" when no hour crosses the threshold. When `now_hour` is set (the day is
    today, *in the forecast location's timezone*), hours already past are
    skipped so we don't report rain that supposedly happened this morning.
    Pass `None` for a future day, where every hour is still ahead.
    """
    times = hourly.get("time") or []
    probs = hourly.get("precipitation_probability") or []
    amts = hourly.get("precipitation") or []

    runs: list[list[datetime]] = []
    run: list[datetime] = []
    for i, t in enumerate(times):
        try:
            dt = datetime.fromisoformat(t)
        except TypeError, ValueError:
            continue
        if now_hour is not None and dt.hour < now_hour:
            continue
        prob = probs[i] if i < len(probs) else None
        amt = amts[i] if i < len(amts) else None
        if prob is not None:
            wet = prob >= _PRECIP_PROB_THRESHOLD
        else:
            wet = amt is not None and amt > 0
        if wet:
            run.append(dt)
        elif run:
            runs.append(run)
            run = []
    if run:
        runs.append(run)

    if not runs:
        return ""
    return " and ".join(_fmt_run(r) for r in runs)


class WeatherClient:
    """Fetches forecasts from Open-Meteo.

    Configured at orchestrator startup via `configure()`. The home
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
    ) -> str:
        supplied = [v is not None for v in (latitude, longitude, place_label)]
        if any(supplied) and not all(supplied):
            return (
                "To look up a specific place, please supply latitude, "
                "longitude, and a place label all together."
            )

        if all(supplied):
            assert latitude is not None and longitude is not None
            if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
                return "Those coordinates look out of range."
            label = place_label or "there"
        elif self._latitude is not None and self._longitude is not None:
            latitude = self._latitude
            longitude = self._longitude
            label = "Home"
        else:
            return "No home location is configured. Try asking for a specific city."

        today = datetime.now().astimezone().date()
        if date_str is None:
            req_date = today
        else:
            try:
                req_date = date.fromisoformat(date_str)
            except TypeError, ValueError:
                return "I didn't understand that date. Try asking for a specific day."
            delta = (req_date - today).days
            if delta < 0:
                return "I can only look ahead, not back — try today or a day to come."
            if delta > _MAX_FORECAST_DAYS:
                return "I can only forecast about two weeks ahead."

        try:
            forecast = await self._fetch_forecast(latitude, longitude, req_date)
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning(
                "Weather forecast fetch failed (%s): %s",
                type(exc).__name__,
                exc or "(no message)",
                exc_info=True,
            )
            return "Sorry, I couldn't reach the weather service right now."

        return _format_forecast(label, forecast, req_date, today, self._temp_symbol)

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


# Singleton instance. The orchestrator calls `configure()` on startup.
weather_client = WeatherClient()


def get_tool_definitions() -> list[ToolDefinition]:
    return [
        {
            "name": "get_weather",
            "description": (
                "Get weather conditions and the forecast for a single day. "
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
                            "today. Must be today or within the next 14 days."
                        ),
                    },
                },
            },
        },
    ]


async def handle(fn_name: str, args: dict) -> str:
    """Handle a weather-related function call."""
    if fn_name == "get_weather":
        return await weather_client.get_weather(
            latitude=args.get("latitude"),
            longitude=args.get("longitude"),
            place_label=args.get("place_label"),
            date_str=args.get("date"),
        )
    return f"Unknown weather function: {fn_name}"
