"""Weather tool for the voice assistant.

Provides `get_weather`, which fetches current conditions plus a short
forecast from Open-Meteo (free, no API key). The user's home location is
configured as explicit `[location] latitude` / `longitude` in
`meeko.toml`. For ad-hoc queries ("weather in Tokyo"), Sonnet supplies
`latitude`, `longitude`, and `place_label` directly from its own
geographic knowledge — no geocoder involved.
"""

import logging
from typing import Any

import httpx

from meeko.tools.dispatch import ToolDefinition

logger = logging.getLogger("meeko")

_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
# Generous timeout — a cold DNS lookup + TLS handshake from a Pi can
# eat several seconds on its own; we'd rather wait than fail and have
# Sonnet apologize on a flaky network.
_HTTP_TIMEOUT = 10.0

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

_COMPASS = [
    "N",
    "NNE",
    "NE",
    "ENE",
    "E",
    "ESE",
    "SE",
    "SSE",
    "S",
    "SSW",
    "SW",
    "WSW",
    "W",
    "WNW",
    "NW",
    "NNW",
]


def _wmo_phrase(code: int | None) -> str:
    if code is None:
        return "unknown conditions"
    return _WMO_PHRASES.get(int(code), f"weather code {code}")


def _compass_from_degrees(deg: float | None) -> str:
    if deg is None:
        return ""
    idx = int((deg % 360) / 22.5 + 0.5) % 16
    return _COMPASS[idx]


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
    def _unit_symbols(self) -> tuple[str, str, str]:
        # (temperature, wind speed, precipitation amount)
        if self._units == "metric":
            return ("°C", "km/h", "mm")
        return ("°F", "mph", "in")

    async def get_weather(
        self,
        latitude: float | None,
        longitude: float | None,
        place_label: str | None,
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

        try:
            forecast = await self._fetch_forecast(latitude, longitude)
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning(
                "Weather forecast fetch failed (%s): %s",
                type(exc).__name__,
                exc or "(no message)",
                exc_info=True,
            )
            return "Sorry, I couldn't reach the weather service right now."

        return _format_forecast(label, forecast, self._unit_symbols)

    async def _fetch_forecast(self, lat: float, lon: float) -> dict[str, Any]:
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,weather_code,wind_speed_10m,wind_direction_10m",
            "daily": (
                "temperature_2m_max,temperature_2m_min,"
                "precipitation_probability_max,weather_code"
            ),
            "forecast_days": 2,
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


def _format_forecast(
    place: str,
    forecast: dict[str, Any],
    symbols: tuple[str, str, str],
) -> str:
    temp_sym, wind_sym, _ = symbols

    current = forecast.get("current") or {}
    daily = forecast.get("daily") or {}

    cur_temp = _round(current.get("temperature_2m"))
    cur_phrase = _wmo_phrase(current.get("weather_code"))
    cur_wind = _round(current.get("wind_speed_10m"))
    cur_dir = _compass_from_degrees(current.get("wind_direction_10m"))
    wind_clause = f"wind {cur_wind} {wind_sym}"
    if cur_dir:
        wind_clause += f" from {cur_dir}"

    parts = [f"{place}: {cur_temp}{temp_sym}, {cur_phrase}, {wind_clause}."]

    highs = daily.get("temperature_2m_max") or []
    lows = daily.get("temperature_2m_min") or []
    pops = daily.get("precipitation_probability_max") or []
    codes = daily.get("weather_code") or []

    day_labels = ["Today", "Tomorrow"]
    for i, label in enumerate(day_labels):
        if i >= len(highs) or i >= len(lows):
            break
        segment = (
            f"{label}: high {_round(highs[i])}{temp_sym}, "
            f"low {_round(lows[i])}{temp_sym}"
        )
        if i < len(pops) and pops[i] is not None:
            segment += f", {int(pops[i])}% chance of precipitation"
        if i < len(codes):
            segment += f", {_wmo_phrase(codes[i])}"
        parts.append(segment + ".")

    return " ".join(parts)


# Singleton instance. The orchestrator calls `configure()` on startup.
weather_client = WeatherClient()


def get_tool_definitions() -> list[ToolDefinition]:
    return [
        {
            "name": "get_weather",
            "description": (
                "Get current weather conditions and a short forecast. "
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
        )
    return f"Unknown weather function: {fn_name}"
