"""
Weather Data from Apple WeatherKit

Required credentials (set in environment or config):
- WEATHERKIT_KEY_ID: Your key ID from Apple Developer
- WEATHERKIT_TEAM_ID: Your team ID
- WEATHERKIT_SERVICE_ID: Your service ID (e.g., com.yourapp.weatherkit)
- WEATHERKIT_PRIVATE_KEY: Path to .p8 private key file

Fallback: Open-Meteo (free, no auth) if WeatherKit fails
"""
import os
import httpx
import jwt
import time
from datetime import datetime

# WeatherKit config (set via environment variables)
WEATHERKIT_KEY_ID = os.getenv("WEATHERKIT_KEY_ID")
WEATHERKIT_TEAM_ID = os.getenv("WEATHERKIT_TEAM_ID")
WEATHERKIT_SERVICE_ID = os.getenv("WEATHERKIT_SERVICE_ID")
WEATHERKIT_PRIVATE_KEY_PATH = os.getenv("WEATHERKIT_PRIVATE_KEY")

WEATHERKIT_URL = "https://weatherkit.apple.com/api/v1/weather"
OPENMETEO_URL = "https://api.open-meteo.com/v1/forecast"


def _generate_weatherkit_token() -> str:
    """Generate JWT token for WeatherKit API."""
    if not all([WEATHERKIT_KEY_ID, WEATHERKIT_TEAM_ID, WEATHERKIT_SERVICE_ID, WEATHERKIT_PRIVATE_KEY_PATH]):
        raise ValueError("WeatherKit credentials not configured")

    # Support both file path and direct key content
    if WEATHERKIT_PRIVATE_KEY_PATH.startswith("-----BEGIN"):
        private_key = WEATHERKIT_PRIVATE_KEY_PATH
    else:
        with open(WEATHERKIT_PRIVATE_KEY_PATH, 'r') as f:
            private_key = f.read()

    now = int(time.time())
    payload = {
        "iss": WEATHERKIT_TEAM_ID,
        "iat": now,
        "exp": now + 3600,  # 1 hour
        "sub": WEATHERKIT_SERVICE_ID
    }

    token = jwt.encode(
        payload,
        private_key,
        algorithm="ES256",
        headers={"kid": WEATHERKIT_KEY_ID, "id": f"{WEATHERKIT_TEAM_ID}.{WEATHERKIT_SERVICE_ID}"}
    )
    return token


async def _get_weatherkit(lat: float, lon: float) -> dict:
    """Fetch weather from Apple WeatherKit."""
    token = _generate_weatherkit_token()

    url = f"{WEATHERKIT_URL}/en/{lat}/{lon}"
    params = {"dataSets": "currentWeather"}
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()

    current = data.get("currentWeather", {})

    return {
        "cloudcover": current.get("cloudCover", 0) * 100,  # Convert 0-1 to 0-100
        "precip": current.get("precipitationIntensity", 0),
        "temp": current.get("temperature", 10),
        "pressure": current.get("pressure", 1013),
        "windspeed": current.get("windSpeed", 0) * 3.6,  # m/s to km/h
        "winddir": current.get("windDirection", 0),
        "humidity": current.get("humidity", 50) * 100,  # Convert 0-1 to 0-100
        "dew": current.get("temperatureDewPoint", 0)
    }


async def _get_openmeteo(lat: float, lon: float) -> dict:
    """Fallback: Fetch weather from Open-Meteo (free)."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "cloud_cover,precipitation,temperature_2m,pressure_msl,wind_speed_10m,wind_direction_10m,relative_humidity_2m,dew_point_2m"
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(OPENMETEO_URL, params=params)
        response.raise_for_status()
        data = response.json()

    current = data.get("current", {})

    return {
        "cloudcover": current.get("cloud_cover", 50),
        "precip": current.get("precipitation", 0),
        "temp": current.get("temperature_2m", 10),
        "pressure": current.get("pressure_msl", 1013),
        "windspeed": current.get("wind_speed_10m", 0),
        "winddir": current.get("wind_direction_10m", 0),
        "humidity": current.get("relative_humidity_2m", 50),
        "dew": current.get("dew_point_2m", 0)
    }


async def get_weather(lat: float, lon: float) -> dict:
    """
    Get current weather for a location.
    Uses WeatherKit if configured, falls back to Open-Meteo.
    """
    # Try WeatherKit first if configured
    if all([WEATHERKIT_KEY_ID, WEATHERKIT_TEAM_ID, WEATHERKIT_SERVICE_ID, WEATHERKIT_PRIVATE_KEY_PATH]):
        try:
            return await _get_weatherkit(lat, lon)
        except Exception as e:
            print(f"WeatherKit failed, falling back to Open-Meteo: {e}")

    # Fallback to Open-Meteo
    return await _get_openmeteo(lat, lon)


# For testing
if __name__ == "__main__":
    import asyncio

    async def test():
        # Test with Reykjavik
        lat, lon = 64.1, -21.9
        data = await get_weather(lat, lon)
        print("Weather Data (Reykjavik):")
        for k, v in data.items():
            print(f"  {k}: {v}")

    asyncio.run(test())
