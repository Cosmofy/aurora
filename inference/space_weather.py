"""
Space Weather Data from NOAA SWPC (Free, no API key)

Fetches:
- Kp index
- Bz (IMF component)
- Solar wind speed
- Solar wind density
- Dst index

Cached in Redis (2 min TTL) - data is global, same for all locations
"""
import os
import json
import httpx
from datetime import datetime, timedelta

import redis

# Redis connection
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
CACHE_TTL = 120  # 2 minutes

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# NOAA SWPC endpoints (all free, no auth)
NOAA_KP_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
NOAA_PLASMA_URL = "https://services.swpc.noaa.gov/products/solar-wind/plasma-1-day.json"
NOAA_MAG_URL = "https://services.swpc.noaa.gov/products/solar-wind/mag-1-day.json"
KYOTO_DST_URL = "https://wdc.kugi.kyoto-u.ac.jp/dst_realtime/presentmonth/index.html"

CACHE_KEY = "aurora:space_weather"


async def get_space_weather() -> dict:
    """Fetch current space weather conditions from NOAA (cached in Redis)"""

    # Try cache first
    try:
        cached = redis_client.get(CACHE_KEY)
        if cached:
            return json.loads(cached)
    except Exception as e:
        print(f"Redis read error: {e}")

    # Fetch fresh data
    async with httpx.AsyncClient(timeout=10.0) as client:
        kp_index = await _get_kp(client)
        plasma = await _get_plasma(client)
        bz = await _get_bz(client)
        dst = await _get_dst(client, kp_index)

    data = {
        "kp_index": kp_index,
        "bz": bz,
        "solar_wind_speed": plasma["speed"],
        "solar_wind_density": plasma["density"],
        "dst": dst
    }

    # Cache it
    try:
        redis_client.setex(CACHE_KEY, CACHE_TTL, json.dumps(data))
    except Exception as e:
        print(f"Redis write error: {e}")

    return data


async def _get_kp(client: httpx.AsyncClient) -> float:
    """Get latest Kp index from NOAA"""
    try:
        response = await client.get(NOAA_KP_URL)
        response.raise_for_status()
        data = response.json()

        # Format: [["time_tag", "Kp", "Kp_fraction", ...], [...data rows...]]
        # Get the most recent row
        if len(data) > 1:
            latest = data[-1]
            kp = float(latest[1])  # Kp value
            return kp
    except Exception as e:
        print(f"Error fetching Kp: {e}")

    return 3.0  # Default moderate value


async def _get_plasma(client: httpx.AsyncClient) -> dict:
    """Get solar wind speed and density from NOAA"""
    try:
        response = await client.get(NOAA_PLASMA_URL)
        response.raise_for_status()
        data = response.json()

        # Format: [["time_tag", "density", "speed", "temperature"], [...data rows...]]
        # Get most recent valid row (skip nulls)
        for row in reversed(data[1:]):
            if row[1] is not None and row[2] is not None:
                return {
                    "density": float(row[1]),
                    "speed": float(row[2])
                }
    except Exception as e:
        print(f"Error fetching plasma: {e}")

    return {"density": 5.0, "speed": 400.0}  # Defaults


async def _get_bz(client: httpx.AsyncClient) -> float:
    """Get IMF Bz component from NOAA"""
    try:
        response = await client.get(NOAA_MAG_URL)
        response.raise_for_status()
        data = response.json()

        # Format: [["time_tag", "bx_gsm", "by_gsm", "bz_gsm", ...], [...data rows...]]
        # Bz is index 3
        for row in reversed(data[1:]):
            if row[3] is not None:
                return float(row[3])
    except Exception as e:
        print(f"Error fetching Bz: {e}")

    return 0.0  # Default neutral


async def _get_dst(client: httpx.AsyncClient, kp_index: float) -> float:
    """
    Get Dst index.
    Kyoto WDC provides real-time Dst but parsing HTML is fragile.
    Fallback: estimate from Kp using empirical relationship.
    """
    # Empirical Kp to Dst approximation:
    # Kp 0-2: Dst ~0 to -20
    # Kp 3-4: Dst ~-20 to -50
    # Kp 5-6: Dst ~-50 to -100
    # Kp 7+: Dst < -100

    # Simple linear approximation
    if kp_index < 3:
        dst = -5 * kp_index
    elif kp_index < 5:
        dst = -15 - 15 * (kp_index - 3)
    elif kp_index < 7:
        dst = -45 - 25 * (kp_index - 5)
    else:
        dst = -95 - 30 * (kp_index - 7)

    return round(dst, 1)


# For testing
if __name__ == "__main__":
    import asyncio

    async def test():
        data = await get_space_weather()
        print("Space Weather Data:")
        for k, v in data.items():
            print(f"  {k}: {v}")

    asyncio.run(test())
