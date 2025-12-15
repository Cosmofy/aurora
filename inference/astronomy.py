"""
Astronomical Calculations

Calculates:
- Magnetic latitude
- Sun altitude
- Moon phase
- Moon illumination
- Moon altitude
"""
import math
from datetime import datetime


def calculate_astronomy(lat: float, lon: float, timestamp: str = None) -> dict:
    """Calculate all astronomical features for a location and time."""

    if timestamp:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    else:
        dt = datetime.utcnow()

    return {
        "hour": dt.hour,
        "day_of_year": dt.timetuple().tm_yday,
        "magnetic_latitude": calculate_magnetic_latitude(lat, lon),
        "sun_altitude": calculate_sun_altitude(lat, lon, dt),
        "moon_phase": calculate_moon_phase(dt),
        "moon_illumination": calculate_moon_illumination(dt),
        "moon_altitude": calculate_moon_altitude(lat, lon, dt)
    }


def calculate_magnetic_latitude(lat: float, lon: float) -> float:
    """Geomagnetic latitude using dipole approximation."""
    pole_lat, pole_lon = 80.7, -72.7
    lat_r, lon_r = math.radians(lat), math.radians(lon)
    pole_lat_r, pole_lon_r = math.radians(pole_lat), math.radians(pole_lon)

    cos_mlat = (math.sin(lat_r) * math.sin(pole_lat_r) +
                math.cos(lat_r) * math.cos(pole_lat_r) *
                math.cos(lon_r - pole_lon_r))
    return round(90 - math.degrees(math.acos(max(-1, min(1, cos_mlat)))), 2)


def calculate_moon_phase(dt: datetime) -> float:
    """Moon phase: 0=new, 0.5=full, 1=new."""
    known_new = datetime(2000, 1, 6, 18, 14)
    synodic_month = 29.530588853

    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))

    # Handle timezone-aware datetime
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)

    days_since = (dt - known_new).total_seconds() / 86400
    return round((days_since % synodic_month) / synodic_month, 3)


def calculate_moon_illumination(dt: datetime) -> float:
    """Moon illumination: 0=new moon (dark), 1=full moon (bright)."""
    phase = calculate_moon_phase(dt)
    # Convert phase (0-1 cycle) to illumination (0-1, peaks at 0.5)
    # At phase 0 (new): illumination = 0
    # At phase 0.5 (full): illumination = 1
    # At phase 1 (new again): illumination = 0
    illumination = 1 - abs(2 * phase - 1)
    return round(illumination, 3)


def calculate_sun_altitude(lat: float, lon: float, dt: datetime) -> float:
    """Sun altitude in degrees (negative = night)."""
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))

    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)

    n = dt.timetuple().tm_yday
    declination = 23.45 * math.sin(math.radians((360/365) * (n - 81)))
    hour = dt.hour + dt.minute / 60
    hour_angle = 15 * (hour - 12) - lon

    lat_r = math.radians(lat)
    dec_r = math.radians(declination)
    ha_r = math.radians(hour_angle)

    sin_alt = (math.sin(lat_r) * math.sin(dec_r) +
               math.cos(lat_r) * math.cos(dec_r) * math.cos(ha_r))
    return round(math.degrees(math.asin(max(-1, min(1, sin_alt)))), 1)


def calculate_moon_altitude(lat: float, lon: float, dt: datetime) -> float:
    """Approximate moon altitude."""
    phase = calculate_moon_phase(dt)
    sun_alt = calculate_sun_altitude(lat, lon, dt)

    # Rough approximation: moon is roughly opposite to sun at full moon
    if sun_alt < 0:
        return round(-sun_alt * (0.5 + 0.5 * abs(phase - 0.5) * 2), 1)
    return round(max(-90, min(90, sun_alt - 30)), 1)


# For testing
if __name__ == "__main__":
    # Test with Reykjavik, Iceland
    lat, lon = 64.1, -21.9

    result = calculate_astronomy(lat, lon)
    print("Astronomical Data (Reykjavik, now):")
    for k, v in result.items():
        print(f"  {k}: {v}")
