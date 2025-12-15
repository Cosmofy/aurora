"""
Feature Engineering

Combines weather, space weather, and astronomy into the 30 features
required by the model.
"""

# Feature order expected by the models (must match training)
FEATURE_ORDER = [
    "latitude", "longitude", "hour", "day_of_year",
    "magnetic_latitude", "sun_altitude", "moon_phase", "moon_illumination", "moon_altitude",
    "cloudcover", "precip", "temp", "pressure", "windspeed", "winddir", "dew", "humidity",
    "kp_index", "bz", "solar_wind_speed", "solar_wind_density", "dst",
    "is_dark", "moon_interference", "storm", "strong_storm", "lat_kp", "dark_storm", "good_conditions", "sw_pressure"
]


def engineer_features(
    lat: float,
    lon: float,
    weather: dict,
    space: dict,
    astro: dict
) -> dict:
    """
    Build the complete feature vector for model prediction.

    Args:
        lat: Latitude
        lon: Longitude
        weather: Dict with cloudcover, precip, temp, pressure, windspeed, winddir, humidity, dew
        space: Dict with kp_index, bz, solar_wind_speed, solar_wind_density, dst
        astro: Dict with hour, day_of_year, magnetic_latitude, sun_altitude, moon_phase, moon_illumination, moon_altitude

    Returns:
        Dict with all 30 features
    """

    # Base features
    features = {
        # Location
        "latitude": lat,
        "longitude": lon,

        # Time
        "hour": astro["hour"],
        "day_of_year": astro["day_of_year"],

        # Astronomical
        "magnetic_latitude": astro["magnetic_latitude"],
        "sun_altitude": astro["sun_altitude"],
        "moon_phase": astro["moon_phase"],
        "moon_illumination": astro["moon_illumination"],
        "moon_altitude": astro["moon_altitude"],

        # Weather
        "cloudcover": weather["cloudcover"],
        "precip": weather["precip"],
        "temp": weather["temp"],
        "pressure": weather["pressure"],
        "windspeed": weather["windspeed"],
        "winddir": weather["winddir"],
        "dew": weather["dew"],
        "humidity": weather["humidity"],

        # Space weather
        "kp_index": space["kp_index"],
        "bz": space["bz"],
        "solar_wind_speed": space["solar_wind_speed"],
        "solar_wind_density": space["solar_wind_density"],
        "dst": space["dst"],
    }

    # Engineered features
    features["is_dark"] = int(astro["sun_altitude"] < -6)
    features["moon_interference"] = int(
        astro["moon_altitude"] > 0 and astro["moon_illumination"] > 0.5
    )
    features["storm"] = int(space["kp_index"] >= 5)
    features["strong_storm"] = int(space["dst"] < -50)
    features["lat_kp"] = astro["magnetic_latitude"] * space["kp_index"]
    features["dark_storm"] = features["is_dark"] * features["storm"]
    features["good_conditions"] = int(
        weather["cloudcover"] < 50 and astro["sun_altitude"] < -6
    )
    features["sw_pressure"] = space["solar_wind_speed"] * space["solar_wind_density"]

    return features


# For testing
if __name__ == "__main__":
    # Example inputs
    weather = {
        "cloudcover": 20, "precip": 0, "temp": -5,
        "pressure": 1020, "windspeed": 15, "winddir": 270,
        "humidity": 60, "dew": -10
    }
    space = {
        "kp_index": 6, "bz": -8, "solar_wind_speed": 550,
        "solar_wind_density": 8, "dst": -75
    }
    astro = {
        "hour": 22, "day_of_year": 45,
        "magnetic_latitude": 67.5, "sun_altitude": -15,
        "moon_phase": 0.1, "moon_illumination": 0.2, "moon_altitude": -20
    }

    features = engineer_features(65.0, -18.0, weather, space, astro)

    print("Feature Vector (30 features):")
    for i, (k, v) in enumerate(features.items(), 1):
        print(f"  {i:2d}. {k}: {v}")
