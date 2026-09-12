import os
import numpy as np
import pandas as pd
from datetime import datetime

# ============================================================
# WEATHER DATA GENERATOR
# No Spark required
# Output: data/weather_prediction.csv
# ============================================================

np.random.seed(42)

# ------------------------------------------------------------
# 1. Cities and approximate weather characteristics
# ------------------------------------------------------------

cities = {
    "Delhi": {
        "lat": 28.6139,
        "lon": 77.2090,
        "base_temp": 27,
        "temp_range": 13,
        "humidity": 55,
        "pressure": 1008,
        "wind": 12
    },
    "Mumbai": {
        "lat": 19.0760,
        "lon": 72.8777,
        "base_temp": 28,
        "temp_range": 7,
        "humidity": 72,
        "pressure": 1010,
        "wind": 14
    },
    "Bengaluru": {
        "lat": 12.9716,
        "lon": 77.5946,
        "base_temp": 24,
        "temp_range": 8,
        "humidity": 65,
        "pressure": 1012,
        "wind": 11
    },
    "Chennai": {
        "lat": 13.0827,
        "lon": 80.2707,
        "base_temp": 29,
        "temp_range": 7,
        "humidity": 70,
        "pressure": 1009,
        "wind": 15
    },
    "Kolkata": {
        "lat": 22.5726,
        "lon": 88.3639,
        "base_temp": 28,
        "temp_range": 10,
        "humidity": 70,
        "pressure": 1007,
        "wind": 10
    },
    "Hyderabad": {
        "lat": 17.3850,
        "lon": 78.4867,
        "base_temp": 27,
        "temp_range": 11,
        "humidity": 58,
        "pressure": 1011,
        "wind": 13
    },
    "Lucknow": {
        "lat": 26.8467,
        "lon": 80.9462,
        "base_temp": 26,
        "temp_range": 13,
        "humidity": 58,
        "pressure": 1009,
        "wind": 10
    },
    "Pune": {
        "lat": 18.5204,
        "lon": 73.8567,
        "base_temp": 25,
        "temp_range": 10,
        "humidity": 60,
        "pressure": 1010,
        "wind": 12
    },
    "Jaipur": {
        "lat": 26.9124,
        "lon": 75.7873,
        "base_temp": 28,
        "temp_range": 15,
        "humidity": 42,
        "pressure": 1007,
        "wind": 14
    },
    "Ahmedabad": {
        "lat": 23.0225,
        "lon": 72.5714,
        "base_temp": 29,
        "temp_range": 14,
        "humidity": 48,
        "pressure": 1008,
        "wind": 13
    }
}

# ------------------------------------------------------------
# 2. Generate hourly timestamps
# ------------------------------------------------------------

dates = pd.date_range(
    end=pd.Timestamp.now().floor("h"),
    periods=24 * 90,
    freq="h"
)

all_data = []

# ------------------------------------------------------------
# 3. Generate weather observations
# ------------------------------------------------------------

for city, config in cities.items():

    for timestamp in dates:

        hour = timestamp.hour
        day_of_year = timestamp.dayofyear

        # Daily temperature cycle
        daily_cycle = np.sin(
            ((hour - 6) / 24) * 2 * np.pi
        )

        # Yearly seasonal cycle
        seasonal_cycle = np.sin(
            ((day_of_year - 80) / 365) * 2 * np.pi
        )

        # Temperature
        temperature = (
            config["base_temp"]
            + config["temp_range"] * 0.5 * daily_cycle
            + 4 * seasonal_cycle
            + np.random.normal(0, 1.2)
        )

        # Humidity
        humidity = (
            config["humidity"]
            - 12 * daily_cycle
            + np.random.normal(0, 5)
        )

        humidity = np.clip(humidity, 20, 95)

        # Pressure
        pressure = (
            config["pressure"]
            + np.random.normal(0, 4)
        )

        # Wind
        wind_speed = (
            config["wind"]
            + 4 * np.sin(hour / 24 * 2 * np.pi)
            + np.random.normal(0, 2)
        )

        wind_speed = max(0.5, wind_speed)

        # Cloud cover
        cloud_cover = (
            45
            + 25 * np.sin(
                (hour + 4) / 24 * 2 * np.pi
            )
            + np.random.normal(0, 15)
        )

        cloud_cover = np.clip(cloud_cover, 0, 100)

        # Rain probability
        rain_probability = (
            8
            + cloud_cover * 0.55
            + humidity * 0.18
        )

        rain_probability = np.clip(
            rain_probability + np.random.normal(0, 5),
            0,
            100
        )

        # Precipitation
        if rain_probability > 65 and np.random.random() < 0.35:
            precipitation = np.random.gamma(2.0, 1.2)
        else:
            precipitation = 0

        # Visibility
        visibility = (
            10
            - (humidity - 50) * 0.035
            - cloud_cover * 0.015
            + np.random.normal(0, 0.5)
        )

        visibility = np.clip(visibility, 1, 15)

        # UV index
        if 7 <= hour <= 17:
            uv_index = (
                max(0, 9 * np.sin(
                    ((hour - 6) / 12) * np.pi
                ))
                * (1 - cloud_cover / 150)
            )
        else:
            uv_index = 0

        uv_index = max(0, uv_index)

        # Wind direction
        wind_direction = (
            np.random.randint(0, 360)
        )

        # AQI
        aqi = (
            70
            + max(0, 65 - humidity) * 0.8
            + cloud_cover * 0.15
            + np.random.normal(0, 15)
        )

        aqi = int(np.clip(aqi, 25, 350))

        # Weather condition
        if precipitation > 2:
            condition = "Rain"
        elif cloud_cover > 75:
            condition = "Cloudy"
        elif cloud_cover > 45:
            condition = "Partly Cloudy"
        else:
            condition = "Clear"

        all_data.append({
            "timestamp": timestamp,
            "city": city,
            "latitude": config["lat"],
            "longitude": config["lon"],
            "temperature": round(temperature, 2),
            "humidity": round(humidity, 2),
            "pressure": round(pressure, 2),
            "wind_speed": round(wind_speed, 2),
            "wind_direction": wind_direction,
            "cloud_cover": round(cloud_cover, 2),
            "precipitation": round(precipitation, 2),
            "precipitation_probability": round(
                rain_probability, 2
            ),
            "visibility": round(visibility, 2),
            "uv_index": round(uv_index, 2),
            "aqi": aqi,
            "condition": condition
        })

# ------------------------------------------------------------
# 4. Create DataFrame
# ------------------------------------------------------------

df = pd.DataFrame(all_data)

# Sort data
df = df.sort_values(
    ["city", "timestamp"]
).reset_index(drop=True)

# ------------------------------------------------------------
# 5. Feature Engineering
# ------------------------------------------------------------

df["hour"] = df["timestamp"].dt.hour

df["day_of_year"] = (
    df["timestamp"].dt.dayofyear
)

df["day_of_week"] = (
    df["timestamp"].dt.dayofweek
)

# Temperature lags
df["temperature_lag_1"] = (
    df.groupby("city")["temperature"]
    .shift(1)
)

df["temperature_lag_3"] = (
    df.groupby("city")["temperature"]
    .shift(3)
)

df["temperature_lag_6"] = (
    df.groupby("city")["temperature"]
    .shift(6)
)

# Target = next hour temperature
df["target_temperature"] = (
    df.groupby("city")["temperature"]
    .shift(-1)
)

# ------------------------------------------------------------
# 6. Remove rows with missing lag/target values
# ------------------------------------------------------------

df = df.dropna().reset_index(drop=True)

# ------------------------------------------------------------
# 7. Save dataset
# ------------------------------------------------------------

os.makedirs("data", exist_ok=True)

output_file = "data/weather_prediction.csv"

df.to_csv(
    output_file,
    index=False
)

# ------------------------------------------------------------
# 8. Display information
# ------------------------------------------------------------

print("=" * 60)
print("WEATHER DATASET GENERATED SUCCESSFULLY")
print("=" * 60)

print(f"File: {output_file}")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

print("\nCities:")
print(df["city"].unique())

print("\nDate Range:")
print(df["timestamp"].min())
print("to")
print(df["timestamp"].max())

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset saved successfully!")