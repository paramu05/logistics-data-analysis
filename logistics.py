import pandas as pd
import numpy as np

# Load the logistics dataset
df = pd.read_parquet("yellow_tripdata_2026-01.parquet")

# Convert date columns
df["pickup_datetime"] = pd.to_datetime(
    df["tpep_pickup_datetime"]
)

df["dropoff_datetime"] = pd.to_datetime(
    df["tpep_dropoff_datetime"]
)

# Calculate trip duration in minutes
df["duration_min"] = (
    df["dropoff_datetime"] - df["pickup_datetime"]
).dt.total_seconds() / 60

# Create time features
df["hour"] = df["pickup_datetime"].dt.hour
df["weekday"] = df["pickup_datetime"].dt.day_name()

# Basic data cleaning
df = df[df["duration_min"] > 0]
df = df[df["trip_distance"] > 0]

# Display basic information
print("Number of records:", len(df))
print(df.head())

# Calculate average trip duration
average_duration = df["duration_min"].mean()

print("Average Trip Duration:",
      round(average_duration, 2), "minutes")
