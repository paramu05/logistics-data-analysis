5. Python Code Illustrations

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
# 5.1 Loading and Preparing Data

import pandas as pd
import numpy as np

df = pd.read_parquet("yellow_tripdata_2026-01.parquet")

df["pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
df["dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

df["duration_min"] = (
df["dropoff_datetime"] - df["pickup_datetime"]
).dt.total_seconds() / 60

df["hour"] = df["pickup_datetime"].dt.hour
df["weekday"] = df["pickup_datetime"].dt.day_name()

# Basic quality filters
df = df[df["duration_min"] > 0]
df = df[df["trip_distance"] > 0]
df = df.drop_duplicates()

print(df[["duration_min", "trip_distance", "hour"]].describe())
#5.2 Feature Engineering for Demand Analysis
# Number of trips by pickup zone and hour
demand = (
df.groupby(["PULocationID", "hour"])
.size()
.reset_index(name="trip_count")
)

# Historical average demand by zone
zone_summary = (
df.groupby("PULocationID")
.agg(
avg_distance=("trip_distance", "mean"),
avg_duration=("duration_min", "mean"),
trip_count=("PULocationID", "size")
)
.reset_index()
)
# 5.3 Regression Example: Predict Trip Duration

from sklearn.metrics import mean_absolute_error, root_mean_squared_error

features = ["trip_distance", "hour", "PULocationID", "DOLocationID"]
model_df = df[features + ["duration_min"]].dropna()

X = model_df[features]
y = model_df["duration_min"]

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
n_estimators=200,
random_state=42,
n_jobs=-1
)
model.fit(X_train, y_train)

pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = root_mean_squared_error(y_test, pred)

print("MAE:", mae)
print("RMSE:", rmse)
For a time-sensitive production forecasting problem, a chronological train/validation/test split is preferable to a random split. The exact feature set should also avoid leakage—for example, a future value that would not be known when the delivery decision is made.
# 5.4 Clustering Delivery Zones
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

cluster_features = zone_summary[
["trip_count", "avg_distance", "avg_duration"]
].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_features)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
zone_summary["cluster"] = kmeans.fit_predict(X_scaled)

print(zone_summary.groupby("cluster").mean(numeric_only=True))
# 5.5 Route Optimization Pseudocode
# PSEUDOCODE

orders = load_delivery_orders()
vehicles = load_vehicle_capacity()
travel_time = build_travel_time_matrix(orders)

create_routing_model()

for vehicle in vehicles:
add_vehicle_capacity_constraint(vehicle.capacity)

for order in orders:
add_delivery_stop(order.location)
add_time_window(order.promised_start, order.promised_end)

set_objective(
minimize_total_distance
+ penalty_for_late_delivery
+ penalty_for_unused_capacity
)

solution = solver.solve()

export_routes(solution)

