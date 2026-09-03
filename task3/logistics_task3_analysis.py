import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reproducible hypothetical logistics data
np.random.seed(42)
n = 500
cities = ["Chennai", "Bangalore", "Hyderabad", "Mumbai", "Delhi", "Pune", "Kochi", "Coimbatore"]
modes = ["Road", "Rail", "Air", "Sea"]

origins = np.random.choice(["Chennai", "Bangalore", "Hyderabad", "Mumbai"], n)
destinations = np.random.choice(cities, n)
transport_mode = np.random.choice(modes, n, p=[0.55, 0.20, 0.15, 0.10])
shipment_volume = np.random.randint(20, 501, n)
distance = np.random.randint(50, 2201, n)

base_days = {"Road": 3.6, "Rail": 4.5, "Air": 1.4, "Sea": 7.0}
cost_per_km = {"Road": 2.1, "Rail": 1.35, "Air": 6.2, "Sea": 0.95}

delivery_time = np.maximum(
    0.5,
    np.array([base_days[m] for m in transport_mode])
    + distance / 650
    + np.random.normal(0, 1.0, n)
)

transport_cost = np.maximum(
    250,
    distance * np.array([cost_per_km[m] for m in transport_mode])
    + shipment_volume * np.random.uniform(0.8, 1.25, n)
      * np.array([0.8 if m in ["Road", "Rail"] else 1.2 if m == "Air" else 0.55
                  for m in transport_mode])
    + np.random.normal(0, 350, n)
)

threshold = {"Road": 6.5, "Rail": 7.0, "Air": 4.0, "Sea": 10.0}
on_time = ["Yes" if dt <= threshold[m] else "No"
           for dt, m in zip(delivery_time, transport_mode)]

df = pd.DataFrame({
    "Shipment_ID": [f"SHP{10001+i}" for i in range(n)],
    "Origin": origins,
    "Destination": destinations,
    "Transport_Mode": transport_mode,
    "Shipment_Volume_kg": shipment_volume,
    "Distance_km": distance,
    "Delivery_Time_days": np.round(delivery_time, 2),
    "Transportation_Cost_INR": np.round(transport_cost, 2),
    "On_Time": on_time
})

# EDA
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())
print("\nDescriptive statistics:")
print(df.describe())
print("\nCorrelation matrix:")
print(df[["Shipment_Volume_kg", "Distance_km",
          "Delivery_Time_days", "Transportation_Cost_INR"]].corr())

print("\nAverage delivery time:", round(df["Delivery_Time_days"].mean(), 2))
print("Median delivery time:", round(df["Delivery_Time_days"].median(), 2))
print("Average transportation cost:", round(df["Transportation_Cost_INR"].mean(), 2))
print("On-time rate:", round((df["On_Time"].eq("Yes").mean())*100, 2), "%")

# Visualizations
df["Transport_Mode"].value_counts().reindex(modes).plot(kind="bar")
plt.title("Shipment Count by Transportation Mode")
plt.xlabel("Transportation Mode")
plt.ylabel("Number of Shipments")
plt.tight_layout()
plt.show()

plt.hist(df["Delivery_Time_days"], bins=25, edgecolor="black")
plt.title("Distribution of Delivery Time")
plt.xlabel("Delivery Time (days)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

df.groupby("Transport_Mode")["Transportation_Cost_INR"].mean().reindex(modes).plot(kind="bar")
plt.title("Average Transportation Cost by Mode")
plt.xlabel("Transportation Mode")
plt.ylabel("Average Cost (INR)")
plt.tight_layout()
plt.show()

plt.scatter(df["Distance_km"], df["Transportation_Cost_INR"], alpha=0.55)
plt.title("Distance vs Transportation Cost")
plt.xlabel("Distance (km)")
plt.ylabel("Transportation Cost (INR)")
plt.tight_layout()
plt.show()

df.boxplot(column="Delivery_Time_days", by="Transport_Mode")
plt.suptitle("")
plt.title("Delivery Time by Transportation Mode")
plt.xlabel("Transportation Mode")
plt.ylabel("Delivery Time (days)")
plt.tight_layout()
plt.show()

corr = df[["Shipment_Volume_kg", "Distance_km",
           "Delivery_Time_days", "Transportation_Cost_INR"]].corr()
plt.imshow(corr.values, aspect="auto")
plt.xticks(range(4), ["Volume", "Distance", "Delivery Time", "Cost"], rotation=30)
plt.yticks(range(4), ["Volume", "Distance", "Delivery Time", "Cost"])
for i in range(4):
    for j in range(4):
        plt.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center")
plt.title("Correlation Matrix")
plt.colorbar()
plt.tight_layout()
plt.show()
