import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA


# Data Cleaning & Preprocessing for Air Quality

'''This script preprocesses indoor air pollution data by cleaning, transforming, and normalizing sensor readings. 
It also applies Principal Component Analysis (PCA) and clustering techniques (K-Means and DBSCAN) to analyze air quality'''

### Data Cleaning: Removed unnecessary columns, handled missing values.

# Feature Engineering: Renamed columns, standardized sensor readings.

# Dimensionality Reduction: Applied PCA for better visualization.

# Clustering: Used K-Means and DBSCAN for air quality classification.

# Labeling: Assigned meaningful air quality categories.

# Export: Saved preprocessed data for further use in modeling.

df = pd.read_csv("data/Indoor Air Pollution Data.csv", low_memory=False)

df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
df.drop(columns=["Date"], inplace=True, errors="ignore")

print("🔹 Cleaned Column Names:", df.columns.tolist())

df.rename(columns={
    "Temp": "Temperature",
    "PM2.5": "PM2_5",
    "CO": "Carbon_Monoxide",
}, inplace=True)

sensor_columns = ["NH3", "NO2", "Carbon_Monoxide", "PM2_5", "Temperature", "Pressure", "Humidity", "O3"]

for col in sensor_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.fillna(df.mean(numeric_only=True), inplace=True)

scaler = StandardScaler()
df[sensor_columns] = scaler.fit_transform(df[sensor_columns])

pca = PCA(n_components=2)
df[["PCA1", "PCA2"]] = pca.fit_transform(df[sensor_columns])

kmeans = KMeans(n_clusters=5, random_state=42)
df["Cluster_KMeans"] = kmeans.fit_predict(df[sensor_columns])

dbscan = DBSCAN(eps=0.5, min_samples=5)
df["Cluster_DBSCAN"] = dbscan.fit_predict(df[sensor_columns])

df["Air_Quality"] = df["Cluster_KMeans"].map({
    0: "Good", 1: "Moderate", 2: "Poor", 3: "Very Poor", 4: "Hazardous"
})

df.to_csv("data/cleaned_indoor_air_pollution.csv", index=False)
print("✅ Data Preprocessing Completed! Cleaned data saved.")
