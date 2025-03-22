import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

# ✅ Load the dataset
df = pd.read_csv("data/Indoor Air Pollution Data.csv", low_memory=False)

# ✅ Remove Unnamed Columns & Date
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
df.drop(columns=["Date"], inplace=True, errors="ignore")

# ✅ Print cleaned column names
print("🔹 Cleaned Column Names:", df.columns.tolist())

# ✅ Rename columns for consistency
df.rename(columns={
    "Temp": "Temperature",
    "PM2.5": "PM2_5",
    "CO": "Carbon_Monoxide",
}, inplace=True)

# ✅ Define sensor columns based on available data
sensor_columns = ["NH3", "NO2", "Carbon_Monoxide", "PM2_5", "Temperature", "Pressure", "Humidity", "O3"]

# ✅ Convert sensor columns to numeric & handle errors
for col in sensor_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to float, replace invalid data with NaN

# ✅ Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# ✅ Normalize sensor data
scaler = StandardScaler()
df[sensor_columns] = scaler.fit_transform(df[sensor_columns])

# ✅ Dimensionality Reduction (Optional)
pca = PCA(n_components=2)
df[["PCA1", "PCA2"]] = pca.fit_transform(df[sensor_columns])

# ✅ Clustering (Unsupervised Learning)
kmeans = KMeans(n_clusters=5, random_state=42)
df["Cluster_KMeans"] = kmeans.fit_predict(df[sensor_columns])

dbscan = DBSCAN(eps=0.5, min_samples=5)
df["Cluster_DBSCAN"] = dbscan.fit_predict(df[sensor_columns])

# ✅ Assign Air Quality Categories
df["Air_Quality"] = df["Cluster_KMeans"].map({
    0: "Good", 1: "Moderate", 2: "Poor", 3: "Very Poor", 4: "Hazardous"
})

# ✅ Save Preprocessed Data
df.to_csv("data/cleaned_indoor_air_pollution.csv", index=False)
print("✅ Data Preprocessing Completed! Cleaned data saved.")
