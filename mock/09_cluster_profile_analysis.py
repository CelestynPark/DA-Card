import pandas as pd

df = pd.read_parquet("output/cluster_data.parquet")

print("[INFO] Size of samples per cluster:")
print(df['cluster'].value_counts().sort_index())

cluster_means = df.groupby("cluster").mean().T

for cluster_id in cluster_means.columns:
    print(f"\n[Main features in cluster {cluster_id}]")
    top_features = cluster_means[cluster_id].sort_values(ascending=False).head(5)
    bottom_features = cluster_means[cluster_id].sort_values(ascending=True).head(5)

    print(" -> Top Features:")
    print(top_features)
    print(" -> Bottom Features:")
    print(bottom_features)