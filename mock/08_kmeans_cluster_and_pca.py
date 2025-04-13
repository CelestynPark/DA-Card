import pandas as pd
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

sample_files = sorted(glob.glob("output/cluster_samples/scaled_input_*.parquet"))
dfs = [pd.read_parquet(f) for f in sample_files]
df = pd.concat(dfs, ignore_index=True)
print(f"[INFO] Size of merged sampels : {df.shape}")

scaler = StandardScaler()
scaled = scaler.fit_transform(df)

k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(scaled)

df_clustered = df.copy()
df_clustered["cluster"] = clusters

pca = PCA(n_components=2)
reduced = pca.fit_transform(scaled)

plt.figure(figsize=(10, 7))
for i in range(k):
    plt.scatter(reduced[clusters == i, 0], reduced[clusters == i, 1], label=f"Cluster {i}", s=10)
plt.title("KMeans Clustering (PCA 2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("output/kmeans_pca_plot.png")
plt.show()

df_clustered.to_parquet("output/clustered_data.parquet", index=False)
print("[INFO] Clustering result has completedly saved: output/clustered_data.parquet")
print("[INFO] Visualized result has completedly saved: output/kmeans_pca_plot.png")