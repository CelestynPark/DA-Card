import pandas as pd
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 각 월에서 샘플링한 데이터 파일들을 읽어 하나로 합침
sample_files = sorted(glob.glob("output/cluster_samples/scaled_input_*.parquet"))
dfs = [pd.read_parquet(f) for f in sample_files]
df = pd.concat(dfs, ignore_index=True)
print(f"[INFO] Merged sample shape: {df.shape}")

# 전체 데이터를 정규화하여 클러스터링에 적합한 형태로 변환함
scaler = StandardScaler()
scaled = scaler.fit_transform(df)

# 클러스터 수를 5로 설정하고 KMeans 알고리즘을 적용함
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(scaled)

# 클러스터 결과를 원본 데이터프레임에 추가함
df_clustered = df.copy()
df_clustered["cluster"] = clusters

# 고차원 데이터를 시각화를 위해 2차원으로 축소함 (PCA 사용)
pca = PCA(n_components=2)
reduced = pca.fit_transform(scaled)

# 2차원 축소 결과를 기반으로 클러스터별 분포를 시각화함
plt.figure(figsize=(10, 7))
for i in range(k):
    plt.scatter(
        reduced[clusters == i, 0],  # 첫 번째 주성분
        reduced[clusters == i, 1],  # 두 번째 주성분
        label=f'Cluster {i}', s=10  # 클러스터별 라벨과 점 크기
    )
plt.title("KMeans Clustering (PCA 2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("output/kmeans_pca_plot.png")
plt.show()

# 클러스터링 결과를 저장하고 시각화 결과도 함께 저장함
df_clustered.to_parquet("output/clustered_data.parquet", index=False)
print("[INFO] Clustered data saved to: output/clustered_data.parquet")
print("[INFO] Visualization saved to: output/kmeans_pca_plot.png")
