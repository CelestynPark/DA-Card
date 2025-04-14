import pandas as pd

# 클러스터링 결과 파일을 불러옴
df = pd.read_parquet("output/clustered_data.parquet")

# 각 클러스터에 속한 샘플 수를 출력함
print("[INFO] Number of samples per cluster:")
print(df['cluster'].value_counts().sort_index())

# 클러스터별 평균값을 계산하고, 보기 좋게 전치함 (클러스터 → 열, 변수 → 행)
cluster_means = df.groupby('cluster').mean().T

# 각 클러스터에서 평균값이 가장 높은 특성과 가장 낮은 특성을 각각 5개씩 출력함
for cluster_id in cluster_means.columns:
    print(f"\n[Cluster {cluster_id} - Key Features]")

    # 평균값 기준 상위 5개 피처
    top_features = cluster_means[cluster_id].sort_values(ascending=False).head(5)
    print("  ▶ Top features:")
    print(top_features)

    # 평균값 기준 하위 5개 피처
    bottom_features = cluster_means[cluster_id].sort_values(ascending=True).head(5)
    print("  ▶ Bottom features:")
    print(bottom_features)
