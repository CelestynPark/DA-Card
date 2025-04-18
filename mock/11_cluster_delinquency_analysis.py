import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
df = pd.read_parquet("output/cluster_pipeline/clustered_with_features.parquet")

# 연체 관련 컬럼
late_cols = [
    "연체일수_최근_mean",
    "연체일수_B1M_mean",
    "연체일수_B2M_mean"
]

# 결측치(-999999 등) → 0, 나머지는 절댓값 처리
for col in late_cols:
    df[col] = df[col].apply(lambda x:
        0 if x < -999000 else abs(x)
    )

# 클러스터별 평균 연체일수 계산
mean_lates = df.groupby("cluster")[late_cols].mean()

# 시각화
mean_lates.plot(kind='bar', figsize=(10, 6))
plt.title("Cluster별 평균 연체일수 (최근 / B1M / B2M)")
plt.ylabel("평균 연체일수 (일)")
plt.xlabel("클러스터 ID")
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("output/cluster_pipeline/cluster_delinquency_summary.png")
plt.show()
