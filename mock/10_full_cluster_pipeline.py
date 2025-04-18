import pandas as pd
import glob
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 설정
sample_per_month = 5000
n_clusters = 5
output_dir = "output/cluster_pipeline"
os.makedirs(output_dir, exist_ok=True)

# Step 1: 월별 파일에서 샘플링 (ID 포함)
monthly_files = sorted(glob.glob("output/train_master_20*_final.parquet"))
sample_list = []

for path in monthly_files:
    month = os.path.basename(path).split('_')[2]
    print(f"[INFO] 샘플링 중: {month}")
    df = pd.read_parquet(path)
    df['ID'] = df['ID'].astype(str)
    sample_df = df.sample(n=sample_per_month, random_state=42)
    sample_list.append(sample_df)

sample_all = pd.concat(sample_list, ignore_index=True)
print(f"[INFO] 전체 샘플 크기: {sample_all.shape}")

# Step 2: 수치형 컬럼만 추출 (ID 유지)
sample_all['ID'] = sample_all['ID'].astype(str)
numeric_df = sample_all.select_dtypes(include='number').copy()

# 결측치 30% 이상 컬럼 제거
missing_ratio = numeric_df.isnull().mean()
valid_cols = missing_ratio[missing_ratio < 0.3].index.tolist()
numeric_df = numeric_df[valid_cols]

# 결측값 0으로 대체
numeric_df.fillna(0, inplace=True)

# Step 3: 스케일링
scaler = StandardScaler()
scaled = scaler.fit_transform(numeric_df)
scaled_df = pd.DataFrame(scaled, columns=numeric_df.columns)
scaled_df['ID'] = sample_all['ID'].values  # ID 추가

# Step 4: 클러스터링
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
scaled_df['cluster'] = kmeans.fit_predict(scaled)

# Step 5: 원본 데이터와 병합
# → 원본 통합
raw_dfs = []
for path in monthly_files:
    df = pd.read_parquet(path)
    df['ID'] = df['ID'].astype(str)
    raw_dfs.append(df)
raw_all = pd.concat(raw_dfs, ignore_index=True)

# 병합
merged_df = raw_all.merge(scaled_df[['ID', 'cluster']], on='ID', how='inner')
print(f"[INFO] 최종 병합 크기: {merged_df.shape}")

# 저장
merged_df.to_parquet(f"{output_dir}/clustered_with_features.parquet", index=False)
print(f"[INFO] 저장 완료: {output_dir}/clustered_with_features.parquet")
