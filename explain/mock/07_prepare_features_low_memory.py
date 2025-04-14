import pandas as pd
import glob
from sklearn.preprocessing import StandardScaler
import os

# 결과 파일을 저장할 디렉토리를 정의하고, 없으면 생성함
output_dir = "output/cluster_samples"
os.makedirs(output_dir, exist_ok=True)

# 병합이 완료된 월별 parquet 파일들을 정렬된 리스트로 불러옴
monthly_files = sorted(glob.glob("output/train_master_20*_final.parquet"))

# 월별로 반복 처리
for path in monthly_files:
    # 파일명에서 월(YYYYMM)을 추출함
    month = os.path.basename(path).split('_')[2]
    print(f"[INFO] Processing month: {month}")

    # 해당 월의 데이터를 불러오고 ID를 문자열로 통일함
    df = pd.read_parquet(path)
    df['ID'] = df['ID'].astype(str)

    # 1만 개 샘플을 무작위로 선택함 (seed 고정)
    sample_df = df.sample(n=10000, random_state=42)

    # 수치형 컬럼만 추출하여 별도 데이터프레임 생성
    numeric_df = sample_df.select_dtypes(include='number').copy()

    # 결측치 비율이 30% 이상인 컬럼은 제거함
    threshold = 0.3
    missing_ratio = numeric_df.isnull().mean()
    valid_cols = missing_ratio[missing_ratio < threshold].index.tolist()
    numeric_df = numeric_df[valid_cols]

    # 남은 결측값은 0으로 채움
    numeric_df.fillna(0, inplace=True)

    # 정제된 데이터를 parquet 형식으로 저장
    output_path = f"{output_dir}/scaled_input_{month}.parquet"
    numeric_df.to_parquet(output_path, index=False)
    print(f"[INFO] Saved cleaned numeric sample to: {output_path}")

print("\n[INFO] All monthly samples have been saved successfully.")
