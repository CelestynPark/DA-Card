import pandas as pd
import os

# 병합 분류 정보가 담긴 파일을 불러옴
merge_info = pd.read_csv('output/train_parquet_merge_classification.csv')

# 처리할 월 리스트 정의
months = ['201807', '201808', '201809', '201810', '201811', '201812']
# 1:N 관계로 집계 후 병합할 테이블 목록
one_to_many_targets = ['승인매출정보', '청구정보', '잔액정보']

# 월별 반복 수행
for month in months:
    # 해당 월의 1:1 병합 결과 파일 경로 지정
    base_path = f"output/train_master_{month}.parquet"
    
    # 해당 파일이 존재하지 않으면 스킵
    if not os.path.exists(base_path):
        print(f"[SKIP] {month}: No 1:1 merged file found → {base_path}")
        continue

    print(f"\n[INFO] Starting 1:N merge for {month}...")

    # 기준 데이터프레임 로드
    master_df = pd.read_parquet(base_path)
    master_df['ID'] = master_df['ID'].astype(str)

    # 각 1:N 병합 테이블에 대해 반복 처리
    for table_name in one_to_many_targets:
        # 해당 월, 테이블명이 일치하는 파일 검색
        match = merge_info[
            (merge_info['테이블명'] == table_name) &
            (merge_info['파일명'].str.startswith(month))
        ]
        if match.empty:
            print(f"  - Skipped: {table_name} not found for {month}")
            continue

        # 해당 테이블을 로드
        table_path = match['파일경로'].values[0]
        df = pd.read_parquet(table_path, engine='pyarrow')
        df.columns = df.columns.str.strip()
        df['ID'] = df['ID'].astype(str)

        # 숫자형 컬럼만 추출 (ID 제외)
        numeric_cols = df.select_dtypes(include='number').columns.drop('ID', errors='ignore')
        if len(numeric_cols) == 0:
            print(f"  - Skipped: No numeric columns to aggregate in {table_name}")
            continue

        # ID를 기준으로 평균, 합계, 건수 집계
        agg_df = df.groupby('ID')[numeric_cols].agg(['mean', 'sum', 'count']).reset_index()

        # 멀티 인덱스 컬럼을 단일 컬럼명으로 변환
        agg_df.columns = ['ID'] + [f"{col}_{agg}" for col in numeric_cols for agg in ['mean', 'sum', 'count']]

        # 집계된 테이블을 기준 테이블에 병합함
        master_df = master_df.merge(agg_df, on='ID', how='left')
        print(f"  + Merged aggregated data from {table_name}")

    # 최종 병합 결과 저장
    output_path = f"output/train_master_{month}_final.parquet"
    master_df.to_parquet(output_path, index=False)
    print(f"[INFO] Final merged data saved to: {output_path}")
