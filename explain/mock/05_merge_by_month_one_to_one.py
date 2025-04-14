import pandas as pd

# 병합할 파일 목록과 병합 방식이 정의된 CSV를 불러옴
merge_info = pd.read_csv('output/train_parquet_merge_classification.csv')

# 처리할 대상 월 리스트
months = ['201807', '201808', '201809', '201810', '201811', '201812']

# 1:1 병합 대상으로 지정된 테이블 목록
one_to_one_targets = ['신용정보', '채널정보', '마케팅정보', '성과정보']

# 월별로 병합 작업을 반복함
for month in months:
    print(f"\n[INFO] Processing data for {month}...")

    # 기준이 되는 회원정보 테이블 경로를 찾음
    member_path = merge_info[
        (merge_info['테이블명'] == '회원정보') &
        (merge_info['파일명'].str.startswith(month))
    ]['파일경로'].values[0]

    # 회원정보 데이터를 불러오고, 컬럼명을 정리함
    member_df = pd.read_parquet(member_path, engine='pyarrow')
    member_df.columns = member_df.columns.str.strip()
    # ID 컬럼을 문자열로 통일함 (타 테이블과 병합을 위함)
    member_df['ID'] = member_df['ID'].astype(str)
    # 기준이 되는 데이터프레임을 복사하여 master로 사용함
    master_df = member_df.copy()

    # 1:1 병합 대상 테이블들을 하나씩 순회함
    for table_name in one_to_one_targets:
        # 해당 월과 테이블명이 일치하는 파일을 탐색함
        match = merge_info[
            (merge_info['테이블명'] == table_name) &
            (merge_info['파일명'].str.startswith(month))
        ]

        # 해당 테이블이 존재하지 않으면 스킵
        if match.empty:
            print(f"  - Skipped: {table_name} not found for {month}")
            continue

        # 테이블 경로를 추출하고 불러옴
        table_path = match['파일경로'].values[0]
        df = pd.read_parquet(table_path, engine='pyarrow')
        df.columns = df.columns.str.strip()
        df['ID'] = df['ID'].astype(str)

        # 병합 시 중복되는 컬럼 제거 (ID 제외)
        overlap_cols = [col for col in df.columns if col in master_df.columns and col != 'ID']
        if overlap_cols:
            df = df.drop(columns=overlap_cols)
            print(f"  - Dropped duplicate columns in {table_name}: {overlap_cols}")

        # 기준 테이블(master)과 현재 테이블을 ID 기준으로 left join 방식 병합
        master_df = master_df.merge(df, on='ID', how='left')
        print(f"  + Merged {table_name}")

    # 병합 결과를 parquet 형식으로 저장함
    output_path = f"output/train_master_{month}.parquet"
    master_df.to_parquet(output_path, index=False)
    print(f"[INFO] Merged data saved to: {output_path}")
