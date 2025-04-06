import pandas as pd
import os

merge_info = pd.read_csv('output/train_parquet_merge_classification.csv')

member_row = merge_info[merge_info['테이블명'].str.contains("회원정보")].iloc[0]
member_df = pd.read_parquet(member_row['파일경로'], engine='pyarrow')
member_df.columns = member_df.columns.str.strip()
key_col = next((col for col in member_df.columns if 'cust' in col.lower() or '고객' in col), member_df.columns[0])
master_df = member_df.copy()

for _, row in merge_info.iterrows():
    table_name = row['테이블명']
    path = row['파일경로']
    merge_type = row['병합 방식']

    if table_name == '회원정보':
        continue

    print(f"처리 중: {table_name} ({merge_type})")

    df = pd.read_parquet(path, engine='pyarrow')
    df.columns = df.columns.str.strip()
    join_key = next((col for col in df.columns if 'cust' in col.lower() or '고객' in col), df.columns[0])

    if merge_type == '1:1 병합':
        master_df = master_df.merge(df, on=join_key, how='left')
    elif merge_type == '1:N 집계 후 병합':
        numeric_cols = df.select_dtypes(include='number').columns.drop(join_key, errors='ignore')
        agg_df = df.groupby(join_key)[numeric_cols].agg(['mean', 'sum', 'count']).reset_index()
        agg_df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in agg_df.columns]
        master_df = master_df.merge(agg_df, left_on=key_col, right_on=join_key, how='left')
    else:
        print(f'병합 방식 확인 필요: {table_name}')

master_df.to_csv('output/train_master_table.parquet', index=False)
master_df.to_csv('output/train_master_table.csv', index=False)