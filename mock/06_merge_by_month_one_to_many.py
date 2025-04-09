import pandas as pd
import os

merge_info = pd.read_csv('output/train_parquet_merge_classification.csv')

months = ['201807', '201808', '201809', '201810', '201811', '201812']

one_to_many_targets = ['승인매출정보', '청구정보', '잔액정보']

for month in months:
    base_path = f"output/train_master_{month}.parquet"
    if not os.path.exists(base_path):
        print(f"[SKIP] {month}: 1:1 병합 결과 없음 -> {base_path}")
        continue

    print(f"\n[INFO] Processing {month}...")

    master_df = pd.read_parquet(base_path)
    master_df['ID'] = master_df['ID'].astype(str)

    for table_name in one_to_many_targets:
        match = merge_info[(merge_info['테이블명'] == table_name) & (merge_info['파일명'].str.startswith(month))]
        if match.empty:
            print(f"  - Skip: {table_name} not found for {month}")
            continue

        table_path = match['파일경로'].values[0]
        df = pd.read_parquet(table_path, engine='pyarrow')
        df.columns = df.columns.str.strip()
        df['ID'] = df['ID'].astype(str)

        numeric_cols = df.select_dtypes(include='number').columns.drop('ID', errors='ignore')
        if len(numeric_cols) == 0:
            print(f"  - No numeric columns to aggregate in {table_name}")
            continue

        agg_df = df.groupby('ID')[numeric_cols].agg(['mean', 'sum', 'count']).reset_index()
        agg_df.columns = ['ID'] + [f"{col}_{agg}" for col in numeric_cols for agg in ['mean', 'sum', 'count']]

        master_df = master_df.merge(agg_df, on='ID', how='left')
        print(f"  + Merged aggregated {table_name}")

    output_path = f"output/train_master_{month}_final.parquet"
    master_df.to_parquet(output_path, index=False)
    master_df.to_csv(f"output/train_master_{month}_final.csv", index=False)
    print(f"[INFO] Saved: {output_path}")
