import pandas as pd

merge_info = pd.read_csv('output/train_parquet_merge_classification.csv')

months = ['201807', '201808', '201809', '201810', '201811', '201812']

one_to_one_targets = ['신용정보', '채널정보', '마케팅정보', '성과정보']

for month in months:
    print(f"\n[INFO] Processing {month}...")

    member_path = merge_info[(merge_info['테이블명'] == '회원정보') & (merge_info['파일명'].str.startswith(month))]['파일경로'].values[0]
    member_df = pd.read_parquet(member_path, engine='pyarrow')
    member_df.columns = member_df.columns.str.strip()
    member_df['ID'] = member_df['ID'].astype(str)
    master_df = member_df.copy()

    for table_name in one_to_one_targets:
        match = merge_info[(merge_info['테이블명'] == table_name) & (merge_info['파일명'].str.startswith(month))]
        if match.empty:
            print(f"  - Skip: {table_name} not found for {month}")
            continue

        table_path = match['파일경로'].values[0]
        df = pd.read_parquet(table_path, engine='pyarrow')
        df.columns = df.columns.str.strip()
        df['ID'] = df['ID'].astype(str)

        overlap_cols = [col for col in df.columns if col in master_df.columns and col != 'ID']
        if overlap_cols:
            df = df.drop(columns=overlap_cols)
            print(f"  - Dropped duplicate columns from {table_name}: {overlap_cols}")
        
        master_df = master_df.merge(df, on='ID', how='left')
        print(f"  + Merged {table_name}")

    output_path = f"output/train_master_{month}.parquet"
    master_df.to_parquet(output_path, index=False)
    master_df.to_csv(f"output/train_master_{month}.csv", index=False)
    print(f"[INFO] Saved: {output_path}")
