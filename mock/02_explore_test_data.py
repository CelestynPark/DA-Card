import os
import pandas as pd

root_path = 'test'

data_overview = []

for dirpath, dirnames, filenames in os.walk(root_path):
    for filename in filenames:
        if filename.endswith(".parquet"):
            full_path = os.path.join(dirpath, filename)
            # print(full_path)
            try:
                df = pd.read_parquet(full_path, engine="pyarrow")
                df.columns = df.columns.str.strip()
                key_col = next((col for col in df.columns if 'ID' in col), None)

                data_overview.append({
                    '파일 경로': full_path,
                    '행 수': df.shape[0],
                    '열 수': df.shape[1],
                    'ID': key_col,
                    '내용 일부': ", ".join(df.columns[:5]) + (" ..." if len(df.columns) > 5 else "")
                })
            except Exception as e:
                print(f"Error: {filename} - {e}")

overview_df = pd.DataFrame(data_overview)
overview_df.to_csv('output/test_data_overview.csv', index=False)
print('test data overview saved : output/test_data_overview.csv')