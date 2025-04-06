import os
import re
import pandas as pd

root_path = 'train'

parquet_files = []
for dirpath, _, filenames in os.walk(root_path):
    for filename in filenames:
        if filename.endswith(".parquet"):
            full_path = os.path.join(dirpath, filename)
            parquet_files.append(full_path)

def extract_table_type(filename):
    match = re.search(r"train_([^\.\/\\]+)\.parquet", filename)
    return match.group(1) if match else "UNKNOWN"

file_table = []
for path in parquet_files:
    table_type = extract_table_type(path)
    file_table.append({
        "파일명": os.path.basename(path),
        "테이블명": table_type,
        "파일경로": path
    })

df = pd.DataFrame(file_table)
df.to_csv("output/train_parquet_file_map.csv", index=False)
