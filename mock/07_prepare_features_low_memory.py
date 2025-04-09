import pandas as pd
import glob
from sklearn.preprocessing import StandardScaler
import os

output_dir = "output/cluster_samples"
os.makedirs(output_dir, exist_ok=True)

monthly_files = sorted(glob.glob("output/train_master_20*_final.parquet"))

for path in monthly_files:
    month = os.path.basename(path).split("_")[2]
    print(f"[INFO] processing: {month}")

    df = pd.read_parquet(path)
    df['ID'] = df['ID'].astype(str)

    sample_df = df.sample(n=10000, random_state=42)

    numeric_df = sample_df.select_dtypes(include='number').copy()

    threshold = 0.3
    missing_ratio = numeric_df.isnull().mean()
    valid_cols = missing_ratio[missing_ratio < threshold].index.tolist()
    numeric_df = numeric_df[valid_cols]

    numeric_df.fillna(0, inplace=True)

    numeric_df.to_parquet(f"{output_dir}/scaled_input_{month}.parquet", index=False)
    print(f"  -> {output_dir}/scaled_input_{month}.parquet saved")

print("\n All monthly sampled data are saved")