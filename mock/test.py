# import pandas as pd

# df = pd.read_parquet('test/1.회원정보/201807_test_회원정보.parquet')

# df.columns = df.columns.str.strip()

# print(df.head())


import pandas as pd
import os

path = 'train/8.성과정보'
for file in sorted(os.listdir(path)):
    if file.endswith(".parquet"):
        df = pd.read_parquet(os.path.join(path, file))
        df.columns = df.columns.str.strip()
        print(f"{file}: {'cluster' in df.columns}")