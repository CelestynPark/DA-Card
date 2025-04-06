import pandas as pd

df = pd.read_parquet("test/1.회원정보/201807_test_회원정보.parquet")

print(df.head(50))