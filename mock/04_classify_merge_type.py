import pandas as pd

df = pd.read_csv('output/train_parquet_file_map.csv')

one_to_one_keywords = ['회원정보', '채널정보', '마케팅정보', '신용정보', '성과정보']
one_to_many_keywords = ['승인매출정보', '청구정보', '입금정보', '잔액정보']

def classify_merge_type(table_name):
    for keyword in one_to_one_keywords:
        if keyword in table_name:
            return '1:1 병합'
    for keyword in one_to_many_keywords:
        if keyword in table_name:
            return '1:N 집계 후 병합'
    return '수동 확인 필요'

df['병합 방식'] = df['테이블명'].apply(classify_merge_type)

df.to_csv('output/train_parquet_merge_classification.csv', index=False)