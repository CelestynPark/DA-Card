import pandas as pd

# 앞서 생성한 parquet 파일 메타정보 CSV를 불러옴
df = pd.read_csv('output/train_parquet_file_map.csv')

# 1:1 관계로 병합 가능한 테이블 키워드를 정의함
one_to_one_keywords = ['회원정보', '채널정보', '마케팅정보', '신용정보', '성과정보']

# 1:N 관계로 집계 후 병합이 필요한 테이블 키워드를 정의함
one_to_many_keywords = ['승인매출정보', '청구정보', '입금정보', '잔액정보']

# 테이블명을 기반으로 병합 방식을 분류하는 함수 정의
def classify_merge_type(table_name):
    # 1:1 병합 대상 키워드가 포함되어 있으면 해당 병합 방식으로 분류함
    for keyword in one_to_one_keywords:
        if keyword in table_name:
            return '1:1 병합'
    # 1:N 병합 대상 키워드가 포함되어 있으면 해당 병합 방식으로 분류함
    for keyword in one_to_many_keywords:
        if keyword in table_name:
            return '1:N 집계 후 병합'
    # 어떤 키워드에도 해당하지 않으면 수동 확인이 필요함
    return '수동 확인 필요'

# 위에서 정의한 함수로 각 테이블의 병합 방식을 판별하여 새로운 컬럼으로 추가함
df['병합 방식'] = df['테이블명'].apply(classify_merge_type)

# 결과를 새로운 CSV 파일로 저장함
df.to_csv('output/train_parquet_merge_classification.csv', index=False)

# 저장 완료 메시지를 출력함
print("Merge classification has been saved to: output/train_parquet_merge_classification.csv")
