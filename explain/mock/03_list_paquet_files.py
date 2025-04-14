import os
import re
import pandas as pd

# 데이터를 탐색할 루트 경로를 설정함
root_path = 'train'

# 발견된 parquet 파일들의 전체 경로를 저장할 리스트
parquet_files = []

# os.walk를 통해 train 폴더 이하의 모든 파일을 순회함
for dirpath, _, filenames in os.walk(root_path):
    for filename in filenames:
        # parquet 확장자를 가진 파일만 선택함
        if filename.endswith(".parquet"):
            full_path = os.path.join(dirpath, filename)
            parquet_files.append(full_path)

# 파일명에서 테이블 이름을 추출하는 함수 정의
def extract_table_type(filename):
    # 정규표현식을 사용해 'train_테이블명.parquet' 형식에서 테이블명을 추출함
    match = re.search(r"train_([^\.\/\\]+)\.parquet", filename)
    return match.group(1) if match else "UNKNOWN"

# 파일별 메타정보를 저장할 리스트
file_table = []

# 수집한 parquet 파일들에 대해 반복 처리
for path in parquet_files:
    # 파일 이름으로부터 테이블명을 추출함
    table_type = extract_table_type(path)
    # 파일명, 테이블명, 전체 경로를 딕셔너리로 저장
    file_table.append({
        "파일명": os.path.basename(path),  # 경로 없이 파일명만 추출
        "테이블명": table_type,            # 테이블명 추출 결과
        "파일경로": path                   # 전체 파일 경로
    })

# 리스트를 데이터프레임으로 변환함
df = pd.DataFrame(file_table)

# 결과를 CSV 파일로 저장함
df.to_csv("output/train_parquet_file_map.csv", index=False)

# 저장 완료 메시지를 출력함
print("File map for train data has been saved to: output/train_parquet_file_map.csv")
