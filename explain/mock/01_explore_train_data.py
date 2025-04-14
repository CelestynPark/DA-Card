import os
import pandas as pd

# 분석할 데이터가 들어 있는 최상위 폴더 경로를 지정함
root_path = 'train'

# 각 파일에 대한 개요 정보를 저장할 리스트를 초기화함
data_overview = []

# os.walk를 사용하여 하위 폴더를 포함한 모든 경로의 파일들을 탐색함
for dirpath, dirnames, filenames in os.walk(root_path):
    for filename in filenames:
        # parquet 파일만 대상으로 처리함
        if filename.endswith(".parquet"):
            # 파일의 전체 경로를 구성함
            full_path = os.path.join(dirpath, filename)
            try:
                # parquet 파일을 읽어서 DataFrame으로 불러옴
                df = pd.read_parquet(full_path, engine="pyarrow")
                
                # 열 이름에 불필요한 공백이 있을 수 있으므로 양 끝 공백을 제거함
                df.columns = df.columns.str.strip()

                # 'ID'라는 단어가 포함된 첫 번째 열을 식별자로 간주함
                key_col = next((col for col in df.columns if 'ID' in col), None)

                # 현재 파일의 개요 정보를 정리하여 리스트에 추가함
                data_overview.append({
                    '파일 경로': full_path,                      # 파일의 전체 경로
                    '행 수': df.shape[0],                       # 데이터프레임의 행 개수
                    '열 수': df.shape[1],                       # 열 개수
                    'ID': key_col,                              # 식별자로 판단되는 컬럼
                    '내용 일부': ", ".join(df.columns[:5]) +    # 앞 5개의 컬럼명만 보여줌
                                (" ..." if len(df.columns) > 5 else "")
                })
            except Exception as e:
                # 오류가 발생한 경우 파일명과 함께 에러 메시지를 출력함
                print(f"Error: {filename} - {e}")

# 수집된 모든 개요 정보를 하나의 DataFrame으로 변환함
overview_df = pd.DataFrame(data_overview)

# 최종 결과를 CSV 파일로 저장함
overview_df.to_csv('output/train_data_overview.csv', index=False)

# 저장 완료 메시지를 출력함
print('Train data overview has been saved to: output/train_data_overview.csv')
