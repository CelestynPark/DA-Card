import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = "Malgun Gothic"

# 데이터 불러오기
df = pd.read_parquet("output/cluster_pipeline/clustered_with_features.parquet")

# 주요 피처 선정 (존재하는 컬럼 기준)
features = [
    "이용금액_R3M_신용체크", "이용카드수_신용", "소지카드수_유효_신용",
    "최종이용일자_기본", "입회일자_신용", "최종카드발급일자",
    "연체일수_최근_mean", "연체일수_B1M_mean", "연체일수_B2M_mean"
]

# 존재하지 않는 컬럼 제거
features = [f for f in features if f in df.columns]

# 연체 컬럼 정제
for col in features:
    if "연체일수" in col:
        df[col] = df[col].apply(lambda x: 0 if x < -999000 else abs(x))

# 클러스터별 평균
cluster_means = df.groupby("cluster")[features].mean()

# 저장용 테이블
cluster_means.to_csv("output/cluster_pipeline/cluster_feature_summary.csv")
print("[INFO] 평균 테이블 저장 완료: cluster_feature_summary.csv")

# Radar Chart
def plot_radar(data, title, output_path):
    labels = data.columns.tolist()
    num_vars = len(labels)

    # 각 축에 해당하는 각도 계산
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 시작점으로 다시 돌아오기

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for idx, row in data.iterrows():
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, label=f'Cluster {idx}')
        ax.fill(angles, values, alpha=0.1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    plt.title(title)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

# 실행
plot_radar(cluster_means, "클러스터별 고객 프로파일 Radar Chart", "output/cluster_pipeline/cluster_radar_chart.png")
print("[INFO] Radar Chart 저장 완료: cluster_radar_chart.png")
