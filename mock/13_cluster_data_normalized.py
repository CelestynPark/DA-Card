import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

plt.rcParams['font.family'] = "Malgun Gothic"

# 평균 테이블 불러오기
df = pd.read_csv("output/cluster_pipeline/cluster_feature_summary.csv", index_col=0)

# MinMax 스케일링 (0~1)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled, columns=df.columns, index=df.index)

# Radar Chart 함수
def plot_radar(data, title, output_path):
    labels = data.columns.tolist()
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

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
plot_radar(scaled_df, "정규화된 클러스터 고객 특성 Radar Chart", "output/cluster_pipeline/cluster_radar_chart_scaled.png")
print("[INFO] 정규화된 Radar Chart 저장 완료")
