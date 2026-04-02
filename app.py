import streamlit as st
import pandas as pd
import numpy as np
import time

from scripts.cluster_experiments import DataLoader
from scripts.pso_kmeans import PSOKMeans
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scripts.visualize import plot_radar_chart, plot_feature_heatmap, generate_education_diagnosis

st.set_page_config(page_title="学生成绩聚类分析系统", page_icon="🎓", layout="wide")

st.title("🎓 基于改进聚类算法的学生成绩分析系统")
st.markdown("""
> **毕业论文原型系统**
>
> 本系统集成了特征工程、PSO-KMeans 融合聚类算法、群体画像可视化以及学业诊断闭环。
""")

st.sidebar.header("⚙️ 实验参数配置")
data_path = st.sidebar.text_input("数据集路径", "data/student-mat.csv")
k_value = st.sidebar.slider("聚类簇数 (k)", min_value=2, max_value=10, value=3)
algo_choice = st.sidebar.selectbox(
    "选择聚类算法",
    ("K-Means (Random Init)", "K-Means++", "PSO-KMeans (粒子群优化)")
)

@st.cache_data
def load_and_preprocess(path):
    loader = DataLoader()
    try:
        df_raw = loader.load(path)
        df_processed = loader.prepare_features(df_raw)
        return df_raw, df_processed
    except Exception as e:
        st.error(f"数据加载失败: {e}")
        return None, None

df_raw, df_processed = load_and_preprocess(data_path)

if df_processed is not None:
    st.subheader("1. 数据概览与特征工程")
    with st.expander("查看原始数据与衍生特征", expanded=False):
        st.dataframe(df_processed.head())
        st.caption("注：已自动添加 `grade_volatility` (成绩波动率), `grade_trend` (成绩线性趋势) 等衍生特征。")

    st.subheader(f"2. 聚类模型训练: {algo_choice}")

    if st.button("🚀 开始训练"):
        with st.spinner("模型训练中，请稍候..."):
            start_time = time.time()

            if algo_choice == "PSO-KMeans (粒子群优化)":
                model = PSOKMeans(n_clusters=k_value, pso_max_iter=30, random_state=42)
            else:
                init_method = "random" if "Random" in algo_choice else "k-means++"
                model = KMeans(n_clusters=k_value, init=init_method, random_state=42)

            labels = model.fit_predict(df_processed)
            elapsed_time = time.time() - start_time

            sse = model.inertia_
            sil = silhouette_score(df_processed, labels)
            db = davies_bouldin_score(df_processed, labels)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("耗时 (秒)", f"{elapsed_time:.3f}")
            col2.metric("SSE (误差平方和)", f"{sse:.2f}")
            col3.metric("Silhouette (轮廓系数)", f"{sil:.4f}", help="越接近 1 越好")
            col4.metric("Davies-Bouldin Index", f"{db:.4f}", help="越小越好")

            if hasattr(model, 'cluster_centers_'):
                centroids = model.cluster_centers_
            else:
                centroids = df_processed.groupby(labels).mean().values

            centroids_df = pd.DataFrame(centroids, columns=df_processed.columns)

            st.markdown("---")
            st.subheader("3. 群体画像可视化")

            tab1, tab2 = st.tabs(["📊 雷达图 (群体特征对比)", "🔥 热力图 (特征强度分布)"])

            with tab1:
                fig_radar = plot_radar_chart(centroids_df)
                st.pyplot(fig_radar)

            with tab2:
                fig_heatmap = plot_feature_heatmap(centroids_df)
                st.pyplot(fig_heatmap)

            st.markdown("---")
            st.subheader("4. 学业诊断与教学干预策略")

            diagnoses = generate_education_diagnosis(centroids_df)

            for diag in diagnoses:
                with st.container():
                    st.markdown(f"### 🏷️ 群体 {diag['cluster_id']}: **{diag['label']}**")
                    st.info(f"**画像描述**: {diag['description']}")
                    st.success(f"**教学干预建议**: {diag['suggestion']}")

            st.balloons()
