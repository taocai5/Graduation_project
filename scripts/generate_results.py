"""
生成毕业论文实验结果的完整可视化
包含特征工程、算法对比、聚类结果、学业诊断等所有图表
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cluster_experiments import DataLoader
from pso_kmeans import PSOKMeans
from visualize import plot_radar_chart, plot_feature_heatmap, generate_education_diagnosis

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

class ResultGenerator:
    def __init__(self, output_dir="results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.loader = DataLoader()

    def step1_feature_engineering(self):
        """第一步：特征工程可视化"""
        print("\n=== 第一步：特征工程可视化 ===")

        df_raw = self.loader.load("data/student_all_augmented.csv")
        df_processed = self.loader.prepare_features(df_raw)

        # 1.1 衍生特征分布图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        derived_features = ['grade_volatility', 'grade_trend', 'total_grade', 'study_efficiency']
        titles = ['成绩波动率分布', '成绩线性趋势分布', '总成绩分布', '学习效率分布']

        for idx, (feature, title) in enumerate(zip(derived_features, titles)):
            ax = axes[idx // 2, idx % 2]
            ax.hist(df_processed[feature], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
            ax.set_title(title, fontsize=12, weight='bold')
            ax.set_xlabel('归一化值', fontsize=10)
            ax.set_ylabel('频数', fontsize=10)
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "1_特征工程_衍生特征分布.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("[OK] 生成: 1_特征工程_衍生特征分布.png")

        # 1.2 特征相关性热力图
        fig, ax = plt.subplots(figsize=(14, 12))
        corr_matrix = df_processed.corr()
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('特征相关性矩阵', fontsize=14, weight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.output_dir / "1_特征工程_相关性矩阵.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("[OK] 生成: 1_特征工程_相关性矩阵.png")

        # 1.3 关键特征对比（原始 vs 衍生）
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].scatter(df_processed['G1'], df_processed['G3'], alpha=0.3, s=10)
        axes[0].set_xlabel('期初成绩 (G1)', fontsize=11)
        axes[0].set_ylabel('期末成绩 (G3)', fontsize=11)
        axes[0].set_title('原始特征: G1 vs G3', fontsize=12, weight='bold')
        axes[0].grid(alpha=0.3)

        axes[1].scatter(df_processed['grade_volatility'], df_processed['grade_trend'],
                       alpha=0.3, s=10, c='coral')
        axes[1].set_xlabel('成绩波动率', fontsize=11)
        axes[1].set_ylabel('成绩趋势', fontsize=11)
        axes[1].set_title('衍生特征: 波动率 vs 趋势', fontsize=12, weight='bold')
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "1_特征工程_特征对比.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("[OK] 生成: 1_特征工程_特征对比.png")

        return df_processed

    def step2_algorithm_comparison(self, df_processed):
        """第二步：算法对比实验"""
        print("\n=== 第二步：PSO-KMeans 算法对比 ===")

        k_values = [3, 4, 5, 6]
        results = []

        for k in k_values:
            print(f"  测试 k={k}...")
            for algo_name, model in [
                ("K-Means", KMeans(n_clusters=k, init='random', n_init=10, random_state=42)),
                ("K-Means++", KMeans(n_clusters=k, init='k-means++', n_init=20, random_state=42)),
                ("PSO-KMeans", PSOKMeans(n_clusters=k, pso_max_iter=30, random_state=42))
            ]:
                start_time = time.time()
                model.fit(df_processed)
                elapsed = time.time() - start_time

                results.append({
                    'k': k,
                    'algorithm': algo_name,
                    'sse': model.inertia_,
                    'time': elapsed
                })

        results_df = pd.DataFrame(results)

        # 2.1 SSE 对比图
        fig, ax = plt.subplots(figsize=(10, 6))
        for algo in ['K-Means', 'K-Means++', 'PSO-KMeans']:
            data = results_df[results_df['algorithm'] == algo]
            ax.plot(data['k'], data['sse'], marker='o', linewidth=2, label=algo, markersize=8)

        ax.set_xlabel('聚类簇数 (k)', fontsize=12)
        ax.set_ylabel('SSE (误差平方和)', fontsize=12)
        ax.set_title('三种算法的 SSE 对比', fontsize=14, weight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "2_算法对比_SSE.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("[OK] 生成: 2_算法对比_SSE.png")

        # 2.2 运行时间对比
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(k_values))
        width = 0.25

        for idx, algo in enumerate(['K-Means', 'K-Means++', 'PSO-KMeans']):
            data = results_df[results_df['algorithm'] == algo]
            ax.bar(x + idx * width, data['time'], width, label=algo, alpha=0.8)

        ax.set_xlabel('聚类簇数 (k)', fontsize=12)
        ax.set_ylabel('运行时间 (秒)', fontsize=12)
        ax.set_title('三种算法的运行时间对比', fontsize=14, weight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(k_values)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "2_算法对比_运行时间.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("[OK] 生成: 2_算法对比_运行时间.png")

        # 2.3 综合对比表格
        pivot_sse = results_df.pivot(index='k', columns='algorithm', values='sse')
        pivot_time = results_df.pivot(index='k', columns='algorithm', values='time')

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].axis('tight')
        axes[0].axis('off')
        table1 = axes[0].table(cellText=pivot_sse.round(2).values,
                              rowLabels=pivot_sse.index,
                              colLabels=pivot_sse.columns,
                              cellLoc='center', loc='center')
        table1.auto_set_font_size(False)
        table1.set_fontsize(10)
        table1.scale(1, 2)
        axes[0].set_title('SSE 对比表', fontsize=12, weight='bold', pad=20)

        axes[1].axis('tight')
        axes[1].axis('off')
        table2 = axes[1].table(cellText=pivot_time.round(3).values,
                              rowLabels=pivot_time.index,
                              colLabels=pivot_time.columns,
                              cellLoc='center', loc='center')
        table2.auto_set_font_size(False)
        table2.set_fontsize(10)
        table2.scale(1, 2)
        axes[1].set_title('运行时间对比表 (秒)', fontsize=12, weight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(self.output_dir / "2_算法对比_综合表格.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("[OK] 生成: 2_算法对比_综合表格.png")

        return results_df

    def step3_clustering_visualization(self, df_processed):
        """第三步：聚类结果可视化"""
        print("\n=== 第三步：聚类结果可视化 ===")

        k = 4
        model = PSOKMeans(n_clusters=k, pso_max_iter=30, random_state=42)
        labels = model.fit_predict(df_processed)

        # 3.1 PCA 降维可视化
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(df_processed)

        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis',
                            alpha=0.6, s=20, edgecolors='k', linewidth=0.3)
        ax.scatter(pca.transform(model.cluster_centers_)[:, 0],
                  pca.transform(model.cluster_centers_)[:, 1],
                  c='red', marker='X', s=300, edgecolors='black', linewidth=2,
                  label='聚类中心')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} 方差)', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} 方差)', fontsize=12)
        ax.set_title(f'聚类结果 PCA 可视化 (k={k})', fontsize=14, weight='bold')
        ax.legend(fontsize=11)
        plt.colorbar(scatter, label='簇标签')
        plt.tight_layout()
        plt.savefig(self.output_dir / "3_聚类结果_PCA降维.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("[OK] 生成: 3_聚类结果_PCA降维.png")

        # 3.2 t-SNE 降维可视化
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(df_processed.sample(n=min(3000, len(df_processed)), random_state=42))
        labels_sample = labels[df_processed.sample(n=min(3000, len(df_processed)), random_state=42).index]

        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels_sample, cmap='viridis',
                            alpha=0.6, s=20, edgecolors='k', linewidth=0.3)
        ax.set_xlabel('t-SNE 维度 1', fontsize=12)
        ax.set_ylabel('t-SNE 维度 2', fontsize=12)
        ax.set_title(f't-SNE 聚类可视化 (k={k})', fontsize=14, weight='bold')
        plt.colorbar(scatter, label='簇标签')
        plt.tight_layout()
        plt.savefig(self.output_dir / "3_聚类结果_tSNE降维.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("[OK] 生成: 3_聚类结果_tSNE降维.png")

        # 3.3 簇大小分布
        cluster_sizes = pd.Series(labels).value_counts().sort_index()

        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(cluster_sizes.index, cluster_sizes.values, color='steelblue', alpha=0.8, edgecolor='black')
        ax.set_xlabel('簇标签', fontsize=12)
        ax.set_ylabel('样本数量', fontsize=12)
        ax.set_title('各簇样本分布', fontsize=14, weight='bold')
        ax.set_xticks(cluster_sizes.index)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(self.output_dir / "3_聚类结果_簇大小分布.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("[OK] 生成: 3_聚类结果_簇大小分布.png")

        return model, labels

    def step4_group_portrait(self, df_processed, model):
        """第四步：群体画像与学业诊断"""
        print("\n=== 第四步：群体画像与学业诊断 ===")

        centroids_df = pd.DataFrame(model.cluster_centers_, columns=df_processed.columns)

        # 4.1 雷达图
        plot_radar_chart(centroids_df, title="学生群体特征雷达图",
                        save_path=self.output_dir / "4_群体画像_雷达图.png")
        print("[OK] 生成: 4_群体画像_雷达图.png")

        # 4.2 热力图
        plot_feature_heatmap(centroids_df, save_path=self.output_dir / "4_群体画像_热力图.png")
        print("[OK] 生成: 4_群体画像_热力图.png")

        # 4.3 学业诊断报告
        diagnoses = generate_education_diagnosis(centroids_df)

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')

        report_text = "学生群体学业诊断报告\n" + "="*50 + "\n\n"
        for diag in diagnoses:
            report_text += f"群体 {diag['cluster_id']}: {diag['label']}\n"
            report_text += f"  画像: {diag['description']}\n"
            report_text += f"  建议: {diag['suggestion']}\n\n"

        ax.text(0.1, 0.9, report_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        plt.savefig(self.output_dir / "4_学业诊断_报告.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("[OK] 生成: 4_学业诊断_报告.png")

        # 4.4 关键特征对比（各簇）
        key_features = ['total_grade', 'grade_volatility', 'grade_trend', 'study_efficiency']
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        for idx, feature in enumerate(key_features):
            ax = axes[idx // 2, idx % 2]
            values = centroids_df[feature].values
            bars = ax.bar(range(len(values)), values, color='coral', alpha=0.7, edgecolor='black')
            ax.set_xlabel('群体编号', fontsize=11)
            ax.set_ylabel('归一化值', fontsize=11)
            ax.set_title(f'{feature} 各群体对比', fontsize=12, weight='bold')
            ax.set_xticks(range(len(values)))
            ax.grid(axis='y', alpha=0.3)

            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(self.output_dir / "4_群体画像_关键特征对比.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("[OK] 生成: 4_群体画像_关键特征对比.png")

        return diagnoses

    def step5_summary_report(self, df_processed, results_df, diagnoses):
        """第五步：生成总结报告"""
        print("\n=== 第五步：生成总结报告 ===")

        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 5.1 数据概览
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')
        summary_text = f"""
        实验总结报告
        {'='*80}

        数据集信息:
          - 原始数据: 1044 条
          - 增强后数据: {len(df_processed)} 条
          - 特征数量: {df_processed.shape[1]} 个 (包含 4 个衍生特征)

        算法对比结果 (k=4):
          - K-Means:     SSE = {results_df[(results_df['k']==4) & (results_df['algorithm']=='K-Means')]['sse'].values[0]:.2f}
          - K-Means++:   SSE = {results_df[(results_df['k']==4) & (results_df['algorithm']=='K-Means++')]['sse'].values[0]:.2f}
          - PSO-KMeans:  SSE = {results_df[(results_df['k']==4) & (results_df['algorithm']=='PSO-KMeans')]['sse'].values[0]:.2f}

        学生群体分类:
        """

        for diag in diagnoses:
            summary_text += f"  群体 {diag['cluster_id']}: {diag['label']}\n"

        ax1.text(0.05, 0.95, summary_text, transform=ax1.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        # 5.2 特征重要性（方差分析）
        ax2 = fig.add_subplot(gs[1, 0])
        feature_var = df_processed.var().sort_values(ascending=False)[:10]
        ax2.barh(range(len(feature_var)), feature_var.values, color='steelblue', alpha=0.7)
        ax2.set_yticks(range(len(feature_var)))
        ax2.set_yticklabels(feature_var.index, fontsize=9)
        ax2.set_xlabel('方差', fontsize=10)
        ax2.set_title('Top 10 特征方差', fontsize=11, weight='bold')
        ax2.grid(axis='x', alpha=0.3)

        # 5.3 算法性能对比
        ax3 = fig.add_subplot(gs[1, 1])
        k4_data = results_df[results_df['k'] == 4]
        x_pos = np.arange(len(k4_data))
        ax3.bar(x_pos, k4_data['sse'], color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(k4_data['algorithm'], fontsize=9)
        ax3.set_ylabel('SSE', fontsize=10)
        ax3.set_title('k=4 时算法 SSE 对比', fontsize=11, weight='bold')
        ax3.grid(axis='y', alpha=0.3)

        # 5.4 数据增强效果
        ax4 = fig.add_subplot(gs[2, :])
        categories = ['原始数据', '增强后数据']
        values = [1044, len(df_processed)]
        bars = ax4.bar(categories, values, color=['coral', 'steelblue'], alpha=0.7, edgecolor='black')
        ax4.set_ylabel('样本数量', fontsize=11)
        ax4.set_title('数据增强效果对比', fontsize=12, weight='bold')

        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=11, weight='bold')

        plt.savefig(self.output_dir / "5_实验总结报告.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("[OK] 生成: 5_实验总结报告.png")

    def run_all(self):
        """运行所有步骤"""
        print("\n" + "="*60)
        print("开始生成毕业论文实验结果")
        print("="*60)

        df_processed = self.step1_feature_engineering()
        results_df = self.step2_algorithm_comparison(df_processed)
        model, labels = self.step3_clustering_visualization(df_processed)
        diagnoses = self.step4_group_portrait(df_processed, model)
        self.step5_summary_report(df_processed, results_df, diagnoses)

        print("\n" + "="*60)
        print(f"所有结果已生成到: {self.output_dir.absolute()}")
        print("="*60)

if __name__ == "__main__":
    generator = ResultGenerator(output_dir="results")
    generator.run_all()
