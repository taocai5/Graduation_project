"""
聚类实验流水线
支持肘部法、轮廓系数评估、多算法对比实验
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pso_kmeans import PSOKMeans


@dataclass
class ExperimentConfig:
    """聚类实验配置"""
    data_path: str = "data/student_all_augmented.csv"
    output_dir: str = "output"
    k_min: int = 2
    k_max: int = 10
    compare_k: List[int] = None
    random_state: int = 42

    def __post_init__(self):
        if self.compare_k is None:
            self.compare_k = [3, 4, 5]


class DataLoader:
    """数据加载与特征工程"""
    
    DEFAULT_FEATURES = [
        "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
        "famrel", "freetime", "goout", "Dalc", "Walc", "health",
        "absences", "G1", "G2", "G3"
    ]
    
    @staticmethod
    def load(path: str) -> pd.DataFrame:
        """智能加载不同格式的数据集"""
        if not Path(path).exists():
            raise FileNotFoundError(f"数据集不存在: {path}")
        
        if path.endswith(("student_all.csv", "student_all_augmented.csv")):
            return pd.read_csv(path)
        else:
            return pd.read_csv(path, sep=";")
    
    @staticmethod
    def prepare_features(df: pd.DataFrame, feature_cols: List[str] = None) -> pd.DataFrame:
        """特征选择、清洗和归一化"""

        for col in ['G1', 'G2', 'G3']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace('"', ''), errors='coerce')

        if all(c in df.columns for c in ['G1', 'G2', 'G3']):
            df['grade_volatility'] = df[['G1', 'G2', 'G3']].std(axis=1)
            df['grade_trend'] = df['G3'] - df['G1']
            df['total_grade'] = df['G1'] + df['G2'] + df['G3']

        if 'total_grade' in df.columns and 'studytime' in df.columns:
            df['study_efficiency'] = df['total_grade'] / df['studytime'].clip(lower=1)

        if feature_cols is None:
            feature_cols = DataLoader.DEFAULT_FEATURES.copy()
            for new_col in ['grade_volatility', 'grade_trend', 'total_grade', 'study_efficiency']:
                if new_col in df.columns:
                    feature_cols.append(new_col)

        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"缺失特征列: {missing}")

        x = df[feature_cols].copy()
        x = x.dropna(axis=0, how="any")

        scaler = MinMaxScaler()
        x_scaled = scaler.fit_transform(x)

        return pd.DataFrame(x_scaled, columns=feature_cols)


class ClusteringEvaluator:
    """聚类评估与可视化"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_k_range(self, x: pd.DataFrame, k_min: int, k_max: int,
                        random_state: int = 42) -> pd.DataFrame:
        """评估不同 k 值的聚类效果"""
        rows = []
        for k in range(k_min, k_max + 1):
            model = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=20,
                max_iter=300,
                random_state=random_state,
            )
            labels = model.fit_predict(x)
            sse = float(model.inertia_)
            sil = float(silhouette_score(x, labels))
            rows.append({"k": k, "sse": sse, "silhouette": sil})

        return pd.DataFrame(rows)
    
    def plot_metrics(self, metrics_df: pd.DataFrame, prefix: str = "") -> Tuple[str, str]:
        """绘制肘部法和轮廓系数图"""
        elbow_path = self.output_dir / f"{prefix}elbow_curve.png"
        sil_path = self.output_dir / f"{prefix}silhouette_curve.png"
        
        # Elbow Curve
        plt.figure(figsize=(8, 5))
        plt.plot(metrics_df["k"], metrics_df["sse"], marker="o")
        plt.title("Elbow Method (K-Means++)")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("SSE (Inertia)")
        plt.xticks(metrics_df["k"].tolist())
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(elbow_path, dpi=150)
        plt.close()
        
        # Silhouette Score
        plt.figure(figsize=(8, 5))
        plt.plot(metrics_df["k"], metrics_df["silhouette"], marker="o")
        plt.title("Silhouette Score by k")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Silhouette Coefficient")
        plt.xticks(metrics_df["k"].tolist())
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(sil_path, dpi=150)
        plt.close()
        
        return str(elbow_path), str(sil_path)
    
    def compare_initializers(self, x: pd.DataFrame, k_values: List[int]) -> pd.DataFrame:
        """对比不同聚类算法"""
        rows = []
        for k in k_values:
            for init_name in ["random", "k-means++", "pso"]:
                start_time = time.time()

                if init_name == "pso":
                    model = PSOKMeans(n_clusters=k, random_state=42)
                    algo_name = "PSO-KMeans"
                else:
                    model = KMeans(
                        n_clusters=k,
                        init=init_name,
                        n_init=10 if init_name == "random" else 20,
                        max_iter=300,
                        random_state=42,
                    )
                    algo_name = "K-Means" if init_name == "random" else "K-Means++"

                labels = model.fit_predict(x)
                elapsed_time = time.time() - start_time

                rows.append({
                    "k": k,
                    "algorithm": algo_name,
                    "init": init_name,
                    "sse": float(model.inertia_),
                    "silhouette": float(silhouette_score(x, labels)),
                    "davies_bouldin": float(davies_bouldin_score(x, labels)),
                    "time_sec": elapsed_time
                })

        return pd.DataFrame(rows).sort_values(["k", "algorithm"]).reset_index(drop=True)


class ExperimentRunner:
    """聚类实验运行主类"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.loader = DataLoader()
        self.evaluator = ClusteringEvaluator(config.output_dir)
    
    def run(self):
        """执行完整实验流程"""
        print(f"开始聚类实验 - 数据源: {self.config.data_path}")
        
        df = self.loader.load(self.config.data_path)
        X = self.loader.prepare_features(df)
        
        print(f"使用样本数: {len(X)} 条")
        
        # 1. Elbow + Silhouette 评估
        metrics_df = self.evaluator.evaluate_k_range(
            X, self.config.k_min, self.config.k_max, self.config.random_state
        )
        metrics_df.to_csv(
            self.evaluator.output_dir / "elbow_silhouette_metrics.csv", 
            index=False, encoding="utf-8-sig"
        )
        
        self.evaluator.plot_metrics(metrics_df)
        
        # 2. 对比实验
        compare_df = self.evaluator.compare_initializers(X, self.config.compare_k)
        compare_df.to_csv(
            self.evaluator.output_dir / "kmeans_vs_kmeanspp_comparison.csv",
            index=False, encoding="utf-8-sig"
        )
        
        print("实验完成！")
        print(f"输出目录: {self.evaluator.output_dir.absolute()}")
        print(f"生成文件: elbow_silhouette_metrics.csv, kmeans_vs_kmeanspp_comparison.csv, *.png")


def main():
    parser = argparse.ArgumentParser(description="毕业设计 - 聚类分析实验工具")
    parser.add_argument("--data-path", default="data/student_all_augmented.csv", help="数据集路径")
    parser.add_argument("--output-dir", default="output", help="输出目录")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--compare-k", default="3,4,5")
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()
    compare_k = [int(k.strip()) for k in args.compare_k.split(",")]

    config = ExperimentConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        k_min=args.k_min,
        k_max=args.k_max,
        compare_k=compare_k,
        random_state=args.random_state,
    )

    runner = ExperimentRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
