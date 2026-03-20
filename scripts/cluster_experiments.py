"""
Cluster experiment pipeline for graduation project.

Outputs:
1) Elbow metrics CSV (SSE + Silhouette for k range)
2) Elbow/Silhouette plots
3) K-Means(random) vs K-Means++ comparison CSV (>= 3 k values)
"""

import argparse
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler


DEFAULT_NUMERIC_FEATURES = [
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences",
    "G1",
    "G2",
    "G3",
]


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    # Student dataset commonly uses ';' separator.
    return pd.read_csv(path, sep=";")


def prepare_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    x = df[feature_cols].copy()
    x = x.dropna(axis=0, how="any")
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    return pd.DataFrame(x_scaled, columns=feature_cols)


def evaluate_k_range(
    x: pd.DataFrame,
    k_min: int,
    k_max: int,
    random_state: int,
) -> pd.DataFrame:
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


def plot_metrics(metrics_df: pd.DataFrame, output_dir: str) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    elbow_png = os.path.join(output_dir, "elbow_curve.png")
    sil_png = os.path.join(output_dir, "silhouette_curve.png")

    plt.figure(figsize=(8, 5))
    plt.plot(metrics_df["k"], metrics_df["sse"], marker="o")
    plt.title("Elbow Method (K-Means++)")
    plt.xlabel("k")
    plt.ylabel("SSE (Inertia)")
    plt.xticks(metrics_df["k"].tolist())
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(elbow_png, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(metrics_df["k"], metrics_df["silhouette"], marker="o")
    plt.title("Silhouette Score by k (K-Means++)")
    plt.xlabel("k")
    plt.ylabel("Silhouette")
    plt.xticks(metrics_df["k"].tolist())
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(sil_png, dpi=150)
    plt.close()

    return elbow_png, sil_png


def compare_algorithms(x: pd.DataFrame, k_values: List[int]) -> pd.DataFrame:
    rows = []
    for k in k_values:
        for init_name in ["random", "k-means++"]:
            model = KMeans(
                n_clusters=k,
                init=init_name,
                n_init=20,
                max_iter=300,
                random_state=42,
            )
            labels = model.fit_predict(x)
            rows.append(
                {
                    "k": k,
                    "algorithm": "K-Means" if init_name == "random" else "K-Means++",
                    "init": init_name,
                    "sse": float(model.inertia_),
                    "silhouette": float(silhouette_score(x, labels)),
                    "iterations": int(model.n_iter_),
                }
            )
    result = pd.DataFrame(rows).sort_values(["k", "algorithm"]).reset_index(drop=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run clustering experiments.")
    parser.add_argument(
        "--data-path",
        default="data/student-mat.csv",
        help="Path to dataset CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory to save experiment results.",
    )
    parser.add_argument("--k-min", type=int, default=2, help="Min k for elbow scan.")
    parser.add_argument("--k-max", type=int, default=10, help="Max k for elbow scan.")
    parser.add_argument(
        "--compare-k",
        default="3,4,5",
        help="Comma-separated k values for K-Means vs K-Means++ comparison.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for elbow/silhouette run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compare_k_values = [int(v.strip()) for v in args.compare_k.split(",") if v.strip()]
    if len(compare_k_values) < 3:
        raise ValueError("Please provide at least 3 k values in --compare-k.")

    df = load_dataset(args.data_path)
    x = prepare_features(df, DEFAULT_NUMERIC_FEATURES)
    os.makedirs(args.output_dir, exist_ok=True)

    metrics_df = evaluate_k_range(x, args.k_min, args.k_max, args.random_state)
    metrics_csv = os.path.join(args.output_dir, "elbow_silhouette_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")

    elbow_png, sil_png = plot_metrics(metrics_df, args.output_dir)

    compare_df = compare_algorithms(x, compare_k_values)
    compare_csv = os.path.join(args.output_dir, "kmeans_vs_kmeanspp_comparison.csv")
    compare_df.to_csv(compare_csv, index=False, encoding="utf-8-sig")

    print("Experiment completed.")
    print(f"Data path: {args.data_path}")
    print(f"Rows used for clustering: {len(x)}")
    print(f"Metrics CSV: {metrics_csv}")
    print(f"Elbow plot: {elbow_png}")
    print(f"Silhouette plot: {sil_png}")
    print(f"Comparison CSV: {compare_csv}")


if __name__ == "__main__":
    main()
