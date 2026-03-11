"""
下载学生成绩数据集
优先从 UCI 官网直接下载（无需登录），也可通过 Kaggle API 下载
"""

import os
import zipfile
import urllib.request

# UCI 官方直接下载（无需 Kaggle 账号）
UCI_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def download_from_uci():
    """从 UCI 官网直接下载 Student Performance 数据集"""
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "student_performance.zip")

    print("正在从 UCI 官网下载 Student Performance 数据集...")
    urllib.request.urlretrieve(UCI_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR)
    print(f"已解压到: {os.path.abspath(DATA_DIR)}")
    print("下载完成。student-mat.csv、student-por.csv 可用于聚类分析。")


def download_from_kaggle():
    """通过 Kaggle API 下载（需配置 kaggle.json）"""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise ImportError("请先安装: pip install kaggle")

    api = KaggleApi()
    api.authenticate()
    os.makedirs(DATA_DIR, exist_ok=True)
    api.dataset_download_files("dskagglemt/student-performance-data-set", path=DATA_DIR, unzip=True)
    print(f"已保存到: {os.path.abspath(DATA_DIR)}")


def download(use_kaggle: bool = False):
    if use_kaggle:
        download_from_kaggle()
    else:
        download_from_uci()


if __name__ == "__main__":
    download()
