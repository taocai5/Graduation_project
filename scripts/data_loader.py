"""
数据加载与基本信息统计模块
用于学生成绩数据集的读取、初步探查和基本统计
"""

import os
from typing import Optional

import pandas as pd


def load_student_data(
    filepath: str,
    encoding: str = "utf-8",
    sep: Optional[str] = None,
) -> pd.DataFrame:
    """
    使用 pandas 读取学生成绩原始 CSV 文件。

    Args:
        filepath: 数据文件路径（支持相对路径或绝对路径）
        encoding: 文件编码，默认 utf-8 以处理中文字符
        sep: 分隔符，None 时自动检测（逗号或制表符）

    Returns:
        加载后的 DataFrame

    Raises:
        FileNotFoundError: 文件不存在时抛出
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"数据文件不存在: {filepath}")

    # 常见分隔符尝试顺序
    if sep is None:
        with open(filepath, "r", encoding=encoding, errors="replace") as f:
            first_line = f.readline()
            sep = "," if "," in first_line else "\t"

    df = pd.read_csv(filepath, encoding=encoding, sep=sep)
    return df


def get_data_info(df: pd.DataFrame) -> str:
    """
    获取数据概览信息，包括非空计数、数据类型等（等价于 df.info() 的字符串输出）。

    Args:
        df: 待探查的 DataFrame

    Returns:
        格式化的信息字符串
    """
    buffer = []
    df.info(buf=lambda x: buffer.append(x))
    return "\n".join(buffer)


def get_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    对数值型字段进行描述性统计（等价于 df.describe()）。

    Args:
        df: 待统计的 DataFrame

    Returns:
        描述性统计结果
    """
    return df.describe()


def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    统计各列缺失值数量及占比。

    Args:
        df: 待检查的 DataFrame

    Returns:
        包含列名、缺失数、缺失占比的 DataFrame
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    return pd.DataFrame(
        {"缺失数": missing, "缺失占比(%)": missing_pct},
        index=missing.index,
    ).loc[missing > 0]


def load_and_inspect(
    filepath: str,
    encoding: str = "utf-8",
    sep: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    一站式：加载数据并输出基本信息统计。

    Args:
        filepath: 数据文件路径
        encoding: 文件编码
        sep: 分隔符
        verbose: 是否打印概览信息

    Returns:
        加载后的 DataFrame
    """
    df = load_student_data(filepath, encoding=encoding, sep=sep)

    if verbose:
        print("=" * 60)
        print("数据概览 (df.info)")
        print("=" * 60)
        print(get_data_info(df))
        print("\n" + "=" * 60)
        print("描述性统计 (df.describe)")
        print("=" * 60)
        print(get_descriptive_stats(df))
        missing = get_missing_summary(df)
        if not missing.empty:
            print("\n" + "=" * 60)
            print("缺失值统计")
            print("=" * 60)
            print(missing)

    return df


# 便于作为脚本直接运行做简单测试
if __name__ == "__main__":
    # 示例：加载 data/ 目录下的数据文件（需用户放置实际 CSV）
    default_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "student_performance.csv"
    )
    if os.path.exists(default_path):
        df = load_and_inspect(default_path)
        print(f"\n成功加载 {len(df)} 条记录")
    else:
        print(f"请将学生成绩 CSV 文件放入 data/ 目录，或指定路径。")
        print(f"预期默认路径: {default_path}")
