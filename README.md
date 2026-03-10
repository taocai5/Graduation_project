# Graduation_project

毕业论文项目：基于聚类的教育数据分析（K-Means / K-Means++ 等）

## 环境要求

- Python 3.9
- Microsoft C++ Build Tools（部分库如 scikit-learn 在 Windows 上编译时需要）

## 快速开始

### 1. 安装依赖

在项目根目录打开终端（PowerShell 或 CMD），执行：

```bash
pip install -r requirements.txt
```

### 3. 项目结构

```
Graduation_project/
├── data/           # 原始数据（如 Student Performance Dataset CSV）
├── src/            # 源代码
│   ├── data_loader.py   # 数据加载与基本信息统计
│   └── __init__.py
├── notebooks/      # 实验性 Jupyter Notebook
├── output/         # 结果图表与模型
├── md文档/         # 论文相关文档
└── requirements.txt
```

### 4. 使用数据加载模块

```python
from src.data_loader import load_and_inspect, load_student_data

# 加载并查看概览
df = load_and_inspect("data/student_performance.csv")

# 仅加载数据
df = load_student_data("data/student_performance.csv", encoding="utf-8")
```

### 5. 数据准备

从 [Kaggle Student Performance Dataset](https://www.kaggle.com/datasets) 等平台获取学生成绩 CSV，放入 `data/` 目录。文件编码建议为 UTF-8。

## 主要依赖

| 库 | 版本 | 用途 |
|----|------|------|
| pandas | 1.4.2 | 数据读取、清洗、特征工程 |
| numpy | 1.21.5 | 数值计算 |
| scikit-learn | 1.0.2 | K-Means、聚类评估 |
| matplotlib | 3.5.1 | 可视化 |
| seaborn | 0.11.2 | 可视化 |
| scipy | - | 优化与距离计算 |
| joblib | - | 并行加速 |
