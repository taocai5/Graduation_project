import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

def plot_radar_chart(centroids_df: pd.DataFrame, title: str = "群体特征雷达图", save_path: str = None):
    """绘制多维特征的雷达图"""
    categories = centroids_df.columns.tolist()
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

    for idx, row in centroids_df.iterrows():
        values = row.tolist()
        values += values[:1]

        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'群体 {idx}')
        ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)

    ax.set_title(title, size=15, weight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
    else:
        return fig

def plot_feature_heatmap(centroids_df: pd.DataFrame, save_path: str = None):
    """绘制特征热力图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(centroids_df, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
    ax.set_title("各聚类群体特征热力图", size=14)
    ax.set_ylabel("群体 (Cluster ID)")
    ax.set_xlabel("特征 (Features)")

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
    else:
        return fig


def _classify_by_performance(row):
    """根据总成绩分类"""
    if 'total_grade' not in row:
        return None

    if row['total_grade'] > 0.7:
        return {
            "label": "均衡优秀型",
            "description": "整体成绩优异，学习状态稳定。",
            "suggestion": "保持当前学习节奏，可适当增加拔高训练，参与竞赛。"
        }
    elif row['total_grade'] < 0.3:
        return {
            "label": "基础薄弱型",
            "description": "整体成绩偏低，基础知识掌握不牢固。",
            "suggestion": "回归课本基础，制定补救计划，老师需加强一对一辅导。"
        }
    return None

def _classify_by_volatility(row):
    """根据成绩波动分类"""
    if 'grade_volatility' not in row:
        return None

    if row['grade_volatility'] > 0.6:
        return {
            "label": "成绩波动型",
            "description": "成绩起伏较大，可能存在偏科或学习状态不稳定。",
            "suggestion": "分析偏科原因，稳定心态，加强薄弱学科的练习。"
        }
    return None

def _classify_by_trend(row):
    """根据成绩趋势分类"""
    if 'grade_trend' not in row:
        return None

    if row['grade_trend'] > 0.5:
        return {
            "label": "稳步提升型",
            "description": "成绩呈现明显上升趋势，学习方法有效。",
            "suggestion": "给予肯定和鼓励，总结成功经验，继续保持。"
        }
    return None

def generate_education_diagnosis(centroids_df: pd.DataFrame):
    """生成学业诊断与教学建议"""
    diagnosis = []
    classifiers = [_classify_by_performance, _classify_by_volatility, _classify_by_trend]

    for idx, row in centroids_df.iterrows():
        diag = {"cluster_id": idx}

        for classifier in classifiers:
            result = classifier(row)
            if result:
                diag.update(result)
                break

        if "label" not in diag:
            diag.update({
                "label": "常规发展型",
                "description": "各方面表现平平，无明显短板也无突出优势。",
                "suggestion": "发掘学生兴趣点，寻找突破口，激发学习动力。"
            })

        diagnosis.append(diag)

    return diagnosis
