"""edu_cluster - 教育数据聚类分析项目"""

from .data_loader import (
    load_student_data,
    get_data_info,
    get_descriptive_stats,
    get_missing_summary,
    load_and_inspect,
)

__all__ = [
    "load_student_data",
    "get_data_info",
    "get_descriptive_stats",
    "get_missing_summary",
    "load_and_inspect",
]
