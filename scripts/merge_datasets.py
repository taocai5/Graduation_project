"""
学生成绩数据集合并模块（面向对象版本）
Author: Graduation Project
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


@dataclass
class MergeConfig:
    """数据集合并配置类"""
    data_dir: str = "data"
    mat_file: str = "student-mat.csv"
    por_file: str = "student-por.csv"
    output_file: str = "student_all.csv"
    subject_column: str = "subject"


class StudentDataMerger:
    """学生成绩数据集合并器"""
    
    def __init__(self, config: Optional[MergeConfig] = None):
        self.config = config or MergeConfig()
        self.data_dir = Path(self.config.data_dir)
        self.mat_path = self.data_dir / self.config.mat_file
        self.por_path = self.data_dir / self.config.por_file
        self.output_path = self.data_dir / self.config.output_file

    def validate_input_files(self) -> bool:
        """验证输入文件是否存在"""
        if not self.mat_path.exists():
            raise FileNotFoundError(f"数学课程数据文件不存在: {self.mat_path}")
        if not self.por_path.exists():
            raise FileNotFoundError(f"葡萄牙语课程数据文件不存在: {self.por_path}")
        print(f"✓ 找到输入文件: {self.mat_path.name} 和 {self.por_path.name}")
        return True

    def load_raw_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载两个原始数据集"""
        print("正在读取原始数据集...")
        mat_df = pd.read_csv(self.mat_path, sep=";")
        por_df = pd.read_csv(self.por_path, sep=";")
        
        print(f"student-mat.csv 形状: {mat_df.shape}")
        print(f"student-por.csv 形状: {por_df.shape}")
        return mat_df, por_df

    def add_subject_labels(self, mat_df: pd.DataFrame, por_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """为两个数据集添加科目标签"""
        mat_df = mat_df.copy()
        por_df = por_df.copy()
        mat_df[self.config.subject_column] = "mat"
        por_df[self.config.subject_column] = "por"
        return mat_df, por_df

    def merge(self) -> pd.DataFrame:
        """执行完整的数据集合并流程"""
        self.validate_input_files()
        mat_df, por_df = self.load_raw_datasets()
        mat_df, por_df = self.add_subject_labels(mat_df, por_df)

        combined_df = pd.concat([mat_df, por_df], ignore_index=True)
        
        print(f"\n✅ 数据集合并完成！")
        print(f"合并后总记录数: {len(combined_df):,} 条")
        print(f"科目分布: 数学(mat) = {len(mat_df):,} 条, 葡萄牙语(por) = {len(por_df):,} 条")
        
        return combined_df

    def save(self, df: pd.DataFrame) -> Path:
        """保存合并后的数据集"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_path, index=False, encoding="utf-8-sig")
        print(f"已保存至: {self.output_path.absolute()}")
        print(f"总字段数: {len(df.columns)}")
        return self.output_path

    def run(self) -> bool:
        """一键执行完整合并流程"""
        try:
            combined_df = self.merge()
            self.save(combined_df)
            return True
        except Exception as e:
            print(f"❌ 数据集合并失败: {e}")
            return False


if __name__ == "__main__":
    merger = StudentDataMerger()
    merger.run()
