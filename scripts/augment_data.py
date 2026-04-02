"""
学生成绩数据增强模块（面向对象版本）
通过添加高斯噪声实现数据集扩充
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class AugmentConfig:
    """数据增强配置"""
    input_file: str = "data/student_all.csv"
    output_file: str = "data/student_all_augmented.csv"
    n_augments: int = 2
    noise_level: float = 0.03
    random_state: int = 42


class DataAugmenter:
    """学生成绩数据增强器"""
    
    def __init__(self, config: Optional[AugmentConfig] = None):
        self.config = config or AugmentConfig()
        self.input_path = Path(self.config.input_file)
        self.output_path = Path(self.config.output_file)
        np.random.seed(self.config.random_state)

    def validate(self) -> bool:
        """验证输入文件"""
        if not self.input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {self.input_path}")
        print(f"[OK] 找到输入文件: {self.input_path.name}")
        return True

    def get_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """获取适合做噪声增强的数值列"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = {'age', 'Medu', 'Fedu', 'failures', 'subject'}
        return [col for col in numeric_cols if col not in exclude_cols]

    def augment(self, df: pd.DataFrame) -> pd.DataFrame:
        """对数据进行增强"""
        numeric_cols = self.get_numeric_columns(df)
        augmented_dfs = [df.copy()]

        print(f"开始进行 {self.config.n_augments} 倍数据增强...")

        for i in range(self.config.n_augments):
            df_aug = df.copy()
            for col in numeric_cols:
                if col not in df_aug.columns:
                    continue
                std = df_aug[col].std()
                if std < 1e-6:
                    continue
                noise = np.random.normal(0, self.config.noise_level * std, size=len(df_aug))
                df_aug[col] = df_aug[col] + noise
                
                # 约束合理范围
                if col in ['G1', 'G2', 'G3']:
                    df_aug[col] = df_aug[col].clip(0, 20)
                elif col in ['absences']:
                    df_aug[col] = df_aug[col].clip(0, None)
                elif col in ['studytime', 'traveltime', 'famrel', 'freetime', 
                           'goout', 'Dalc', 'Walc', 'health']:
                    df_aug[col] = df_aug[col].clip(1, 5)
            
            augmented_dfs.append(df_aug)
            print(f"  已完成第 {i+1} 轮增强")

        final_df = pd.concat(augmented_dfs, ignore_index=True)
        print(f"数据增强完成！最终形状: {final_df.shape}")
        return final_df

    def run(self) -> bool:
        """执行完整的数据增强流程"""
        try:
            self.validate()
            df = pd.read_csv(self.input_path)
            print(f"原始数据形状: {df.shape}")

            augmented_df = self.augment(df)
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            augmented_df.to_csv(self.output_path, index=False, encoding="utf-8-sig")

            print(f"[SUCCESS] 增强数据已保存至: {self.output_path.name}")
            print(f"总记录数: {len(augmented_df):,} 条（原始 {len(df)} 条 x {self.config.n_augments + 1}）")
            return True

        except Exception as e:
            print(f"[ERROR] 数据增强失败: {e}")
            return False


if __name__ == "__main__":
    augmenter = DataAugmenter()
    augmenter.run()
