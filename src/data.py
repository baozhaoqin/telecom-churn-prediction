"""
数据加载与清洗模块
@description 加载 Telco Customer Churn 数据集，处理缺失值与异常值
"""
import pandas as pd
import numpy as np
from pathlib import Path


def load_data(filepath: str) -> pd.DataFrame:
    """加载原始 CSV 数据

    Args:
        filepath: CSV 文件路径

    Returns:
        pd.DataFrame: 原始数据
    """
    df = pd.read_csv(filepath)
    print(f"[data] 加载数据: {df.shape[0]} 行 × {df.shape[1]} 列")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """数据清洗

    - 删除 customerID 列
    - 转换 TotalCharges 为数值类型（空值→NaN→中位数填充）
    - 编码 Churn 目标变量

    Args:
        df: 原始 DataFrame

    Returns:
        pd.DataFrame: 清洗后数据
    """
    # 删除无用列
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # TotalCharges: 空字符串 → NaN → 中位数填充
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    nan_count = df["TotalCharges"].isna().sum()
    if nan_count > 0:
        median_val = df["TotalCharges"].median()
        df["TotalCharges"] = df["TotalCharges"].fillna(median_val)
        print(f"[data] TotalCharges 缺失值 {nan_count} 个，已用中位数 {median_val:.2f} 填充")

    # 目标变量编码
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # 数值型二元变量转换
    if df["SeniorCitizen"].dtype == "object":
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    print(f"[data] 清洗完成: {df.shape[0]} 行 × {df.shape[1]} 列, Churn 占比={df['Churn'].mean():.2%}")
    return df


def get_summary(df: pd.DataFrame) -> dict:
    """输出数据摘要统计

    Args:
        df: 清洗后 DataFrame

    Returns:
        dict: 统计信息
    """
    return {
        "n_samples": len(df),
        "n_features": len(df.columns) - 1,
        "churn_rate": df["Churn"].mean(),
        "missing_values": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
