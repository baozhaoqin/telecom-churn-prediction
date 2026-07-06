"""
特征工程模块
@description 类别编码、特征缩放、训练/测试集划分
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, List


# 特征分类
BINARY_CATEGORICAL = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "PaperlessBilling",
]

MULTI_CATEGORICAL = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]

NUMERICAL = ["tenure", "MonthlyCharges", "TotalCharges"]


def encode_binary(df: pd.DataFrame) -> pd.DataFrame:
    """对二元类别变量做 0/1 编码

    Args:
        df: 原始 DataFrame

    Returns:
        pd.DataFrame: 部分编码后数据
    """
    df = df.copy()
    for col in BINARY_CATEGORICAL:
        if col in df.columns:
            # 取众数作为正向编码
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) == 2:
                # 按字母序，第一个为 0，第二个为 1
                mapping = {val: i for i, val in enumerate(sorted(unique_vals))}
                df[col] = df[col].map(mapping)
    return df


def build_preprocessor() -> ColumnTransformer:
    """构建特征预处理管道

    - 多类别变量: OneHotEncoder (drop='first')
    - 数值变量: StandardScaler
    - 二元类别变量已预先编码，通过 remainder='passthrough' 保留

    Returns:
        ColumnTransformer: 预处理管道
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=False), MULTI_CATEGORICAL),
            ("num", StandardScaler(), NUMERICAL),
        ],
        remainder="passthrough",  # 保留已编码的二元变量
    )
    return preprocessor


def prepare_data(
    df: pd.DataFrame, target_col: str, test_size: float, val_size: float, random_state: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], ColumnTransformer]:
    """完整特征工程流水线

    Args:
        df: 清洗后 DataFrame
        target_col: 目标列名
        test_size: 测试集比例
        val_size: 验证集比例（从训练集中划分）
        random_state: 随机种子

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names, preprocessor
    """
    # 分离特征与标签
    X = df.drop(columns=[target_col])
    y = df[target_col].values

    # 二元编码
    X = encode_binary(X)

    # 拆分 train/val/test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_ratio = val_size / (1 - test_size)  # 从剩余数据中划分验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )

    # 构建并拟合预处理器
    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(X_test)

    # 获取特征名
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = (
        cat_encoder.get_feature_names_out(MULTI_CATEGORICAL).tolist()
    )
    feature_names = cat_names + NUMERICAL + BINARY_CATEGORICAL

    print(f"[features] 训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
    print(f"[features] 特征数: {len(feature_names)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names, preprocessor
