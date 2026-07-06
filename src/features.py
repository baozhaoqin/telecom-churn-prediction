"""
特征工程模块
@description 类别编码、特征缩放、训练/测试集划分、特征选择、SMOTE 过采样
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from typing import Tuple, List, Optional


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

# 根据 EDA 发现的低重要性特征（可配置）
LOW_IMPORTANCE_FEATURES = ["gender", "PhoneService"]


def encode_binary(df: pd.DataFrame) -> pd.DataFrame:
    """对二元类别变量做 0/1 编码"""
    df = df.copy()
    for col in BINARY_CATEGORICAL:
        if col in df.columns:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) == 2:
                mapping = {val: i for i, val in enumerate(sorted(unique_vals))}
                df[col] = df[col].map(mapping)
    return df


def select_features_by_correlation(
    X: pd.DataFrame, threshold: float = 0.8
) -> Tuple[pd.DataFrame, List[str]]:
    """基于相关性剔除冗余特征

    对数值列计算皮尔逊相关系数，高于阈值的特征对保留一个（与目标相关性更强的）。

    Args:
        X: 特征 DataFrame（编码后）
        threshold: 相关性阈值

    Returns:
        剔除后的 DataFrame, 被剔除的特征名列表
    """
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        return X, []

    corr_matrix = X[num_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper.columns:
        for idx in upper.index:
            if upper.loc[idx, col] > threshold and col not in to_drop:
                to_drop.add(col)

    if to_drop:
        print(f"[features] 相关性特征选择: 剔除 {len(to_drop)} 个冗余特征: {list(to_drop)}")
        X = X.drop(columns=list(to_drop))

    return X, list(to_drop)


def select_features_by_importance(X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """基于 EDA 低重要性剔除特征

    Args:
        X: 特征 DataFrame

    Returns:
        剔除后的 DataFrame, 被剔除的特征名列表
    """
    to_drop = [c for c in LOW_IMPORTANCE_FEATURES if c in X.columns]
    if to_drop:
        print(f"[features] 低重要性特征选择: 剔除 {len(to_drop)} 个特征: {to_drop}")
        X = X.drop(columns=to_drop)
    return X, to_drop


def build_preprocessor() -> ColumnTransformer:
    """构建特征预处理管道

    - 多类别变量: OneHotEncoder (drop='first')
    - 数值变量: StandardScaler
    - 二元变量通过 remainder='passthrough' 保留
    """
    # 过滤掉不存在的列
    multi = [c for c in MULTI_CATEGORICAL if c in NUMERICAL or c in BINARY_CATEGORICAL]
    multi = [c for c in MULTI_CATEGORICAL if c not in set(NUMERICAL + BINARY_CATEGORICAL)]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=False), MULTI_CATEGORICAL),
            ("num", StandardScaler(), NUMERICAL),
        ],
        remainder="passthrough",
    )
    return preprocessor


def apply_smote(
    X_train: np.ndarray, y_train: np.ndarray, sampling_strategy="auto", random_state=42
) -> Tuple[np.ndarray, np.ndarray]:
    """对训练集应用 SMOTE 过采样

    Args:
        X_train: 训练特征
        y_train: 训练标签
        sampling_strategy: 'auto' 或 0.0~1.0 之间的比例
        random_state: 随机种子

    Returns:
        过采样后的 X_train, y_train
    """
    from imblearn.over_sampling import SMOTE

    before = dict(zip(*np.unique(y_train, return_counts=True)))
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    after = dict(zip(*np.unique(y_resampled, return_counts=True)))

    print(f"[features] SMOTE 过采样: {before} → {after}")
    return X_resampled, y_resampled


def prepare_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    val_size: float,
    random_state: int,
    selection_method: str = "none",
    correlation_threshold: float = 0.8,
    imbalance_method: str = "none",
    sampling_strategy: str = "auto",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], ColumnTransformer]:
    """完整特征工程流水线

    Args:
        df: 清洗后 DataFrame
        target_col: 目标列名
        test_size: 测试集比例
        val_size: 验证集比例
        random_state: 随机种子
        selection_method: 特征选择方法 ('correlation' | 'drop_low_importance' | 'none')
        correlation_threshold: 相关性阈值
        imbalance_method: 不均衡处理方法 ('smote' | 'class_weight' | 'none')
        sampling_strategy: SMOTE 采样策略

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names, preprocessor
    """
    # 分离特征与标签
    X = df.drop(columns=[target_col])
    y = df[target_col].values

    # 二元编码
    X = encode_binary(X)

    # 特征选择
    dropped_features = []
    if selection_method == "correlation":
        X, dropped = select_features_by_correlation(X, correlation_threshold)
        dropped_features.extend(dropped)
    elif selection_method == "drop_low_importance":
        X, dropped = select_features_by_importance(X)
        dropped_features.extend(dropped)

    # 更新类别列表（剔除已删除的特征）
    global MULTI_CATEGORICAL, NUMERICAL, BINARY_CATEGORICAL
    multi = [c for c in MULTI_CATEGORICAL if c in X.columns]
    num = [c for c in NUMERICAL if c in X.columns]
    binary = [c for c in BINARY_CATEGORICAL if c in X.columns]

    # 拆分 train/val/test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )

    # 构建预处理器（使用过滤后的列）
    from sklearn.preprocessing import OneHotEncoder as OHE, StandardScaler as SS
    from sklearn.compose import ColumnTransformer as CT

    preprocessor = build_preprocessor()
    # 重建 preprocessor（可能有列被删了，用 X_train 的列来决定）
    preprocessor = CT(
        transformers=[
            ("cat", OHE(drop="first", sparse_output=False), [c for c in MULTI_CATEGORICAL if c in X.columns]),
            ("num", SS(), [c for c in NUMERICAL if c in X.columns]),
        ],
        remainder="passthrough",
    )

    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(X_test)

    # SMOTE 过采样
    if imbalance_method == "smote":
        X_train, y_train = apply_smote(X_train, y_train, sampling_strategy, random_state)

    # 获取特征名
    cat_names = []
    cat_enc = preprocessor.named_transformers_.get("cat")
    if cat_enc is not None:
        cat_cols = [c for c in MULTI_CATEGORICAL if c in X.columns]
        if cat_cols:
            cat_names = cat_enc.get_feature_names_out(cat_cols).tolist()

    num_cols = [c for c in NUMERICAL if c in X.columns]
    bin_cols = [c for c in BINARY_CATEGORICAL if c in X.columns]
    # passthrough 还会包含其他未显式指定的列
    passthrough_cols = [c for c in X.columns if c not in set(cat_cols + num_cols)]
    feature_names = cat_names + num_cols + passthrough_cols

    print(f"[features] 训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
    print(f"[features] 特征数: {len(feature_names)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names, preprocessor
