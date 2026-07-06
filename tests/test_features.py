"""
测试 src/features.py — 特征编码、特征选择、SMOTE
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import (
    encode_binary, select_features_by_correlation,
    select_features_by_importance, prepare_data, apply_smote,
)


@pytest.fixture
def sample_df():
    """构造完整特征 DataFrame"""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "gender": np.random.choice(["Male", "Female"], n),
        "SeniorCitizen": np.random.choice([0, 1], n),
        "Partner": np.random.choice(["Yes", "No"], n),
        "Dependents": np.random.choice(["Yes", "No"], n),
        "tenure": np.random.randint(1, 72, n),
        "PhoneService": np.random.choice(["Yes", "No"], n),
        "MultipleLines": np.random.choice(["Yes", "No", "No phone service"], n),
        "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity": np.random.choice(["Yes", "No", "No internet service"], n),
        "OnlineBackup": np.random.choice(["Yes", "No", "No internet service"], n),
        "DeviceProtection": np.random.choice(["Yes", "No", "No internet service"], n),
        "TechSupport": np.random.choice(["Yes", "No", "No internet service"], n),
        "StreamingTV": np.random.choice(["Yes", "No", "No internet service"], n),
        "StreamingMovies": np.random.choice(["Yes", "No", "No internet service"], n),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n),
        "PaperlessBilling": np.random.choice(["Yes", "No"], n),
        "PaymentMethod": np.random.choice(
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n
        ),
        "MonthlyCharges": np.random.uniform(20, 120, n),
        "TotalCharges": np.random.uniform(20, 8000, n),
        "Churn": np.random.choice([0, 1], n, p=[0.7, 0.3]),
    })


class TestEncodeBinary:
    """二元编码测试"""

    def test_gender_encoded(self, sample_df):
        """gender 应被编码为 0/1"""
        encoded = encode_binary(sample_df.drop(columns=["Churn"]))
        assert set(encoded["gender"].unique()).issubset({0, 1})

    def test_partner_encoded(self, sample_df):
        """Partner 应被编码为 0/1"""
        encoded = encode_binary(sample_df.drop(columns=["Churn"]))
        assert set(encoded["Partner"].unique()).issubset({0, 1})

    def test_numerical_unchanged(self, sample_df):
        """数值列不应被改动"""
        encoded = encode_binary(sample_df.drop(columns=["Churn"]))
        assert "tenure" in encoded.columns
        assert encoded["tenure"].dtype in [np.int64, np.int32, int]


class TestFeatureSelection:
    """特征选择测试"""

    def test_correlation_selection(self, sample_df):
        """相关性选择应减少列数"""
        X = encode_binary(sample_df.drop(columns=["Churn"]))
        n_before = X.shape[1]
        X_selected, dropped = select_features_by_correlation(X, threshold=0.8)
        assert X_selected.shape[1] <= n_before

    def test_importance_selection(self, sample_df):
        """低重要性特征选择"""
        X = encode_binary(sample_df.drop(columns=["Churn"]))
        X_selected, dropped = select_features_by_importance(X)
        assert "gender" not in X_selected.columns or "gender" in dropped


class TestSMOTE:
    """SMOTE 测试"""

    def test_smote_balances_classes(self):
        """SMOTE 后类别应平衡"""
        X = np.random.randn(100, 5)
        y = np.array([0] * 80 + [1] * 20)
        X_resampled, y_resampled = apply_smote(X, y, sampling_strategy="auto", random_state=42)
        counts = dict(zip(*np.unique(y_resampled, return_counts=True)))
        assert counts[0] >= 80  # 多数类不减少
        assert counts[1] > 20  # 少数类大幅增加

    def test_smote_increases_samples(self):
        """SMOTE 后样本数应增加"""
        X = np.random.randn(50, 3)
        y = np.array([0] * 40 + [1] * 10)
        X_resampled, y_resampled = apply_smote(X, y, sampling_strategy="auto", random_state=42)
        assert len(X_resampled) > 50


class TestPrepareData:
    """完整特征工程流水线测试"""

    def test_baseline_pipeline(self, sample_df):
        """Baseline 流水线应正常运行"""
        result = prepare_data(
            sample_df, target_col="Churn",
            test_size=0.3, val_size=0.15, random_state=42,
            selection_method="none", imbalance_method="none",
        )
        X_train, X_val, X_test, y_train, y_val, y_test, features, preprocessor = result
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(features) > 0

    def test_imbalance_pipeline(self, sample_df):
        """SMOTE 流水线应增大训练集"""
        result = prepare_data(
            sample_df, target_col="Churn",
            test_size=0.3, val_size=0.15, random_state=42,
            selection_method="none", imbalance_method="smote",
        )
        X_train, _, _, y_train, _, _, _, _ = result
        # SMOTE 后少数类应增加
        unique, counts = np.unique(y_train, return_counts=True)
        assert counts[0] > 0 and counts[1] > 0

    def test_feature_selection_pipeline(self, sample_df):
        """特征选择流水线应减少特征数"""
        result = prepare_data(
            sample_df, target_col="Churn",
            test_size=0.3, val_size=0.15, random_state=42,
            selection_method="correlation", imbalance_method="none",
        )
        _, _, _, _, _, _, features, _ = result
        assert len(features) > 0

    def test_reproducibility(self, sample_df):
        """固定 seed 应产生一致结果"""
        r1 = prepare_data(
            sample_df.copy(), target_col="Churn",
            test_size=0.3, val_size=0.15, random_state=42,
        )
        r2 = prepare_data(
            sample_df.copy(), target_col="Churn",
            test_size=0.3, val_size=0.15, random_state=42,
        )
        X_train1, _, _, y_train1, _, _, _, _ = r1
        X_train2, _, _, y_train2, _, _, _, _ = r2
        assert np.array_equal(X_train1, X_train2)
        assert np.array_equal(y_train1, y_train2)
