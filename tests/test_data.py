"""
测试 src/data.py — 数据加载、清洗、摘要
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_data, clean_data, get_summary


@pytest.fixture
def sample_df():
    """构造测试用 DataFrame"""
    data = {
        "customerID": ["A001", "A002", "A003", "A004"],
        "gender": ["Female", "Male", "Female", "Male"],
        "SeniorCitizen": [0, 1, 0, 0],
        "tenure": [1, 34, 2, 45],
        "MonthlyCharges": [29.85, 56.95, 53.85, 42.30],
        "TotalCharges": ["29.85", "1889.50", "", "1840.75"],
        "Churn": ["No", "No", "Yes", "No"],
    }
    return pd.DataFrame(data)


class TestLoadData:
    """数据加载测试"""

    def test_load_real_data(self):
        """加载真实数据集"""
        df = load_data("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
        assert df is not None
        assert len(df) == 7043
        assert "Churn" in df.columns

    def test_load_data_shape(self):
        """验证加载后行列数"""
        df = load_data("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
        assert df.shape == (7043, 21)


class TestCleanData:
    """数据清洗测试"""

    def test_clean_removes_customer_id(self, sample_df):
        """清洗后应删除 customerID"""
        cleaned = clean_data(sample_df)
        assert "customerID" not in cleaned.columns

    def test_clean_converts_totalcharges(self, sample_df):
        """TotalCharges 应转为数值"""
        cleaned = clean_data(sample_df)
        assert cleaned["TotalCharges"].dtype in [np.float64, float]

    def test_clean_handles_empty_totalcharges(self, sample_df):
        """空 TotalCharges 应被填充"""
        cleaned = clean_data(sample_df)
        assert cleaned["TotalCharges"].notna().all()

    def test_clean_encodes_churn(self, sample_df):
        """Churn 应编码为 0/1"""
        cleaned = clean_data(sample_df)
        assert cleaned["Churn"].dtype in [np.int64, int]
        assert set(cleaned["Churn"].unique()).issubset({0, 1})

    def test_clean_churn_values(self):
        """真实数据 Churn 含 Yes/No"""
        df = load_data("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
        assert "Yes" in df["Churn"].values
        cleaned = clean_data(df)
        assert 1 in cleaned["Churn"].values
        assert 0 in cleaned["Churn"].values


class TestGetSummary:
    """数据摘要测试"""

    def test_get_summary_keys(self, sample_df):
        """摘要应包含预期字段"""
        cleaned = clean_data(sample_df)
        summary = get_summary(cleaned)
        assert "n_samples" in summary
        assert "n_features" in summary
        assert "churn_rate" in summary
        assert "missing_values" in summary
        assert "dtypes" in summary

    def test_get_summary_sample_count(self, sample_df):
        """样本数应正确"""
        cleaned = clean_data(sample_df)
        summary = get_summary(cleaned)
        assert summary["n_samples"] == 4

    def test_get_summary_churn_rate(self, sample_df):
        """流失率应正确"""
        cleaned = clean_data(sample_df)
        summary = get_summary(cleaned)
        assert summary["churn_rate"] == 0.25
