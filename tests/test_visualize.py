"""
测试 src/visualize.py — 可视化函数输出文件
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.visualize import (
    plot_feature_importance, plot_roc_curves, plot_confusion_matrices,
    plot_churn_distribution,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


@pytest.fixture
def results_dir(tmp_path):
    """临时结果目录"""
    return str(tmp_path)


@pytest.fixture
def sample_data():
    """生成测试用数据和模型"""
    np.random.seed(42)
    X_test = np.random.randn(50, 8)
    y_test = np.random.randint(0, 2, 50)

    lr = LogisticRegression(max_iter=500, random_state=42)
    rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    lr.fit(X_test, y_test)
    rf.fit(X_test, y_test)

    model_results = {
        "LogisticRegression": {"model": lr, "train_score": 0.72, "val_score": 0.68},
        "RandomForest": {"model": rf, "train_score": 0.95, "val_score": 0.72},
    }
    return model_results, X_test, y_test


@pytest.fixture
def churn_df():
    """流失分布测试数据"""
    return pd.DataFrame({"Churn": [0, 0, 0, 1, 0, 1, 0, 0]})


class TestPlotFeatureImportance:
    """特征重要性图测试"""

    def test_creates_file(self, results_dir):
        """应生成图片文件"""
        importance = {f"特征_{i}": 0.1 * i for i in range(10)}
        path = f"{results_dir}/test_importance.png"
        plot_feature_importance(importance, "测试标题", path, top_n=5, dpi=72)
        assert Path(path).exists()

    def test_file_not_empty(self, results_dir):
        """生成的文件不应为空"""
        importance = {"A": 0.5, "B": 0.3, "C": 0.2}
        path = f"{results_dir}/test_imp2.png"
        plot_feature_importance(importance, "test", path, dpi=72)
        assert Path(path).stat().st_size > 0


class TestPlotROCCurves:
    """ROC 曲线测试"""

    def test_creates_file(self, sample_data, results_dir):
        """应生成 ROC 曲线图片"""
        model_results, X_test, y_test = sample_data
        path = f"{results_dir}/test_roc.png"
        plot_roc_curves(model_results, X_test, y_test, path, dpi=72)
        assert Path(path).exists()

    def test_file_not_empty(self, sample_data, results_dir):
        """生成的文件不应为空"""
        model_results, X_test, y_test = sample_data
        path = f"{results_dir}/test_roc2.png"
        plot_roc_curves(model_results, X_test, y_test, path, dpi=72)
        assert Path(path).stat().st_size > 0


class TestPlotConfusionMatrices:
    """混淆矩阵测试"""

    def test_creates_file(self, sample_data, results_dir):
        """应生成混淆矩阵图片"""
        model_results, X_test, y_test = sample_data
        path = f"{results_dir}/test_cm.png"
        plot_confusion_matrices(model_results, X_test, y_test, path, dpi=72)
        assert Path(path).exists()

    def test_single_model(self, sample_data, results_dir):
        """单模型也应生成图片"""
        model_results, X_test, y_test = sample_data
        single = {"LR": model_results["LogisticRegression"]}
        path = f"{results_dir}/test_cm_single.png"
        plot_confusion_matrices(single, X_test, y_test, path, dpi=72)
        assert Path(path).exists()


class TestPlotChurnDistribution:
    """流失分布图测试"""

    def test_creates_file(self, churn_df, results_dir):
        """应生成饼图"""
        path = f"{results_dir}/test_pie.png"
        plot_churn_distribution(churn_df, path, dpi=72)
        assert Path(path).exists()

    def test_file_not_empty(self, churn_df, results_dir):
        """生成的文件不应为空"""
        path = f"{results_dir}/test_pie2.png"
        plot_churn_distribution(churn_df, path, dpi=72)
        assert Path(path).stat().st_size > 0
