"""
测试 src/evaluate.py — 评估指标计算
"""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import evaluate_model, evaluate_all
from sklearn.linear_model import LogisticRegression


@pytest.fixture
def trained_model_and_data():
    """训练一个简单的逻辑回归模型"""
    np.random.seed(42)
    X_train = np.random.randn(200, 3)
    y_train = np.random.randint(0, 2, 200)
    X_test = np.random.randn(50, 3)
    y_test = np.random.randint(0, 2, 50)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model, X_test, y_test


class TestEvaluateModel:
    """单模型评估测试"""

    def test_returns_all_metrics(self, trained_model_and_data):
        """应返回所有预期指标"""
        model, X_test, y_test = trained_model_and_data
        metrics = evaluate_model(model, X_test, y_test)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "auc_roc" in metrics
        assert "confusion_matrix" in metrics

    def test_accuracy_range(self, trained_model_and_data):
        """Accuracy 应在 0-1 之间"""
        model, X_test, y_test = trained_model_and_data
        metrics = evaluate_model(model, X_test, y_test)
        assert 0 <= metrics["accuracy"] <= 1

    def test_confusion_matrix_shape(self, trained_model_and_data):
        """混淆矩阵应为 2x2"""
        model, X_test, y_test = trained_model_and_data
        metrics = evaluate_model(model, X_test, y_test)
        cm = metrics["confusion_matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2

    def test_confusion_matrix_sum(self, trained_model_and_data):
        """混淆矩阵元素和应等于样本数"""
        model, X_test, y_test = trained_model_and_data
        metrics = evaluate_model(model, X_test, y_test)
        cm = metrics["confusion_matrix"]
        assert sum(sum(row) for row in cm) == len(y_test)


class TestEvaluateAll:
    """多模型评估测试"""

    def test_evaluate_all_returns_dict(self, trained_model_and_data):
        """应返回所有模型的指标"""
        model, X_test, y_test = trained_model_and_data
        model_results = {
            "ModelA": {"model": model},
            "ModelB": {"model": model},
        }
        all_metrics = evaluate_all(model_results, X_test, y_test)
        assert len(all_metrics) == 2
        assert "ModelA" in all_metrics

    def test_evaluate_all_consistent(self, trained_model_and_data):
        """相同模型应得到相同指标"""
        model, X_test, y_test = trained_model_and_data
        model_results = {
            "M1": {"model": model},
            "M2": {"model": model},
        }
        all_metrics = evaluate_all(model_results, X_test, y_test)
        assert all_metrics["M1"]["accuracy"] == all_metrics["M2"]["accuracy"]
