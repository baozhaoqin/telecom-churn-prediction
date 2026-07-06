"""
测试 src/models.py — 模型创建、训练、交叉验证、特征重要性
"""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import (
    create_models, train_models, cross_validate, get_feature_importance
)


@pytest.fixture
def config():
    return {
        "max_iter": 100, "C": 0.1, "random_seed": 42,
        "n_estimators": 10, "max_depth": 3, "min_samples_split": 5,
        "learning_rate": 0.1, "subsample": 0.8,
    }


@pytest.fixture
def sample_data():
    """生成简单二分类数据"""
    np.random.seed(42)
    X_train = np.random.randn(100, 5)
    y_train = np.random.randint(0, 2, 100)
    X_val = np.random.randn(30, 5)
    y_val = np.random.randint(0, 2, 30)
    return X_train, y_train, X_val, y_val


class TestCreateModels:
    """模型创建测试"""

    def test_creates_three_models(self, config):
        """应创建 3 个模型"""
        models = create_models(config)
        assert len(models) == 3
        assert "LogisticRegression" in models
        assert "RandomForest" in models
        assert "XGBoost" in models

    def test_models_have_fit_method(self, config):
        """每个模型应有 fit 方法"""
        models = create_models(config)
        for name, model in models.items():
            assert hasattr(model, "fit"), f"{name} 缺少 fit 方法"

    def test_models_have_predict_method(self, config):
        """每个模型应有 predict 方法"""
        models = create_models(config)
        for name, model in models.items():
            assert hasattr(model, "predict"), f"{name} 缺少 predict 方法"

    def test_logistic_regression_max_iter(self, config):
        """逻辑回归 max_iter 应可配置"""
        models = create_models({**config, "max_iter": 500})
        assert models["LogisticRegression"].max_iter == 500


class TestTrainModels:
    """模型训练测试"""

    def test_train_returns_results(self, config, sample_data):
        """训练应返回结果字典"""
        X_train, y_train, X_val, y_val = sample_data
        models = create_models(config)
        results = train_models(models, X_train, y_train, X_val, y_val)
        assert len(results) == 3
        for name in results:
            assert "model" in results[name]
            assert "train_score" in results[name]
            assert "val_score" in results[name]

    def test_train_score_between_0_and_1(self, config, sample_data):
        """训练/验证分数应在 0-1 之间"""
        X_train, y_train, X_val, y_val = sample_data
        models = create_models(config)
        results = train_models(models, X_train, y_train, X_val, y_val)
        for name in results:
            assert 0 <= results[name]["train_score"] <= 1
            assert 0 <= results[name]["val_score"] <= 1


class TestCrossValidate:
    """交叉验证测试"""

    def test_cv_returns_dict(self, config, sample_data):
        """CV 应返回字典"""
        X_train, y_train, _, _ = sample_data
        models = create_models(config)
        train_models(models, X_train, y_train, X_train, y_train)
        cv_results = cross_validate(models, X_train, y_train, cv_folds=3, random_seed=42)
        assert len(cv_results) == 3
        for name in cv_results:
            assert "mean" in cv_results[name]
            assert "std" in cv_results[name]

    def test_cv_mean_between_0_and_1(self, config, sample_data):
        """CV 均值应在 0-1 之间"""
        X_train, y_train, _, _ = sample_data
        models = create_models(config)
        train_models(models, X_train, y_train, X_train, y_train)
        cv_results = cross_validate(models, X_train, y_train, cv_folds=3, random_seed=42)
        for name in cv_results:
            assert 0 <= cv_results[name]["mean"] <= 1


class TestFeatureImportance:
    """特征重要性测试"""

    def test_random_forest_importance(self, config, sample_data):
        """随机森林应有特征重要性"""
        X_train, y_train, _, _ = sample_data
        models = create_models(config)
        model = models["RandomForest"]
        model.fit(X_train, y_train)
        importance = get_feature_importance(model, [f"feat_{i}" for i in range(5)])
        assert len(importance) == 5
        assert all(v >= 0 for v in importance.values())

    def test_logistic_regression_importance(self, config, sample_data):
        """逻辑回归应有系数"""
        X_train, y_train, _, _ = sample_data
        models = create_models(config)
        model = models["LogisticRegression"]
        model.fit(X_train, y_train)
        importance = get_feature_importance(model, [f"feat_{i}" for i in range(5)])
        assert len(importance) == 5

    def test_importance_sorted_descending(self, config, sample_data):
        """重要性应按降序排列"""
        X_train, y_train, _, _ = sample_data
        models = create_models(config)
        model = models["RandomForest"]
        model.fit(X_train, y_train)
        importance = get_feature_importance(model, [f"feat_{i}" for i in range(5)])
        values = list(importance.values())
        assert values == sorted(values, reverse=True)
