"""
模型训练与预测模块
@description 逻辑回归、随机森林、XGBoost 三模型训练与交叉验证
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
from typing import Dict, Any


def create_models(config: dict) -> Dict[str, Any]:
    """根据配置创建模型实例

    Args:
        config: 配置字典（含各模型超参数）

    Returns:
        Dict[str, Any]: 模型名称 -> 模型实例
    """
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=config.get("max_iter", 1000),
            C=config.get("C", 1.0),
            random_state=config.get("random_seed", 42),
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=config.get("n_estimators", 100),
            max_depth=config.get("max_depth", 10),
            min_samples_split=config.get("min_samples_split", 5),
            random_state=config.get("random_seed", 42),
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=config.get("n_estimators", 100),
            max_depth=config.get("max_depth", 6),
            learning_rate=config.get("learning_rate", 0.1),
            subsample=config.get("subsample", 0.8),
            random_state=config.get("random_seed", 42),
            eval_metric="logloss",
        ),
    }
    return models


def train_models(
    models: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, Any]:
    """训练所有模型并返回训练结果

    Args:
        models: 模型字典
        X_train: 训练特征
        y_train: 训练标签
        X_val: 验证特征
        y_val: 验证标签

    Returns:
        Dict: 模型名 -> {"model": 训练后模型, "train_score": 训练集准确率, "val_score": 验证集准确率}
    """
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)
        results[name] = {
            "model": model,
            "train_score": train_score,
            "val_score": val_score,
        }
        print(f"[models] {name}: train_acc={train_score:.4f}, val_acc={val_score:.4f}")

    return results


def cross_validate(
    models: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int,
    random_seed: int,
) -> Dict[str, Dict[str, float]]:
    """K 折交叉验证

    Args:
        models: 模型字典
        X_train: 训练特征
        y_train: 训练标签
        cv_folds: 折数
        random_seed: 随机种子

    Returns:
        Dict: 模型名 -> {"mean": 平均准确率, "std": 标准差}
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    cv_results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        cv_results[name] = {"mean": scores.mean(), "std": scores.std()}
        print(f"[models] {name} CV: {scores.mean():.4f} ± {scores.std():.4f}")

    return cv_results


def get_feature_importance(model, feature_names: list) -> dict:
    """提取特征重要性

    Args:
        model: 训练好的模型（需有 feature_importances_ 属性）
        feature_names: 特征名列表

    Returns:
        dict: 特征名 → 重要性分数，降序排列
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).flatten()
    else:
        return {}

    sorted_idx = np.argsort(importances)[::-1]
    return {feature_names[i]: importances[i] for i in sorted_idx}


# GridSearchCV 搜索空间定义
PARAM_GRIDS = {
    "LogisticRegression": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "max_iter": [1000],
        "class_weight": ["balanced", None],
    },
    "RandomForest": {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "class_weight": ["balanced", "balanced_subsample", None],
    },
    "XGBoost": {
        "n_estimators": [100, 200],
        "max_depth": [3, 6, 9],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "scale_pos_weight": [1, 3],
    },
}


def tune_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_seed: int,
    cv: int = 3,
    scoring: str = "roc_auc",
    n_jobs: int = -1,
    verbose: int = 1,
) -> Dict:
    """对三个模型执行 GridSearchCV 超参数搜索

    Args:
        X_train: 训练特征
        y_train: 训练标签
        random_seed: 随机种子
        cv: 交叉验证折数
        scoring: 评分指标
        n_jobs: 并行数
        verbose: 详细程度

    Returns:
        Dict: {model_name: {"best_model": ..., "best_params": ..., "best_score": ...}}
    """
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression as LR
    from sklearn.ensemble import RandomForestClassifier as RF
    from xgboost import XGBClassifier as XGB

    base_models = {
        "LogisticRegression": (LR(random_state=random_seed), PARAM_GRIDS["LogisticRegression"]),
        "RandomForest": (RF(random_state=random_seed, n_jobs=-1), PARAM_GRIDS["RandomForest"]),
        "XGBoost": (XGB(random_state=random_seed, eval_metric="logloss"), PARAM_GRIDS["XGBoost"]),
    }

    tuning_results = {}
    for name, (model, param_grid) in base_models.items():
        print(f"\n[models] GridSearchCV: {name} ({len(param_grid)} 参数 × {cv} 折 = ...)")
        gs = GridSearchCV(
            model, param_grid, cv=cv, scoring=scoring,
            n_jobs=n_jobs, verbose=verbose, refit=True,
        )
        gs.fit(X_train, y_train)

        tuning_results[name] = {
            "best_model": gs.best_estimator_,
            "best_params": gs.best_params_,
            "best_score": gs.best_score_,
        }
        print(f"[models] {name} 最佳 {scoring}: {gs.best_score_:.4f}")
        print(f"[models] {name} 最佳参数: {gs.best_params_}")

    return tuning_results
