"""
模型评估模块
@description 准确率、召回率、F1、AUC-ROC、混淆矩阵等指标计算
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from typing import Dict, Any


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """计算模型各项评估指标

    Args:
        model: 训练好的模型（需有 predict + predict_proba 方法）
        X_test: 测试集特征
        y_test: 测试集标签

    Returns:
        Dict: 评估指标
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]),
    }
    return metrics


def evaluate_all(
    model_results: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    """对所有模型进行评估并打印对比

    Args:
        model_results: train_models 返回的结果字典
        X_test: 测试集特征
        y_test: 测试集标签

    Returns:
        Dict: 模型名 → 指标字典
    """
    all_metrics = {}
    print("\n" + "=" * 70)
    print(f"{'Model':<25} {'Accuracy':>8} {'Precision':>8} {'Recall':>8} {'F1':>8} {'AUC':>8}")
    print("-" * 70)

    best_model = None
    best_auc = 0

    for name, result in model_results.items():
        model = result["model"]
        metrics = evaluate_model(model, X_test, y_test)
        all_metrics[name] = metrics

        print(
            f"{name:<25} {metrics['accuracy']:>8.4f} {metrics['precision']:>8.4f} "
            f"{metrics['recall']:>8.4f} {metrics['f1']:>8.4f} {metrics['auc_roc']:>8.4f}"
        )

        if metrics["auc_roc"] > best_auc:
            best_auc = metrics["auc_roc"]
            best_model = name

    print("-" * 70)
    print(f"最佳模型: {best_model} (AUC={best_auc:.4f})")
    print("=" * 70 + "\n")

    return all_metrics
