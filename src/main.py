"""
主入口模块
@description 整合数据加载、特征工程、模型训练、评估、可视化全流程
运行方式: python src/main.py
"""
import sys
import yaml
import warnings
from pathlib import Path

# 添加项目根目录到 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_data, clean_data, get_summary
from src.features import prepare_data
from src.models import create_models, train_models, cross_validate, get_feature_importance, tune_models
from src.tracking import log_experiment
from src.evaluate import evaluate_all, evaluate_model
from src.visualize import (
    plot_feature_importance,
    plot_roc_curves,
    plot_confusion_matrices,
    plot_churn_distribution,
)

warnings.filterwarnings("ignore")


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """加载 YAML 配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def main():
    # 1. 加载配置
    print("=" * 70)
    print("电信客户流失预测 — Telco Customer Churn Prediction")
    print("=" * 70)

    config = load_config()
    seed = config["random_seed"]
    print(f"\n[main] 随机种子: {seed}")

    # 2. 加载与清洗数据
    df = load_data(config["data"]["raw_path"])
    df = clean_data(df)

    # 数据摘要
    summary = get_summary(df)
    print(f"[main] 样本数: {summary['n_samples']}, 特征数: {summary['n_features']}")
    print(f"[main] 流失率: {summary['churn_rate']:.2%}")

    # 3. 特征工程（含特征选择 + SMOTE）
    feat_cfg = config.get("features", {})
    imb_cfg = config.get("imbalance", {})
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names, preprocessor = (
        prepare_data(
            df,
            target_col="Churn",
            test_size=config["model"]["test_size"],
            val_size=config["model"]["val_size"],
            random_state=seed,
            selection_method=feat_cfg.get("selection_method", "none"),
            correlation_threshold=feat_cfg.get("correlation_threshold", 0.8),
            imbalance_method=imb_cfg.get("method", "none"),
            sampling_strategy=imb_cfg.get("sampling_strategy", "auto"),
        )
    )

    # 4. 创建并训练模型
    models = create_models({
        "max_iter": config["logistic_regression"]["max_iter"],
        "C": config["logistic_regression"]["C"],
        "random_seed": seed,
        **{k: v for d in [config["random_forest"], config["xgboost"]] for k, v in d.items()},
    })

    model_results = train_models(models, X_train, y_train, X_val, y_val)

    # 4b. GridSearchCV 超参数调优（可选）
    best_params = {}
    gs_cfg = config.get("grid_search", {})
    if gs_cfg.get("enabled", False):
        tuning_results = tune_models(
            X_train, y_train,
            random_seed=seed,
            cv=gs_cfg.get("cv", 3),
            scoring=gs_cfg.get("scoring", "roc_auc"),
            n_jobs=gs_cfg.get("n_jobs", -1),
            verbose=gs_cfg.get("verbose", 1),
        )
        # 用调优后的模型替换
        for name, t_result in tuning_results.items():
            model_results[name] = {
                "model": t_result["best_model"],
                "train_score": t_result["best_model"].score(X_train, y_train),
                "val_score": t_result["best_model"].score(X_val, y_val),
            }
            best_params[name] = t_result["best_params"]
        print("\n[main] GridSearchCV 完成，已替换为最优模型")

    # 5. 交叉验证（使用最终模型）
    cv_results = cross_validate(
        models, X_train, y_train, config["model"]["cv_folds"], seed
    )

    # 6. 测试集评估
    all_metrics = evaluate_all(model_results, X_test, y_test)

    # 为 ROC 曲线补充 AUC
    for name in model_results:
        metrics = evaluate_model(model_results[name]["model"], X_test, y_test)
        model_results[name]["auc_roc"] = metrics["auc_roc"]

    # 7. 可视化
    viz_cfg = config["visualization"]
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # 流失分布
    plot_churn_distribution(
        df,
        save_path=str(results_dir / f"churn_distribution.{viz_cfg['save_format']}"),
        dpi=viz_cfg["figure_dpi"],
    )

    # 特征重要性（取最佳模型）
    best_model_name = max(all_metrics, key=lambda k: all_metrics[k]["auc_roc"])
    best_model = model_results[best_model_name]["model"]
    importance = get_feature_importance(best_model, feature_names)
    if importance:
        plot_feature_importance(
            importance,
            title=f"特征重要性排序 — {best_model_name}",
            save_path=str(results_dir / f"feature_importance.{viz_cfg['save_format']}"),
            figsize=tuple(viz_cfg["figsize"]),
            dpi=viz_cfg["figure_dpi"],
        )

    # ROC 曲线
    plot_roc_curves(
        model_results,
        X_test,
        y_test,
        save_path=str(results_dir / f"roc_curves.{viz_cfg['save_format']}"),
        figsize=(8, 6),
        dpi=viz_cfg["figure_dpi"],
    )

    # 混淆矩阵
    plot_confusion_matrices(
        model_results,
        X_test,
        y_test,
        save_path=str(results_dir / f"confusion_matrices.{viz_cfg['save_format']}"),
        dpi=viz_cfg["figure_dpi"],
    )

    # 8. 输出摘要
    print("\n[main] ====== 实验完成 ======")
    print(f"[main] 最佳模型: {best_model_name}")
    best_metrics = all_metrics[best_model_name]
    print(
        f"[main] Accuracy={best_metrics['accuracy']:.4f}, "
        f"Precision={best_metrics['precision']:.4f}, "
        f"Recall={best_metrics['recall']:.4f}, "
        f"F1={best_metrics['f1']:.4f}, "
        f"AUC={best_metrics['auc_roc']:.4f}"
    )
    print(f"\n[main] CV 结果:")
    for name, cv in cv_results.items():
        print(f"  {name}: {cv['mean']:.4f} ± {cv['std']:.4f}")

    # 9. 实验追踪
    log_experiment(
        config=config,
        model_name=best_model_name,
        metrics=best_metrics,
        cv_results=cv_results,
        best_params=best_params.get(best_model_name) if best_params else None,
        notes="feature_selection + SMOTE + GridSearchCV",
        output_path=str(results_dir / "experiments.csv"),
    )

    return model_results, all_metrics, cv_results


if __name__ == "__main__":
    main()
