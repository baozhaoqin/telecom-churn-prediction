"""
消融实验 — 量化每个模块的贡献
依次测试不同配置组合，对比各模块对模型性能的影响
"""
import sys
import yaml
import warnings
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_data, clean_data
from src.features import prepare_data
from src.models import create_models, train_models, get_feature_importance
from src.evaluate import evaluate_model

warnings.filterwarnings("ignore")


# 实验配置组合
CONFIGS = [
    {
        "name": "Baseline (无优化)",
        "selection": "none",
        "imbalance": "none",
        "note": "原始数据 + 默认模型参数",
    },
    {
        "name": "+ 特征选择",
        "selection": "correlation",
        "imbalance": "none",
        "note": "相关性剔除冗余特征",
    },
    {
        "name": "+ SMOTE",
        "selection": "none",
        "imbalance": "smote",
        "note": "SMOTE 过采样处理不均衡",
    },
    {
        "name": "+ 特征选择 + SMOTE",
        "selection": "correlation",
        "imbalance": "smote",
        "note": "特征选择 + SMOTE 组合",
    },
]


def run_experiment(config: dict, seed: int, exp_name: str, selection: str, imbalance: str):
    """运行单个实验配置"""
    print(f"\n{'='*60}")
    print(f"消融实验: {exp_name}")
    print(f"{'='*60}")

    df = load_data(config["data"]["raw_path"])
    df = clean_data(df)

    X_train, X_val, X_test, y_train, y_val, y_test, feature_names, _ = prepare_data(
        df, target_col="Churn",
        test_size=config["model"]["test_size"],
        val_size=config["model"]["val_size"],
        random_state=seed,
        selection_method=selection,
        imbalance_method=imbalance,
    )

    models = create_models({
        "max_iter": 1000, "C": 1.0, "random_seed": seed,
        "n_estimators": 100, "max_depth": 10, "min_samples_split": 5,
        "learning_rate": 0.1, "subsample": 0.8,
    })

    model_results = train_models(models, X_train, y_train, X_val, y_val)

    # 评估
    results = {}
    for name, result in model_results.items():
        metrics = evaluate_model(result["model"], X_test, y_test)
        results[name] = {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "auc_roc": metrics["auc_roc"],
        }

    # 取 AUC 最高的模型
    best_name = max(results, key=lambda k: results[k]["auc_roc"])
    return best_name, results[best_name], feature_names


def main():
    config = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
    seed = config["random_seed"]

    all_results = []

    for cfg in CONFIGS:
        best_model, metrics, features = run_experiment(
            config, seed, cfg["name"], cfg["selection"], cfg["imbalance"]
        )
        all_results.append({
            "实验配置": cfg["name"],
            "最佳模型": best_model,
            "Accuracy": f"{metrics['accuracy']:.4f}",
            "Precision": f"{metrics['precision']:.4f}",
            "Recall": f"{metrics['recall']:.4f}",
            "F1": f"{metrics['f1']:.4f}",
            "AUC": f"{metrics['auc_roc']:.4f}",
            "特征数": len(features),
            "说明": cfg["note"],
        })

    # 输出汇总表
    df_results = pd.DataFrame(all_results)
    print("\n\n" + "=" * 90)
    print("消融实验汇总")
    print("=" * 90)
    print(df_results.to_string(index=False))

    # 保存
    output_path = "results/ablation.csv"
    df_results.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n[ablation] 结果已保存: {output_path}")

    # 计算各模块贡献
    print("\n=== 各模块贡献分析 ===")
    baseline_auc = float(all_results[0]["AUC"])
    for i, row in enumerate(all_results[1:], 1):
        auc = float(row["AUC"])
        delta = auc - baseline_auc
        print(f"  {row['实验配置']}: AUC={row['AUC']} (Δ={delta:+.4f})")

    return df_results


if __name__ == "__main__":
    main()
