"""
实验追踪模块
@description 每次实验自动记录参数、指标、commit hash 到 CSV
"""
import csv
import datetime
import subprocess
from pathlib import Path
from typing import Dict, Any


def get_commit_hash() -> str:
    """获取当前 git commit 短哈希"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()[:8] if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def log_experiment(
    config: dict,
    model_name: str,
    metrics: Dict[str, Any],
    cv_results: Dict[str, Any] = None,
    best_params: Dict[str, Any] = None,
    notes: str = "",
    output_path: str = "results/experiments.csv",
):
    """将单次实验记录追加到 CSV

    Args:
        config: 配置字典
        model_name: 最佳模型名
        metrics: 最佳模型评估指标
        cv_results: 交叉验证结果
        best_params: 最佳超参数
        notes: 备注
        output_path: 输出 CSV 路径
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "commit_hash": get_commit_hash(),
        "selection_method": config.get("features", {}).get("selection_method", "none"),
        "imbalance_method": config.get("imbalance", {}).get("method", "none"),
        "best_model": model_name,
        "accuracy": f"{metrics.get('accuracy', 0):.4f}",
        "precision": f"{metrics.get('precision', 0):.4f}",
        "recall": f"{metrics.get('recall', 0):.4f}",
        "f1": f"{metrics.get('f1', 0):.4f}",
        "auc_roc": f"{metrics.get('auc_roc', 0):.4f}",
        "cv_mean": "",
        "cv_std": "",
        "best_params": str(best_params) if best_params else "",
        "notes": notes,
    }

    # CV 结果（取最佳模型的）
    if cv_results and model_name in cv_results:
        row["cv_mean"] = f"{cv_results[model_name]['mean']:.4f}"
        row["cv_std"] = f"{cv_results[model_name]['std']:.4f}"

    # 写入 CSV（文件不存在则写表头）
    file_exists = Path(output_path).exists()
    with open(output_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"[tracking] 实验记录已保存: {output_path}")
    print(f"[tracking] commit={row['commit_hash']}, model={model_name}, AUC={row['auc_roc']}")
