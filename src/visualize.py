"""
可视化模块
@description 特征重要性、ROC 曲线、混淆矩阵、流失分布图
所有图表标注使用中文。
"""
import matplotlib
matplotlib.use("Agg")  # 非交互后端，保存到文件
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List


def _setup_chinese_font():
    """配置中文字体，确保中文标注正常显示

    优先级：系统字体 → 项目自带 Noto Sans SC
    """
    from pathlib import Path

    # 1. 尝试系统自带中文字体
    candidates = ["SimHei", "Microsoft YaHei", "STXihei", "STSong", "FangSong", "KaiTi",
                  "Noto Sans SC", "Noto Sans CJK SC", "WenQuanYi Micro Hei",
                  "PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            plt.rcParams["axes.unicode_minus"] = False
            return

    # 2. 回退：加载项目自带字体
    bundled_font = Path(__file__).resolve().parent.parent / "assets" / "fonts" / "NotoSansSC.ttf"
    if bundled_font.exists():
        fm.fontManager.addfont(str(bundled_font))
        plt.rcParams["font.family"] = "Noto Sans SC"
        plt.rcParams["axes.unicode_minus"] = False
        return

    plt.rcParams["axes.unicode_minus"] = False


_setup_chinese_font()

# 模型名称中英映射
MODEL_NAMES_ZH = {
    "LogisticRegression": "逻辑回归",
    "RandomForest": "随机森林",
    "XGBoost": "XGBoost",
}


def _model_label(name: str) -> str:
    """将模型英文名转为中文标签"""
    return MODEL_NAMES_ZH.get(name, name)


def plot_feature_importance(
    importance_dict: dict,
    title: str,
    save_path: str,
    top_n: int = 15,
    figsize: tuple = (10, 8),
    dpi: int = 150,
):
    """绘制特征重要性条形图

    Args:
        importance_dict: 特征名 → 重要性分数
        title: 图标题
        save_path: 保存路径
        top_n: 展示前 N 个特征
        figsize: 图片尺寸
        dpi: 分辨率
    """
    items = list(importance_dict.items())[:top_n]
    names = [item[0] for item in items]
    scores = [item[1] for item in items]

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(names)))
    ax.barh(range(len(names)), scores, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("特征重要性")
    ax.set_title(title)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] 特征重要性图已保存: {save_path}")


def plot_roc_curves(
    model_results: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: str,
    figsize: tuple = (8, 6),
    dpi: int = 150,
):
    """绘制多模型 ROC 曲线对比

    Args:
        model_results: train_models 返回的结果字典
        X_test: 测试集特征
        y_test: 测试集标签
        save_path: 保存路径
        figsize: 图片尺寸
        dpi: 分辨率
    """
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#2ecc71", "#3498db", "#e74c3c"]

    for (name, result), color in zip(model_results.items(), colors):
        model = result["model"]
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = result.get("auc_roc", 0)
        zh_name = _model_label(name)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{zh_name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="随机分类器")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("假阳率 (False Positive Rate)")
    ax.set_ylabel("真阳率 (True Positive Rate)")
    ax.set_title("ROC 曲线对比")
    ax.legend(loc="lower right")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] ROC 曲线图已保存: {save_path}")


def plot_confusion_matrices(
    model_results: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: str,
    figsize: tuple = (14, 4),
    dpi: int = 150,
):
    """绘制多模型混淆矩阵

    Args:
        model_results: train_models 返回的结果字典
        X_test: 测试集特征
        y_test: 测试集标签
        save_path: 保存路径
        figsize: 图片尺寸
        dpi: 分辨率
    """
    n_models = len(model_results)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    if n_models == 1:
        axes = [axes]

    for ax, (name, result) in zip(axes, model_results.items()):
        model = result["model"]
        y_pred = model.predict(X_test)
        from sklearn.metrics import confusion_matrix as cm

        cm_mat = cm(y_test, y_pred)
        sns.heatmap(
            cm_mat,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["未流失", "流失"],
            yticklabels=["未流失", "流失"],
        )
        ax.set_title(_model_label(name))
        ax.set_xlabel("预测值")
        ax.set_ylabel("真实值")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] 混淆矩阵已保存: {save_path}")


def plot_churn_distribution(
    df, save_path: str, figsize: tuple = (6, 6), dpi: int = 150
):
    """绘制流失分布饼图

    Args:
        df: 含 Churn 列的 DataFrame
        save_path: 保存路径
        figsize: 图片尺寸
        dpi: 分辨率
    """
    churn_counts = df["Churn"].value_counts()
    labels = ["未流失", "流失"]
    colors = ["#2ecc71", "#e74c3c"]

    fig, ax = plt.subplots(figsize=figsize)
    wedges, texts, autotexts = ax.pie(
        churn_counts.values,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        explode=(0, 0.05),
    )
    # 百分比文字样式
    for autotext in autotexts:
        autotext.set_fontsize(12)
        autotext.set_fontweight("bold")
    ax.set_title("客户流失分布", fontsize=14, fontweight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] 流失分布图已保存: {save_path}")
