"""
EDA 图表生成脚本 — 使用与 visualize.py 一致的字体配置
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# 先设 seaborn 样式，再设中文字体（sns.set_style 会重置 font.family！）
sns.set_style("whitegrid")
candidates = ["SimHei", "Microsoft YaHei", "STXihei", "STSong", "FangSong", "KaiTi"]
available = {f.name for f in fm.fontManager.ttflist}
for font in candidates:
    if font in available:
        plt.rcParams["font.family"] = font
        break
plt.rcParams["axes.unicode_minus"] = False

from src.data import load_data, clean_data

RESULTS = Path("results")
RESULTS.mkdir(parents=True, exist_ok=True)
DPI = 150


def main():
    print("=== EDA 图表生成 ===")
    print(f"[eda] 使用字体: {plt.rcParams['font.family']}")

    # 加载数据
    df = load_data("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = clean_data(df)

    # ---- 1. 流失分布 ----
    print("[eda] 生成 流失分布图...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    churn_counts = df["Churn"].value_counts()
    axes[0].pie(churn_counts.values, labels=["留存", "流失"], colors=["#2ecc71", "#e74c3c"],
                autopct="%1.1f%%", startangle=90, explode=(0, 0.05))
    axes[0].set_title("客户流失比例", fontsize=13, fontweight="bold")

    bars = axes[1].bar(["留存", "流失"], churn_counts.values, color=["#2ecc71", "#e74c3c"])
    for bar, val in zip(bars, churn_counts.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50, str(val),
                     ha="center", fontsize=12, fontweight="bold")
    axes[1].set_title("客户流失数量", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("人数")
    plt.tight_layout()
    fig.savefig(RESULTS / "eda_churn_distribution.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # ---- 2. 数值特征分布 ----
    print("[eda] 生成 数值特征分布图...")
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for i, col in enumerate(num_cols):
        df[df["Churn"] == 0][col].hist(ax=axes[0, i], bins=30, alpha=0.6, color="#2ecc71", label="留存")
        df[df["Churn"] == 1][col].hist(ax=axes[0, i], bins=30, alpha=0.6, color="#e74c3c", label="流失")
        axes[0, i].set_title(f"{col} 分布（按流失分组）", fontsize=11)
        axes[0, i].legend()
        df.boxplot(column=col, by="Churn", ax=axes[1, i], patch_artist=True)
        axes[1, i].set_title(f"{col} 箱线图", fontsize=11)
        axes[1, i].set_xlabel("Churn")
    plt.suptitle("")
    plt.tight_layout()
    fig.savefig(RESULTS / "eda_numerical.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # ---- 3. 类别特征 vs 流失率 ----
    print("[eda] 生成 类别特征流失率图...")
    cat_cols = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod",
    ]
    churn_by_cat = {}
    for col in cat_cols:
        churn_rate = df.groupby(col)["Churn"].mean().sort_values(ascending=False)
        churn_by_cat[col] = churn_rate

    top_features = sorted(
        churn_by_cat.keys(),
        key=lambda c: churn_by_cat[c].max() - churn_by_cat[c].min(),
        reverse=True,
    )[:8]

    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    axes = axes.flatten()
    for ax, col in zip(axes, top_features):
        data = churn_by_cat[col].sort_values()
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(data)))
        ax.barh(range(len(data)), data.values, color=colors)
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(data.index)
        ax.set_xlabel("流失率")
        ax.set_title(f"{col} 各取值流失率", fontsize=11, fontweight="bold")
        ax.axvline(df["Churn"].mean(), color="black", linestyle="--", alpha=0.5,
                   label=f"平均 ({df['Churn'].mean():.1%})")
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(RESULTS / "eda_categorical.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # ---- 4. 相关性热力图 ----
    print("[eda] 生成 相关性热力图...")
    df_encoded = df.copy()
    for col in cat_cols:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].astype("category").cat.codes
    corr = df_encoded.corr()

    churn_corr = corr["Churn"].drop("Churn").sort_values(key=abs, ascending=False)
    print("\n[eda] 与 Churn 相关性 Top 10:")
    for feat, val in churn_corr.head(10).items():
        print(f"  {feat}: {val:+.3f}")

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("特征相关性热力图", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(RESULTS / "eda_correlation.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[eda] 全部 4 张 EDA 图已保存到 {RESULTS}/")
    print("[eda] 完成!")


if __name__ == "__main__":
    main()
