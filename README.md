# 电信客户流失预测 (Telco Customer Churn Prediction)

> **版本：v0.2 改进版** | EDA + 特征选择 + SMOTE + GridSearchCV + 消融实验 + 46 单元测试

基于机器学习的电信客户流失预测模型，使用逻辑回归、随机森林、XGBoost 三种算法对比。

## 当前进度

| 阶段 | 状态 | 说明 |
|------|------|------|
| 数据获取 | ✅ 完成 | Telco Customer Churn 数据集（7,043 条） |
| EDA 探索分析 | ✅ 完成 | 流失分布、数值/类别特征分析、相关性热力图 |
| 数据清洗 | ✅ 完成 | 缺失值处理、编码转换 |
| 特征工程 | ✅ 完成 | 独热编码 + 标准化 + 相关性特征选择 |
| 样本不均衡处理 | ✅ 完成 | SMOTE 过采样，Recall 提升 40% |
| 模型训练 | ✅ 完成 | 三模型 baseline + GridSearchCV 超参数调优 |
| 模型评估 | ✅ 完成 | 五指标对比 + 消融实验 + 4 张可视化图表 |
| 单元测试 | ✅ 完成 | 46 tests, 70% 代码覆盖率 |
| 实验追踪 | ✅ 完成 | CSV 自动记录实验参数与指标 |
| 论文 | ⬜ 待做 | 论文撰写与 PPT 制作 |

### Baseline vs 改进后

| 配置 | Accuracy | Precision | Recall | F1 | AUC |
|------|----------|-----------|--------|-----|------|
| Baseline (无优化) | 0.8020 | 0.6690 | 0.5027 | 0.5740 | 0.8436 |
| + 特征选择 | 0.8020 | 0.6667 | 0.5080 | 0.5766 | 0.8403 |
| + SMOTE | 0.7637 | 0.5424 | **0.7005** | 0.6114 | 0.8409 |
| **+ 特征选择 + SMOTE** | 0.7438 | 0.5117 | **0.7620** | **0.6122** | 0.8380 |

## 数据集

- **名称**: Telco Customer Churn (Kaggle)
- **规模**: 7,043 条样本，21 个特征
- **来源**: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **许可证**: 公开数据集，可用于学术研究

## 快速开始

### 1. 环境准备

打开命令行，进入项目根目录（在文件夹地址栏输入 `cmd` 回车，或用 `cd` 加拖拽文件夹的方式），执行：

```bash
pip install -r requirements.txt
```

### 2. 下载数据

从 [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) 下载 `WA_Fn-UseC_-Telco-Customer-Churn.csv`，放入 `data/raw/` 目录。

### 3. 运行

```bash
# 主流程（含 GridSearchCV 调优，约 3-5 分钟）
python src/main.py

# EDA 探索图表
python scripts/eda_charts.py

# 消融实验
python scripts/ablation.py
```

输出：
- 控制台：模型训练过程、GridSearchCV 最佳参数、交叉验证、对比表
- `results/` 目录：ROC 曲线、混淆矩阵、特征重要性、流失分布、EDA 图表
- `results/experiments.csv`：实验追踪记录
- `results/ablation.csv`：消融实验对比

### 4. 运行测试

```bash
python -m pytest tests/ -v
```

## 项目结构

```
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── config.yaml          # 全局配置（seed、特征选择、SMOTE、模型参数）
├── data/
│   ├── raw/                 # 原始数据（gitignore）
│   └── processed/           # 处理后数据（gitignore）
├── notebooks/
│   └── eda.ipynb            # EDA 探索分析
├── src/
│   ├── main.py              # 主入口，运行完整流程
│   ├── data.py              # 数据加载与清洗
│   ├── features.py          # 特征工程 + 特征选择 + SMOTE
│   ├── models.py            # 模型训练 + GridSearchCV
│   ├── evaluate.py          # 评估指标
│   ├── visualize.py         # 可视化
│   └── tracking.py          # 实验追踪 CSV
├── scripts/
│   ├── eda_charts.py        # EDA 图表生成（独立脚本）
│   ├── ablation.py          # 消融实验
│   └── generate_manual.py   # PDF 使用说明生成
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_evaluate.py
│   └── test_visualize.py
└── results/                 # 输出图表 + CSV
```

## 关键技术点

- **特征选择**：相关性分析剔除冗余（TotalCharges 与 MonthlyCharges 共线）
- **SMOTE 过采样**：解决 3:1 类别不均衡，Recall 从 0.50 → 0.76
- **GridSearchCV**：三模型超参数搜索（XGBoost CV-AUC 0.9337）
- **消融实验**：量化特征选择、SMOTE 各自贡献
- **实验追踪**：自动记录 timestamp + commit hash + 指标

## 复现步骤

1. `pip install -r requirements.txt`
2. 下载数据集放入 `data/raw/`
3. `python src/main.py`
4. 查看 `results/` 目录输出
5. （可选）`python -m pytest tests/ -v`

## AI 使用声明

本项目在以下环节使用了 AI 辅助（Claude）：
- 项目框架设计与代码结构规划
- 部分代码生成与优化
- 文档撰写与润色

所有 AI 生成代码均经过人工审核与修改。
