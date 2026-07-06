# 电信客户流失预测 (Telco Customer Churn Prediction)

> **版本：基础跑通版本 v0.1** | 全流程可运行，模型调优与深度分析待后续迭代

基于机器学习的电信客户流失预测模型，使用逻辑回归、随机森林、XGBoost 三种算法对比。

## 当前进度

| 阶段 | 状态 | 说明 |
|------|------|------|
| 数据获取 | ✅ 完成 | Telco Customer Churn 数据集（7,043 条） |
| 数据清洗 | ✅ 完成 | 缺失值处理、编码转换 |
| 特征工程 | ✅ 基础完成 | 独热编码 + 标准化 + train/val/test 划分 |
| 模型训练 | ✅ 基础完成 | 逻辑回归 / 随机森林 / XGBoost baseline + 交叉验证 |
| 模型评估 | ✅ 基础完成 | 五指标对比 + 4 张可视化图表 |
| 模型调优 | ⬜ 待做 | 超参数调优、样本不均衡处理、特征选择 |
| 深度分析 | ⬜ 待做 | EDA 探索分析、消融实验、统计显著性检验 |
| 论文 | ⬜ 待做 | 论文撰写与 PPT 制作 |

### 当前最佳结果

| 模型 | Accuracy | Precision | Recall | F1 | AUC |
|------|----------|-----------|--------|-----|------|
| 逻辑回归 | 0.8048 | 0.6581 | 0.5508 | 0.5997 | 0.8426 |
| 随机森林 | 0.7906 | 0.6561 | 0.4439 | 0.5295 | **0.8429** |
| XGBoost | 0.7935 | 0.6379 | 0.5134 | 0.5689 | 0.8363 |

## 数据集

- **名称**: Telco Customer Churn (Kaggle)
- **规模**: 7,043 条样本，21 个特征
- **来源**: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **许可证**: 公开数据集，可用于学术研究

### 数据字段

| 字段 | 类型 | 说明 |
|------|------|------|
| customerID | str | 客户ID（建模时剔除） |
| gender | cat | 性别 |
| SeniorCitizen | int | 是否老年人 |
| Partner | cat | 是否有伴侣 |
| Dependents | cat | 是否有家属 |
| tenure | int | 入网月数 |
| PhoneService | cat | 是否开通电话服务 |
| MultipleLines | cat | 是否多线 |
| InternetService | cat | 互联网服务类型 |
| OnlineSecurity | cat | 在线安全 |
| OnlineBackup | cat | 在线备份 |
| DeviceProtection | cat | 设备保护 |
| TechSupport | cat | 技术支持 |
| StreamingTV | cat | 流媒体电视 |
| StreamingMovies | cat | 流媒体电影 |
| Contract | cat | 合同类型 |
| PaperlessBilling | cat | 电子账单 |
| PaymentMethod | cat | 支付方式 |
| MonthlyCharges | float | 月费用 |
| TotalCharges | float | 总费用 |
| **Churn** | cat | **目标变量（是否流失）** |

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
python src/main.py
```

输出：
- 控制台：模型训练过程、交叉验证结果、测试集评估指标对比
- `results/` 目录：ROC 曲线、混淆矩阵、特征重要性图、流失分布图

## 项目结构

```
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── config.yaml          # 全局配置（随机种子、模型参数）
├── data/
│   ├── raw/                 # 原始数据
│   └── processed/           # 处理后数据
├── notebooks/               # Jupyter 探索分析
├── src/
│   ├── main.py              # 主入口，运行完整流程
│   ├── data.py              # 数据加载与清洗
│   ├── features.py          # 特征工程（编码、缩放、划分）
│   ├── models.py            # 模型训练、交叉验证
│   ├── evaluate.py          # 评估指标（准确率/召回率/F1/AUC）
│   └── visualize.py         # 可视化（ROC/混淆矩阵/特征重要性）
├── tests/
└── results/                 # 输出图表
```

## 模型

| 模型 | 说明 |
|------|------|
| 逻辑回归 | 线性基线模型 |
| 随机森林 | 集成学习，特征重要性分析 |
| XGBoost | 梯度提升，通常效果最佳 |

## 复现步骤

1. `pip install -r requirements.txt`
2. 下载数据集放入 `data/raw/`
3. `python src/main.py`
4. 查看 `results/` 目录输出

## AI 使用声明

本项目在以下环节使用了 AI 辅助（Claude）：
- 项目框架设计与代码结构规划
- 部分代码生成与优化
- 文档撰写与润色

所有 AI 生成代码均经过人工审核与修改。
