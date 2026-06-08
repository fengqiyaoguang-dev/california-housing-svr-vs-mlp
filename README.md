# 数据挖掘 · 实践环节三：神经网络与 SVM 回归预测

> 加州房价（California Housing）回归预测：**SVR vs MLP Regressor** 对比实验

本项目为数据挖掘课程「实践环节三」作业，基于 Python + scikit-learn 完成。

---

## 1. 项目简介

使用 **支持向量回归（SVR）** 与 **多层感知器回归（MLP Regressor）** 在 California Housing 数据集上完成回归预测，从 MSE / RMSE / MAE / R² 四项指标与预测效率两个维度进行对比分析。

- 核心流程：数据探索（EDA） -> 特征工程 -> 网格搜索（GridSearchCV）超参数优化 -> 模型训练 / 测试 -> 对比分析。
- 运行环境：Python 3.13 + PyCharm，依赖见 requirements.txt。

## 2. 数据集

- 来源：GitHub 仓库 ageron/handson-ml2（Hands-On Machine Learning 教材配套数据），源自 1990 年美国加州人口普查。
- 样本数量：20,640 条街区（block group）数据。
- 特征：8 个数值特征（经度、纬度、房屋中位年龄、总房间数、总卧室数、人口、家庭数、收入中位数）+ 1 个分类特征（近海程度）。
- 目标：median_house_value（房价中位数，单位：美元，范围 $14,999 ~ $500,001，存在截断）。
- 预处理：
  - 缺失值（total_bedrooms 共 207 条，占 1.0%）-> 中位数填充
  - 分类变量 ocean_proximity -> 独热编码（One-Hot Encoding，drop='first'）
  - 数值特征与目标变量 -> StandardScaler 标准化
  - 训练 / 测试集：80% / 20% 随机划分（random_state=42），训练集 16,512 条、测试集 4,128 条。

## 3. 环境与依赖

```bash
pip install -r requirements.txt
```

主要依赖：numpy、pandas、scikit-learn、matplotlib、seaborn、scipy。

## 4. 项目结构

```
├── data_exploration.py          # 1 数据探索 + EDA + 预处理 + 切分
├── svr_model.py                 # 2 SVR 模型训练 + 超参数搜索
├── mlp_regressor_model.py       # 3 MLP 回归器训练 + 超参数搜索
├── compare_models.py            # 4 SVR 与 MLP 结果对比 + 性能图表
├── outputs/                     # 运行产物（图表、npz、原始数据）
├── template_text.txt            # 实践1报告模板文本
├── requirements.txt             # Python 依赖
├── .gitignore                   # Git 忽略规则
└── README.md                    # 本文件
```

## 5. 运行步骤

```bash
python data_exploration.py   # 1 数据探索与预处理
python svr_model.py          # 2 SVR 模型训练
python mlp_regressor_model.py # 3 MLP 回归器训练
python compare_models.py     # 4 SVR vs MLP 性能对比
```

运行结束后，所有图表和 .npz 结果会存放在 outputs/ 目录。

## 6. 主要结论

- MLP 在 R²、RMSE、MAE 三项精度指标上全面优于 SVR（R² 约 0.79 vs 0.76）。
- MLP 预测速度比 SVR 快约 200 倍（SVR 需对数千支持向量做核函数计算）。
- 两种模型在 $500k 附近偏差较大，源于原始数据的截断处理。
- 特征共线性（total_rooms 与 households 相关系数 0.92）对 SVR-RBF 和 MLP 影响有限。

## 7. 许可

课程作业项目，仅用于学习与教学。
