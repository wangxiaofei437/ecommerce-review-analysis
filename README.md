# 电商评论有用性分析

基于机器学习的电商评论有用性预测系统。通过文本特征工程与多种分类模型集成，自动识别高价值用户评论，并基于 Flask + ECharts 提供交互式可视化看板。

## 项目背景

电商平台中存在大量用户评论，但真正对消费者决策有帮助的评论占比不高。本项目通过机器学习方法，自动对评论的"有用性"进行二分类预测，帮助平台优化评论排序、提升用户购物体验。

## 功能特性

- **多模型集成**：逻辑回归（LR）、随机森林（RF）、LightGBM 三模型加权融合
- **文本特征工程**：中文分词（jieba）+ TF-IDF 向量化
- **无数据泄露训练**：严格划分训练/测试集，避免特征泄漏
- **可视化看板**：基于 Flask + ECharts 的交互式数据大屏
- **批量预测**：支持对新评论 CSV 文件进行有用性打分
- **一键运行**：自动检测模型是否存在，未训练则自动启动训练流程

## 技术栈

| 类别 | 技术 |
|---|---|
| 编程语言 | Python 3.10+ |
| 数据处理 | pandas、NumPy、scikit-learn |
| 机器学习 | LightGBM、XGBoost、scikit-learn（LR/RF） |
| 中文分词 | jieba |
| Web 可视化 | Flask、ECharts |
| 模型持久化 | joblib / pickle |

## 项目结构

```
评论有用性分析毕设代码/
├── config.py                  # 全局配置（路径、模型参数、集成权重）
├── data_preprocessor.py       # 数据预处理与特征工程
├── fake_crawler.py            # 评论数据采集模块
├── train.py                   # 模型训练入口
├── model_trainer.py            # 训练流程封装
├── model_lr.py                 # 逻辑回归模型
├── model_rf.py                # 随机森林模型
├── model_lgb.py                # LightGBM 模型
├── main.py                    # 主启动脚本（自动检测模型）
├── requirements.txt            # Python 依赖清单
├── ecommerce_comments_relabeled.csv  # 训练数据集
├── models/                    # 训练好的模型文件
│   ├── lightgbm_model.pkl
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── tfidf_vectorizer.pkl   # TF-IDF 向量化器
│   ├── scaler.pkl             # 特征标准化器
│   └── feature_names.pkl
├── outputs/                    # 实验结果输出
└── web/                        # Flask 可视化大屏
    ├── app.py
    ├── templates/index.html
    └── static/
        ├── css/dashboard.css
        └── js/dashboard.js
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 一键运行

```bash
python main.py
```

系统会自动检测 `models/` 目录下是否已有预训练模型：
- 若已存在，直接跳过训练
- 若不存在，自动启动训练流程（首次运行需几分钟）

### 3. 启动可视化看板

```bash
python web/app.py
```

浏览器访问 http://localhost:5000 查看数据可视化结果。

### 4. 批量预测

将待预测评论整理为 CSV 文件，运行批量预测脚本即可输出有用性评分。

## 模型说明

采用**三模型加权集成**策略，根据各模型特点分配权重：

| 模型 | 权重 | 定位 |
|---|---|---|
| 逻辑回归（LR） | 0.30 | 线性模型，提供算法多样性基线 |
| 随机森林（RF） | 0.30 | Bagging 并行集成，抗过拟合 |
| LightGBM | 0.40 | 梯度提升树，精度最高，权重最大 |

### 关键参数设计

- **测试集比例**：20%
- **随机种子**：42（保证结果可复现）
- **有用性判定阈值**：评论获赞数 ≥ 3 视为"有用"
- **类别不平衡处理**：使用 `class_weight='balanced'`

### 目标准确率

- 逻辑回归：~91%
- 随机森林：~93%
- LightGBM：~94%
- 集成模型整体：90%+

## 设计要点

1. **无数据泄露**：特征工程严格在训练集上拟合，再应用到测试集
2. **模型欠拟合控制**：通过限制树深、叶子数、弱学习器数量，避免模型在小数据集上过拟合
3. **可复现性**：所有随机过程固定种子（`random_state=42`）

## 作者

- 数据科学与大数据技术专业
- 2026 届本科毕业设计
