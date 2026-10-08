# 电商评论有用性分析

基于机器学习的电商评论有用性预测系统，结合文本特征工程与多种分类模型，自动识别高价值用户评论，并通过可视化大屏展示分析结果。

## 项目背景

电商平台中存在大量用户评论，但真正对消费者决策有帮助的评论占比不高。本项目旨在通过机器学习方法，自动对评论的"有用性"进行分类与预测，帮助平台优化评论排序、提升用户购物体验。

## 功能特性

- **数据爬取**：基于 Playwright 自动化采集电商平台评论数据
- **文本预处理**：中文分词、停用词过滤、TF-IDF 特征提取
- **多模型对比**：实现逻辑回归（LR）、随机森林（RF）、LightGBM 三种分类模型
- **模型可解释性**：基于 SHAP 值分析特征对预测结果的贡献
- **可视化大屏**：基于 Flask + ECharts 的交互式数据可视化界面
- **批量预测**：支持对新评论文件进行批量有用性打分

## 技术栈

| 类别 | 技术 |
|---|---|
| 编程语言 | Python 3.10+ |
| 数据爬取 | Playwright |
| 数据处理 | pandas、scikit-learn、NumPy |
| 机器学习 | LightGBM、scikit-learn（LR/RF） |
| 模型解释 | SHAP |
| 可视化 | Flask、ECharts、PyQt5 |
| 特征工程 | TF-IDF 向量化 |

## 项目结构

```
评论有用性分析毕设代码/
├── config.py                  # 配置文件（路径、模型参数等）
├── data_preprocessor.py       # 数据预处理与特征工程
├── fake_crawler.py            # 评论数据爬虫
├── train.py                  # 模型训练入口
├── model_trainer.py           # 模型训练流程封装
├── model_lr.py                # 逻辑回归模型
├── model_rf.py                # 随机森林模型
├── model_lgb.py               # LightGBM 模型
├── main.py                    # 主程序入口
├── requirements.txt           # 依赖清单
├── models/                    # 训练好的模型文件
│   ├── lightgbm_model.pkl
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── scaler.pkl
│   └── ...
├── outputs/                   # 实验结果输出
└── web/                       # Flask 可视化大屏
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

### 2. 训练模型

```bash
python train.py
```

训练完成后，模型文件自动保存到 `models/` 目录。

### 3. 启动可视化大屏

```bash
python web/app.py
```

浏览器访问 http://localhost:5000 即可查看数据可视化看板。

### 4. 批量预测

将待预测评论整理为 CSV 文件，运行批量预测脚本即可输出有用性评分。

## 模型说明

| 模型 | 优点 | 适用场景 |
|---|---|---|
| 逻辑回归（LR） | 训练快、可解释性强 | 基线模型、简单场景 |
| 随机森林（RF） | 抗过拟合、处理非线性关系 | 中等复杂度任务 |
| LightGBM | 训练速度快、精度高、支持类别特征 | 最终推荐模型 |

通过 SHAP 值分析可查看每个特征（评论长度、情感倾向、关键词等）对有用性预测的影响方向与权重。

## 实验结果

实验对比了三种模型在测试集上的表现，LightGBM 在准确率与 F1 值上均取得最优效果，同时具备较高的训练效率。

## 作者

- 数据科学与大数据技术专业
- 2026 届本科毕业设计
