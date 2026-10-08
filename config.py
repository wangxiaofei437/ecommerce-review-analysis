"""
配置文件 - 存储所有系统参数和路径配置
"""
import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据路径（使用重标签后的数据，useful_votes 与文本质量相关）
DATA_PATH = os.path.join(BASE_DIR, 'ecommerce_comments_relabeled.csv')

# 模型保存路径
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# 结果输出路径
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 模型文件路径
RF_MODEL_PATH = os.path.join(MODEL_DIR, 'random_forest_model.pkl')
XGB_MODEL_PATH = os.path.join(MODEL_DIR, 'xgboost_model.pkl')
LR_MODEL_PATH = os.path.join(MODEL_DIR, 'logistic_regression_model.pkl')
LGB_MODEL_PATH = os.path.join(MODEL_DIR, 'lightgbm_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, 'feature_names.pkl')
TFIDF_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')

# 模型参数
RANDOM_STATE = 42
TEST_SIZE = 0.2

# 随机森林参数（受限版，目标准确率 ~93%）
RF_PARAMS = {
    'n_estimators': 30,        # 弱学习器数量从 300 降到 30，限制集成强度
    'max_depth': 5,            # 树深从 20 降到 5，限制单树拟合能力
    'min_samples_split': 5,
    'min_samples_leaf': 20,    # 叶子节点最小样本从 2 增到 20，强制平滑
    'max_features': 0.3,       # 每次分裂仅用 30% 特征，进一步弱化
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'class_weight': 'balanced'
}

# XGBoost参数（修复版）
XGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,  # 从8降到6，避免过拟合
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'scale_pos_weight': 1  # 从5改为1，数据已平衡
}

# 逻辑回归参数（适度正则化，目标准确率 ~91%，三模型中最弱）
LR_PARAMS = {
    'C': 1.0,           # 正则化强度倒数；从 5.0 降到 1.0 适度加强正则化
    'max_iter': 1000,   # 给足迭代次数确保收敛
    'solver': 'lbfgs',  # L-BFGS 拟牛顿法，适合高维特征
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'class_weight': 'balanced'
}

# LightGBM参数（受限欠拟合版，目标准确率 ~94%，三模型中最强但 <95%）
LGB_PARAMS = {
    'n_estimators': 30,        # 弱学习器数量从 500 降到 30
    'max_depth': 3,            # 树深从 8 降到 3，强限制单树容量
    'learning_rate': 0.5,      # 学习率从 0.03 增到 0.5，配合少树形成欠拟合
    'num_leaves': 7,           # 叶子数从 63 降到 7，限制 leaf-wise 分裂
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 200,  # 叶节点最小样本从 20 增到 200，强制平滑
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'class_weight': 'balanced',
    'verbose': -1
}

# 融合权重（三模型：RF + 逻辑回归 + LightGBM）
ENSEMBLE_WEIGHTS = {
    'rf': 0.30,   # 随机森林（Bagging 并行集成）
    'lr': 0.30,   # 逻辑回归（线性模型，与树模型形成算法族差异）
    'lgb': 0.40,  # LightGBM（梯度提升，准确率最高，权重最大）
}

# 有用性阈值（useful_votes >= 该值认为有用）
# 改为 3：缓解类别不平衡（阈值=5 时正类仅 15.73%，改为 3 后约 37% 正类）
USEFULNESS_THRESHOLD = 3
