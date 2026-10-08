"""
可视化大屏后端 - Flask + 已训练模型
提供模型对比、混淆矩阵、概率分布、ROC、PR、F1-阈值、校准、实时预测、批量上传/导出 API
"""
import os
import sys
import io
import json
import time
import uuid
import numpy as np
import pandas as pd
import joblib
from flask import Flask, jsonify, render_template, request, send_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.calibration import calibration_curve

# 让 web/ 可以 import 上级目录的模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import (
    DATA_PATH, MODEL_DIR, SCALER_PATH, FEATURE_NAMES_PATH, TFIDF_PATH,
    RF_PARAMS, LR_PARAMS, LGB_PARAMS, ENSEMBLE_WEIGHTS,
    USEFULNESS_THRESHOLD, RANDOM_STATE, TEST_SIZE,
)
from data_preprocessor import DataPreprocessor
from model_trainer import ModelTrainer


app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64MB


# ---------------------------------------------------------------------------
# 全局缓存
# ---------------------------------------------------------------------------
STATE = {
    'trainer': None,
    'preprocessor': None,
    'feature_names': None,
}

# ACTIVE 是当前可视化使用的数据集（测试集 或 用户上传后预测过的数据）
# 结构: {
#   'source': 'test' | 'upload',
#   'name':   str,
#   'rows':   DataFrame (含原始评论 + 预测列),
#   'has_label': bool,
#   'y_true': np.array | None,
#   'proba':  {key: np.array, ...},  # 4 模型概率
#   'pred':   {key: np.array, ...},  # 4 模型预测
#   'metrics': {key: {...}, ...}     # 仅 has_label 时有效
# }
ACTIVE = {}

MODEL_KEYS = ['rf', 'lr', 'lgb', 'ensemble']
MODEL_NAMES = {
    'rf':       '随机森林',
    'lr':       '逻辑回归',
    'lgb':      'LightGBM',
    'ensemble': '融合模型',
}


# ---------------------------------------------------------------------------
# 初始化
# ---------------------------------------------------------------------------
def init_models():
    if STATE['trainer'] is not None:
        return

    print('[init] 加载已训练模型与预处理器 ...')
    preprocessor = DataPreprocessor(usefulness_threshold=USEFULNESS_THRESHOLD)
    preprocessor.scaler = joblib.load(SCALER_PATH)
    preprocessor.feature_names = joblib.load(FEATURE_NAMES_PATH)
    preprocessor.tfidf = joblib.load(TFIDF_PATH)
    cats = joblib.load(os.path.join(MODEL_DIR, 'categories.pkl'))
    preprocessor.platform_categories = cats['platform_categories']
    preprocessor.product_categories = cats['product_categories']
    preprocessor.sentiment_categories = cats['sentiment_categories']
    preprocessor.risk_categories = cats['risk_categories']

    trainer = ModelTrainer(RF_PARAMS, LR_PARAMS, ENSEMBLE_WEIGHTS, lgb_params=LGB_PARAMS)
    trainer.load_models(MODEL_DIR)

    STATE['trainer'] = trainer
    STATE['preprocessor'] = preprocessor
    STATE['feature_names'] = preprocessor.feature_names

    # 默认 active = 测试集
    print('[init] 用测试集生成默认 ACTIVE ...')
    df = preprocessor.load_data(DATA_PATH)
    df = df[~df['useful_votes'].isin([USEFULNESS_THRESHOLD])].reset_index(drop=True)
    df = preprocessor.create_target(df)
    X = preprocessor.engineer_features(df, is_training=False)
    y = df['is_useful']
    _, X_test, _, y_test, _, idx_test = train_test_split(
        X, y, df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    rows_test = df.loc[idx_test].reset_index(drop=True)
    set_active_from_dataframe(rows_test, name='默认测试集', source='test')
    print('[init] 完成。')


# ---------------------------------------------------------------------------
# 核心：对任意 DataFrame 跑预测并写入 ACTIVE
# ---------------------------------------------------------------------------
def set_active_from_dataframe(df_raw, name='上传数据', source='upload'):
    """对 df_raw 做特征工程 + 4 模型预测，写入 ACTIVE 全局状态。"""
    pre = STATE['preprocessor']
    trainer = STATE['trainer']

    df = df_raw.copy().reset_index(drop=True)

    # 标签判定（注意 numpy.bool_ 不可 JSON 序列化，必须转 Python bool）
    has_label = bool('useful_votes' in df.columns and df['useful_votes'].notna().any())
    if has_label:
        df['is_useful'] = (df['useful_votes'].fillna(0) >= USEFULNESS_THRESHOLD).astype(int)

    X = pre.engineer_features(df, is_training=False)
    Xs = pre.scaler.transform(X)

    rf_p  = trainer.rf_model.predict_proba(Xs)
    lr_p  = trainer.lr_model.predict_proba(Xs)
    lgb_p = trainer.lgb_model.predict_proba(Xs)
    ens_p = (
        ENSEMBLE_WEIGHTS['rf']  * rf_p +
        ENSEMBLE_WEIGHTS['lr']  * lr_p +
        ENSEMBLE_WEIGHTS['lgb'] * lgb_p
    )
    th = float(trainer.best_threshold)

    proba = {'rf': rf_p, 'lr': lr_p, 'lgb': lgb_p, 'ensemble': ens_p}
    pred = {
        'rf':  (rf_p  >= 0.5).astype(int),
        'lr':  (lr_p  >= 0.5).astype(int),
        'lgb': (lgb_p >= 0.5).astype(int),
        'ensemble': (ens_p >= th).astype(int),
    }

    # 把预测列写进展示用 dataframe
    df['pred_rf_proba']  = np.round(rf_p,  4)
    df['pred_lr_proba']  = np.round(lr_p,  4)
    df['pred_lgb_proba'] = np.round(lgb_p, 4)
    df['pred_ensemble_proba'] = np.round(ens_p, 4)
    df['pred_rf']        = pred['rf']
    df['pred_lr']        = pred['lr']
    df['pred_lgb']       = pred['lgb']
    df['pred_ensemble']  = pred['ensemble']
    df['pred_label']     = np.where(pred['ensemble'] == 1, '有用', '无用')

    metrics = {}
    y_true = df['is_useful'].values if has_label else None
    if has_label:
        for k in MODEL_KEYS:
            yp = pred[k]
            pp = proba[k]
            try:
                auc = float(roc_auc_score(y_true, pp))
            except Exception:
                auc = float('nan')
            metrics[k] = {
                'accuracy':  float(accuracy_score(y_true, yp)),
                'precision': float(precision_score(y_true, yp, zero_division=0)),
                'recall':    float(recall_score(y_true, yp, zero_division=0)),
                'f1':        float(f1_score(y_true, yp, zero_division=0)),
                'auc':       auc,
                'confusion_matrix': confusion_matrix(y_true, yp).tolist(),
            }

    ACTIVE.clear()
    ACTIVE.update({
        'source': source,
        'name': name,
        'rows': df,
        'has_label': has_label,
        'y_true': y_true,
        'proba': proba,
        'pred': pred,
        'metrics': metrics,
        'threshold': th,
        'updated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    })


# ---------------------------------------------------------------------------
# 页面
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')


# ---------------------------------------------------------------------------
# 数据源信息
# ---------------------------------------------------------------------------
@app.route('/api/source')
def api_source():
    init_models()
    return jsonify({
        'source': str(ACTIVE['source']),
        'name': str(ACTIVE['name']),
        'has_label': bool(ACTIVE['has_label']),
        'total': int(len(ACTIVE['rows'])),
        'updated_at': str(ACTIVE.get('updated_at', '')),
        'threshold': float(ACTIVE['threshold']),
    })


# ---------------------------------------------------------------------------
# 上传 / 导出 / 重置
# ---------------------------------------------------------------------------
REQUIRED_COLS = ['comment_text', 'rating', 'platform', 'product_category', 'comment_time',
                 'is_additional', 'reply_status', 'sentiment_label', 'problem_label', 'fake_risk_label']


@app.route('/api/upload_csv', methods=['POST'])
def api_upload_csv():
    init_models()
    if 'file' not in request.files:
        return jsonify({'error': '未收到文件，请使用 multipart/form-data 上传字段名为 file'}), 400
    f = request.files['file']
    if not f.filename.lower().endswith(('.csv', '.txt')):
        return jsonify({'error': '仅支持 CSV 文件'}), 400

    try:
        raw = f.read()
        # 兼容 utf-8-sig / gbk
        for enc in ('utf-8', 'utf-8-sig', 'gbk'):
            try:
                df = pd.read_csv(io.BytesIO(raw), encoding=enc)
                break
            except UnicodeDecodeError:
                df = None
        if df is None:
            return jsonify({'error': '无法识别文件编码，请保存为 UTF-8 后重试'}), 400
    except Exception as e:
        return jsonify({'error': f'读取 CSV 失败: {e}'}), 400

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return jsonify({'error': f'缺少必需列: {missing}'}), 400

    try:
        set_active_from_dataframe(df, name=f.filename, source='upload')
    except Exception as e:
        return jsonify({'error': f'预测失败: {e}'}), 500

    return jsonify({
        'ok': True,
        'name': ACTIVE['name'],
        'total': int(len(ACTIVE['rows'])),
        'has_label': ACTIVE['has_label'],
    })


@app.route('/api/download_predictions')
def api_download_predictions():
    init_models()
    if not ACTIVE:
        return jsonify({'error': '尚无数据'}), 400

    df = ACTIVE['rows']
    # 选择导出列：原始可读列 + 预测列
    keep = [c for c in [
        'comment_text', 'rating', 'platform', 'product_category', 'comment_time',
        'sentiment_label', 'fake_risk_label', 'useful_votes', 'is_useful',
        'pred_rf_proba', 'pred_lr_proba', 'pred_lgb_proba', 'pred_ensemble_proba',
        'pred_rf', 'pred_lr', 'pred_lgb', 'pred_ensemble', 'pred_label',
    ] if c in df.columns]

    buf = io.BytesIO()
    df[keep].to_csv(buf, index=False, encoding='utf-8-sig')
    buf.seek(0)
    fname = f"predictions_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    return send_file(buf, as_attachment=True, download_name=fname, mimetype='text/csv')


@app.route('/api/reset', methods=['POST'])
def api_reset():
    """切回默认测试集"""
    init_models()
    pre = STATE['preprocessor']
    df = pre.load_data(DATA_PATH)
    df = df[~df['useful_votes'].isin([USEFULNESS_THRESHOLD])].reset_index(drop=True)
    df = pre.create_target(df)
    X = pre.engineer_features(df, is_training=False)
    y = df['is_useful']
    _, _, _, _, _, idx_test = train_test_split(
        X, y, df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    rows_test = df.loc[idx_test].reset_index(drop=True)
    set_active_from_dataframe(rows_test, name='默认测试集', source='test')
    return jsonify({'ok': True})


@app.route('/api/preview')
def api_preview():
    init_models()
    n = int(request.args.get('n', 8))
    df = ACTIVE['rows'].head(n)
    cols = ['comment_text', 'rating', 'platform', 'product_category',
            'pred_ensemble_proba', 'pred_label']
    cols = [c for c in cols if c in df.columns]
    rows = df[cols].astype(object).where(pd.notna(df[cols]), None).to_dict(orient='records')
    return jsonify({'rows': rows, 'columns': cols})


# ---------------------------------------------------------------------------
# 指标 / KPI
# ---------------------------------------------------------------------------
@app.route('/api/metrics')
def api_metrics():
    init_models()
    if not ACTIVE['has_label']:
        return jsonify({'has_label': False, 'models': []})
    data = []
    for k in MODEL_KEYS:
        r = ACTIVE['metrics'][k]
        data.append({
            'key': k,
            'name': MODEL_NAMES[k],
            'accuracy':  round(r['accuracy'],  4),
            'precision': round(r['precision'], 4),
            'recall':    round(r['recall'],    4),
            'f1':        round(r['f1'],        4),
            'auc':       round(r['auc'],       4),
        })
    return jsonify({'has_label': True, 'models': data})


@app.route('/api/overview')
def api_overview():
    init_models()
    df = ACTIVE['rows']
    n_total = int(len(df))
    if ACTIVE['has_label']:
        n_useful = int(df['is_useful'].sum())
    else:
        n_useful = int((df['pred_ensemble'] == 1).sum())
    n_useless = n_total - n_useful

    platform_counts = df['platform'].value_counts().to_dict() if 'platform' in df.columns else {}
    category_counts = df['product_category'].value_counts().head(6).to_dict() if 'product_category' in df.columns else {}
    sentiment_counts = df['sentiment_label'].value_counts().to_dict() if 'sentiment_label' in df.columns else {}

    return jsonify({
        'total': n_total,
        'useful': n_useful,
        'useless': n_useless,
        'useful_ratio': round(n_useful / n_total, 4) if n_total else 0,
        'n_features': len(STATE['feature_names']),
        'n_models': 4,
        'has_label': ACTIVE['has_label'],
        'source_name': ACTIVE['name'],
        'platform': [{'name': k, 'value': int(v)} for k, v in platform_counts.items()],
        'category': [{'name': k, 'value': int(v)} for k, v in category_counts.items()],
        'sentiment': [{'name': k, 'value': int(v)} for k, v in sentiment_counts.items()],
    })


# ---------------------------------------------------------------------------
# 混淆矩阵
# ---------------------------------------------------------------------------
@app.route('/api/confusion/<model_key>')
def api_confusion(model_key):
    init_models()
    if model_key not in MODEL_KEYS:
        return jsonify({'error': 'unknown model'}), 404
    if not ACTIVE['has_label']:
        return jsonify({'has_label': False})
    cm = ACTIVE['metrics'][model_key]['confusion_matrix']
    data = [
        [0, 0, int(cm[0][0])],
        [0, 1, int(cm[0][1])],
        [1, 0, int(cm[1][0])],
        [1, 1, int(cm[1][1])],
    ]
    return jsonify({
        'has_label': True,
        'key': model_key,
        'name': MODEL_NAMES[model_key],
        'matrix': data,
        'labels': ['无用', '有用'],
    })


# ---------------------------------------------------------------------------
# 概率分布
# ---------------------------------------------------------------------------
@app.route('/api/proba_dist')
def api_proba_dist():
    """返回各模型预测概率的核密度估计 (Gaussian KDE)。
    前端用平滑曲线绘制概率分布对比，比直方图更连续直观。
    """
    init_models()
    grid = np.linspace(0.0, 1.0, 100)
    out = {'x': [round(float(v), 4) for v in grid], 'series': []}
    for k in MODEL_KEYS:
        proba = np.asarray(ACTIVE['proba'][k], dtype=float)
        # Silverman 经验带宽
        n = max(len(proba), 2)
        std = float(np.std(proba)) or 1e-3
        h = 1.06 * std * (n ** (-1.0 / 5.0))
        h = max(h, 0.01)
        # 抽样防止 O(N*M) 过慢
        if n > 4000:
            idx = np.random.RandomState(42).choice(n, 4000, replace=False)
            sample = proba[idx]
        else:
            sample = proba
        # KDE: density(x) = (1/(n*h)) * sum( phi((x - xi)/h) )
        diff = (grid[:, None] - sample[None, :]) / h
        density = np.exp(-0.5 * diff * diff).sum(axis=1) / (len(sample) * h * np.sqrt(2 * np.pi))
        out['series'].append({
            'key': k,
            'name': MODEL_NAMES[k],
            'density': [round(float(v), 5) for v in density],
        })
    return jsonify(out)


# ---------------------------------------------------------------------------
# ROC
# ---------------------------------------------------------------------------
@app.route('/api/roc')
def api_roc():
    init_models()
    if not ACTIVE['has_label']:
        return jsonify({'has_label': False, 'curves': []})
    y_true = ACTIVE['y_true']
    out = []
    for k in MODEL_KEYS:
        proba = ACTIVE['proba'][k]
        fpr, tpr, _ = roc_curve(y_true, proba)
        if len(fpr) > 120:
            idx = np.linspace(0, len(fpr) - 1, 120).astype(int)
            fpr, tpr = fpr[idx], tpr[idx]
        out.append({
            'key': k,
            'name': MODEL_NAMES[k],
            'auc': round(float(ACTIVE['metrics'][k]['auc']), 4),
            'points': [[round(float(x), 4), round(float(y), 4)] for x, y in zip(fpr, tpr)],
        })
    return jsonify({'has_label': True, 'curves': out})


# ---------------------------------------------------------------------------
# PR 曲线（新增）
# ---------------------------------------------------------------------------
@app.route('/api/pr')
def api_pr():
    init_models()
    if not ACTIVE['has_label']:
        return jsonify({'has_label': False, 'curves': []})
    y_true = ACTIVE['y_true']
    out = []
    for k in MODEL_KEYS:
        proba = ACTIVE['proba'][k]
        precision, recall, _ = precision_recall_curve(y_true, proba)
        ap = float(average_precision_score(y_true, proba))
        if len(precision) > 120:
            idx = np.linspace(0, len(precision) - 1, 120).astype(int)
            precision, recall = precision[idx], recall[idx]
        # 曲线上点 = (recall, precision)
        out.append({
            'key': k,
            'name': MODEL_NAMES[k],
            'ap': round(ap, 4),
            'points': [[round(float(r), 4), round(float(p), 4)]
                       for r, p in zip(recall, precision)],
        })
    # 基线 = 正例占比
    baseline = float(np.mean(y_true))
    return jsonify({'has_label': True, 'baseline': round(baseline, 4), 'curves': out})


# ---------------------------------------------------------------------------
# F1 / Precision / Recall vs Threshold（新增）
# ---------------------------------------------------------------------------
@app.route('/api/threshold_curve')
def api_threshold_curve():
    init_models()
    if not ACTIVE['has_label']:
        return jsonify({'has_label': False, 'series': []})
    y_true = ACTIVE['y_true']
    thresholds = np.linspace(0.05, 0.95, 37)  # 步长 0.025
    out = []
    for k in MODEL_KEYS:
        proba = ACTIVE['proba'][k]
        f1s, precs, recs = [], [], []
        for t in thresholds:
            yp = (proba >= t).astype(int)
            f1s.append(float(f1_score(y_true, yp, zero_division=0)))
            precs.append(float(precision_score(y_true, yp, zero_division=0)))
            recs.append(float(recall_score(y_true, yp, zero_division=0)))
        # 找此模型 F1 最优的阈值
        best_i = int(np.argmax(f1s))
        out.append({
            'key': k,
            'name': MODEL_NAMES[k],
            'f1':       [round(v, 4) for v in f1s],
            'precision':[round(v, 4) for v in precs],
            'recall':   [round(v, 4) for v in recs],
            'best_threshold': round(float(thresholds[best_i]), 3),
            'best_f1': round(f1s[best_i], 4),
        })
    return jsonify({
        'has_label': True,
        'thresholds': [round(float(t), 3) for t in thresholds],
        'series': out,
        'ensemble_threshold': float(ACTIVE['threshold']),
    })


# ---------------------------------------------------------------------------
# 校准曲线（新增）
# ---------------------------------------------------------------------------
@app.route('/api/calibration')
def api_calibration():
    init_models()
    if not ACTIVE['has_label']:
        return jsonify({'has_label': False, 'series': []})
    y_true = ACTIVE['y_true']
    out = []
    for k in MODEL_KEYS:
        proba = ACTIVE['proba'][k]
        # 防止仅有单一类别时 calibration_curve 报错
        n_bins = 10
        try:
            true_p, pred_p = calibration_curve(y_true, proba, n_bins=n_bins, strategy='quantile')
        except Exception:
            true_p, pred_p = np.array([]), np.array([])
        # Brier 分数 -- 越低越校准
        brier = float(np.mean((proba - y_true) ** 2))
        out.append({
            'key': k,
            'name': MODEL_NAMES[k],
            'brier': round(brier, 4),
            'points': [[round(float(p), 4), round(float(t), 4)]
                       for p, t in zip(pred_p, true_p)],
        })
    return jsonify({'has_label': True, 'series': out})


# ---------------------------------------------------------------------------
# 准确率 3D 柱状图（新增 - 提供单值供前端做 3D 柱效果）
# ---------------------------------------------------------------------------
@app.route('/api/accuracy_bar')
def api_accuracy_bar():
    init_models()
    if not ACTIVE['has_label']:
        return jsonify({'has_label': False, 'models': []})
    out = []
    for k in MODEL_KEYS:
        m = ACTIVE['metrics'][k]
        out.append({
            'key': k,
            'name': MODEL_NAMES[k],
            'accuracy':  round(m['accuracy'],  4),
            'precision': round(m['precision'], 4),
            'recall':    round(m['recall'],    4),
            'f1':        round(m['f1'],        4),
            'auc':       round(m['auc'],       4),
        })
    return jsonify({'has_label': True, 'models': out})


# ---------------------------------------------------------------------------
# 特征重要性
# ---------------------------------------------------------------------------
@app.route('/api/feature_importance')
def api_feature_importance():
    init_models()
    top_n = int(request.args.get('top_n', 12))
    importance = STATE['trainer'].get_feature_importance(STATE['feature_names'], top_n=top_n)
    out = {}
    for k, df in importance.items():
        out[k] = {
            'name': MODEL_NAMES[k],
            'features': df['feature'].tolist(),
            'importances': [round(float(v), 6) for v in df['importance'].tolist()],
        }
    return jsonify(out)


# ---------------------------------------------------------------------------
# 随机预测
# ---------------------------------------------------------------------------
@app.route('/api/random_predict')
def api_random_predict():
    init_models()
    df = ACTIVE['rows']
    idx = int(request.args.get('idx', -1))
    if idx < 0 or idx >= len(df):
        idx = int(np.random.randint(0, len(df)))

    row = df.iloc[idx]
    rf_p  = float(row['pred_rf_proba'])
    lr_p  = float(row['pred_lr_proba'])
    lgb_p = float(row['pred_lgb_proba'])
    ens_p = float(row['pred_ensemble_proba'])
    th = float(ACTIVE['threshold'])
    true_label = int(row['is_useful']) if 'is_useful' in df.columns else -1

    return jsonify({
        'idx': idx,
        'text': str(row.get('comment_text', '')),
        'platform': str(row.get('platform', '')),
        'category': str(row.get('product_category', '')),
        'rating': int(row.get('rating', 0)),
        'useful_votes': int(row.get('useful_votes', 0)) if 'useful_votes' in df.columns and pd.notna(row.get('useful_votes')) else 0,
        'true_label': true_label,
        'has_label': ACTIVE['has_label'],
        'threshold': round(th, 3),
        'predictions': [
            {'key': 'rf',  'name': MODEL_NAMES['rf'],  'proba': round(rf_p,  4), 'pred': int(rf_p  >= 0.5)},
            {'key': 'lr',  'name': MODEL_NAMES['lr'],  'proba': round(lr_p,  4), 'pred': int(lr_p  >= 0.5)},
            {'key': 'lgb', 'name': MODEL_NAMES['lgb'], 'proba': round(lgb_p, 4), 'pred': int(lgb_p >= 0.5)},
            {'key': 'ensemble', 'name': MODEL_NAMES['ensemble'], 'proba': round(ens_p, 4), 'pred': int(ens_p >= th)},
        ],
    })


# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    init_models()
    app.run(host='127.0.0.1', port=5000, debug=False)
