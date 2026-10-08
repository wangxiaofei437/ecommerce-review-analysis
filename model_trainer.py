"""
模型训练调度模块 - 统一管理三个模型的训练、评估与融合

三种不同算法族（各自封装在独立模块中）：
  - model_rf.py   → RandomForestModel     （Bagging 并行集成）
  - model_lr.py   → LogisticRegressionModel（线性模型）
  - model_lgb.py  → LightGBMModel         （梯度提升，leaf-wise）
"""
import numpy as np
import joblib
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

from model_rf import RandomForestModel
from model_lr import LogisticRegressionModel
from model_lgb import LightGBMModel


class ModelTrainer:
    """模型训练调度类：统一调度三个独立模型的训练、融合与持久化"""

    def __init__(self, rf_params, lr_params, ensemble_weights, lgb_params=None):
        """
        Args:
            rf_params:         随机森林参数字典
            lr_params:         逻辑回归参数字典
            ensemble_weights:  融合权重字典 {'rf': float, 'lr': float, 'lgb': float}
            lgb_params:        LightGBM 参数字典（None 则不使用）
        """
        self.rf_model  = RandomForestModel(rf_params)
        self.lr_model  = LogisticRegressionModel(lr_params)
        self.lgb_model = LightGBMModel(lgb_params) if lgb_params else None

        self.ensemble_weights = ensemble_weights
        self.best_threshold = 0.5
        self.results = {}

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------

    def train_all_models(self, X_train, y_train, X_test, y_test):
        """训练所有模型并记录评估结果"""
        models = [self.rf_model, self.lr_model]
        if self.lgb_model is not None:
            models.append(self.lgb_model)

        for m in models:
            m.train(X_train, y_train)
            self.results[m.key] = m.evaluate(X_test, y_test)

    # ------------------------------------------------------------------
    # 阈值搜索
    # ------------------------------------------------------------------

    def find_best_threshold(self, X_val, y_val):
        """在验证集上搜索使融合模型 F1 最大的分类阈值"""
        _, ensemble_proba, _, _, _ = self._raw_ensemble_predict(X_val)
        best_thresh, best_f1 = 0.5, 0.0
        for thresh in np.arange(0.25, 0.75, 0.01):
            preds = (ensemble_proba >= thresh).astype(int)
            f1 = f1_score(y_val, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        self.best_threshold = best_thresh
        print(f"最优分类阈值: {best_thresh:.2f}  (F1={best_f1:.4f})")
        return best_thresh

    # ------------------------------------------------------------------
    # 预测
    # ------------------------------------------------------------------

    def _raw_ensemble_predict(self, X):
        """返回各模型原始概率及加权融合概率（不应用阈值）"""
        rf_proba  = self.rf_model.predict_proba(X)
        lr_proba  = self.lr_model.predict_proba(X)
        lgb_proba = self.lgb_model.predict_proba(X) if self.lgb_model else np.zeros(len(X))

        ensemble_proba = (
            self.ensemble_weights['rf']  * rf_proba +
            self.ensemble_weights['lr']  * lr_proba +
            self.ensemble_weights.get('lgb', 0.0) * lgb_proba
        )
        return None, ensemble_proba, rf_proba, lr_proba, lgb_proba

    def ensemble_predict(self, X):
        """
        三模型加权融合预测

        各模型概率特征（产生天然差异）：
          - RF ：投票比例，分布保守（0.4~0.8）
          - LR ：sigmoid 输出，对线性特征敏感，两极化
          - LGB：leaf-wise 增益，概率极端（0.05 或 0.95）

        Returns:
            (ensemble_pred, ensemble_proba, rf_proba, lr_proba, lgb_proba)
        """
        _, ensemble_proba, rf_proba, lr_proba, lgb_proba = self._raw_ensemble_predict(X)
        ensemble_pred = (ensemble_proba >= self.best_threshold).astype(int)
        return ensemble_pred, ensemble_proba, rf_proba, lr_proba, lgb_proba

    # ------------------------------------------------------------------
    # 融合模型评估
    # ------------------------------------------------------------------

    def evaluate_ensemble(self, X_test, y_test):
        """评估融合模型并将结果存入 self.results['ensemble']"""
        ensemble_pred, ensemble_proba, rf_proba, lr_proba, lgb_proba = self.ensemble_predict(X_test)

        metrics = {
            'accuracy':         accuracy_score(y_test, ensemble_pred),
            'precision':        precision_score(y_test, ensemble_pred, zero_division=0),
            'recall':           recall_score(y_test, ensemble_pred, zero_division=0),
            'f1':               f1_score(y_test, ensemble_pred, zero_division=0),
            'auc':              roc_auc_score(y_test, ensemble_proba),
            'confusion_matrix': confusion_matrix(y_test, ensemble_pred),
            'y_pred':           ensemble_pred,
            'y_pred_proba':     ensemble_proba,
            'rf_proba':         rf_proba,
            'lr_proba':         lr_proba,
            'lgb_proba':        lgb_proba,
        }

        print(f"\n融合模型评估结果（阈值={self.best_threshold:.2f}）：")
        print(f"  准确率: {metrics['accuracy']:.4f}")
        print(f"  精确率: {metrics['precision']:.4f}")
        print(f"  召回率: {metrics['recall']:.4f}")
        print(f"  F1分数: {metrics['f1']:.4f}")
        print(f"  AUC:    {metrics['auc']:.4f}")

        self.results['ensemble'] = metrics
        return metrics

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def save_models(self, model_dir):
        """保存所有模型及最优阈值"""
        os.makedirs(model_dir, exist_ok=True)
        self.rf_model.save(model_dir)
        self.lr_model.save(model_dir)
        if self.lgb_model:
            self.lgb_model.save(model_dir)
        joblib.dump(self.best_threshold, os.path.join(model_dir, 'best_threshold.pkl'))
        print(f"\n所有模型已保存到: {model_dir}")

    def load_models(self, model_dir):
        """加载所有模型及最优阈值"""
        try:
            self.rf_model.load(model_dir)
            self.lr_model.load(model_dir)

            lgb_path = os.path.join(model_dir, 'lightgbm_model.pkl')
            if self.lgb_model and os.path.exists(lgb_path):
                self.lgb_model.load(model_dir)
            else:
                self.lgb_model = None

            thresh_path = os.path.join(model_dir, 'best_threshold.pkl')
            if os.path.exists(thresh_path):
                self.best_threshold = joblib.load(thresh_path)

            print(f"所有模型已加载（阈值={self.best_threshold:.2f}）")
            return True
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False

    # ------------------------------------------------------------------
    # 特征重要性
    # ------------------------------------------------------------------

    def get_feature_importance(self, feature_names, top_n=20):
        """
        获取各模型特征重要性
        Returns:
            dict: {'rf': DataFrame, 'lr': DataFrame, 'lgb': DataFrame}
        """
        importance_dict = {
            'rf': self.rf_model.get_feature_importance(feature_names, top_n),
            'lr': self.lr_model.get_feature_importance(feature_names, top_n),
        }
        if self.lgb_model:
            importance_dict['lgb'] = self.lgb_model.get_feature_importance(feature_names, top_n)
        return importance_dict
