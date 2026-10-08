"""
随机森林模型封装
算法族：Bagging 并行集成
预测机制：多棵决策树在随机特征子空间独立训练，最终以投票比例作为概率输出
概率特征：分布较保守，多集中在 0.4~0.8 之间
"""
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)


class RandomForestModel:
    """随机森林模型类（Bagging 并行集成）"""

    def __init__(self, params):
        """
        Args:
            params: RandomForestClassifier 参数字典
        """
        self.model = RandomForestClassifier(**params)
        self.name = '随机森林(RF)'
        self.key = 'rf'

    def train(self, X_train, y_train):
        """训练模型"""
        print(f"\n训练 {self.name} 模型...")
        self.model.fit(X_train, y_train)
        print(f"{self.name} 训练完成")

    def predict_proba(self, X):
        """返回正类概率"""
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold=0.5):
        """返回二分类预测结果"""
        return (self.predict_proba(X) >= threshold).astype(int)

    def evaluate(self, X_test, y_test):
        """
        评估模型性能
        Returns:
            dict: accuracy / precision / recall / f1 / auc / confusion_matrix / y_pred / y_pred_proba
        """
        y_pred_proba = self.predict_proba(X_test)
        y_pred = (y_pred_proba >= 0.5).astype(int)

        metrics = {
            'accuracy':         accuracy_score(y_test, y_pred),
            'precision':        precision_score(y_test, y_pred, zero_division=0),
            'recall':           recall_score(y_test, y_pred, zero_division=0),
            'f1':               f1_score(y_test, y_pred, zero_division=0),
            'auc':              roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'y_pred':           y_pred,
            'y_pred_proba':     y_pred_proba,
        }

        print(f"\n{self.name} 评估结果：")
        print(f"  准确率: {metrics['accuracy']:.4f}")
        print(f"  精确率: {metrics['precision']:.4f}")
        print(f"  召回率: {metrics['recall']:.4f}")
        print(f"  F1分数: {metrics['f1']:.4f}")
        print(f"  AUC:    {metrics['auc']:.4f}")

        return metrics

    def get_feature_importance(self, feature_names, top_n=20):
        """
        获取特征重要性（基于基尼不纯度降低量）
        Returns:
            DataFrame: feature / importance，按重要性降序排列
        """
        return pd.DataFrame({
            'feature':    feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)

    def save(self, model_dir):
        """保存模型到 model_dir/random_forest_model.pkl"""
        path = os.path.join(model_dir, 'random_forest_model.pkl')
        joblib.dump(self.model, path)
        print(f"{self.name} 已保存: {path}")

    def load(self, model_dir):
        """从 model_dir/random_forest_model.pkl 加载模型"""
        path = os.path.join(model_dir, 'random_forest_model.pkl')
        self.model = joblib.load(path)
        print(f"{self.name} 已加载: {path}")
