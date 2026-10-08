"""
逻辑回归模型封装
算法族：线性模型
预测机制：对各特征加权求和后经 sigmoid 函数转换为概率
概率特征：对线性可分特征高度敏感，概率分布常呈两极化（接近 0 或 1）
特征重要性：使用系数绝对值 |coef_| 衡量各特征对预测的线性贡献
"""
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)


class LogisticRegressionModel:
    """逻辑回归模型类（线性模型）"""

    def __init__(self, params):
        """
        Args:
            params: LogisticRegression 参数字典
        """
        self.model = LogisticRegression(**params)
        self.name = '逻辑回归(LR)'
        self.key = 'lr'

    def train(self, X_train, y_train):
        """训练模型"""
        print(f"\n训练 {self.name} 模型...")
        self.model.fit(X_train, y_train)
        print(f"{self.name} 训练完成")

    def predict_proba(self, X):
        """返回正类概率（sigmoid 输出）"""
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
        获取特征重要性（系数绝对值 |coef_|，代表各特征对线性决策边界的贡献大小）
        Returns:
            DataFrame: feature / importance，按重要性降序排列
        """
        return pd.DataFrame({
            'feature':    feature_names,
            'importance': np.abs(self.model.coef_[0])
        }).sort_values('importance', ascending=False).head(top_n)

    def save(self, model_dir):
        """保存模型到 model_dir/logistic_regression_model.pkl"""
        path = os.path.join(model_dir, 'logistic_regression_model.pkl')
        joblib.dump(self.model, path)
        print(f"{self.name} 已保存: {path}")

    def load(self, model_dir):
        """从 model_dir/logistic_regression_model.pkl 加载模型"""
        path = os.path.join(model_dir, 'logistic_regression_model.pkl')
        self.model = joblib.load(path)
        print(f"{self.name} 已加载: {path}")
