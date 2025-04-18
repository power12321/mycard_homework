import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pickle

def main():
    print("========== [Model Training with GPU Acceleration] ==========")

    # 1. 读取数据
    data = pd.read_csv('train_selected.csv')
    y_col = 'isDefault'
    X_cols = [col for col in data.columns if col not in ['id', y_col]]

    X = data[X_cols]
    y = data[y_col]

    # 2. 划分训练集和验证集
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=2023, stratify=y
    )

    # 3. 转换成 DMatrix 格式，XGBoost要求的数据格式
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    # 4. 设置XGBoost参数，启用GPU
    params = {
        'objective': 'binary:logistic',  # 二分类
        'eval_metric': 'auc',  # 使用AUC作为评价指标
        'max_depth': 5,  # 树的最大深度
        'learning_rate': 0.005,  # 学习率
        'subsample': 0.75,  # 每棵树的训练样本比例
        'colsample_bytree': 0.75,  # 每棵树的特征选择比例
        'reg_alpha': 1,  # L1正则化
        'reg_lambda': 2,  # L2正则化
        'n_estimators': 4000,  # 表示在训练时要使用的树的最大数量
        'scale_pos_weight': 1,  # 不平衡类别时的加权系数
        'tree_method': 'gpu_hist',  # 启用GPU加速
        'gpu_id': 0  # 使用第一个GPU
    }

    # 5. 设置早停法
    early_stopping_rounds = 1000
    evals = [(dvalid, 'eval'), (dtrain, 'train')]

    # 6. 使用XGBoost训练模型，并打印每一轮的AUC
    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=11111,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=True,  # 每轮训练后显示AUC
        evals_result=evals_result  # 存储评估结果
    )

    # 7. 获取验证集 AUC 最优轮次
    best_iteration = model.best_iteration
    best_auc = evals_result['eval']['auc'][best_iteration]

    print(f"最优验证集 AUC: {best_auc:.6f} (第 {best_iteration+1} 轮)")

    # 8. 使用最佳迭代次数的模型
    best_model = model
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    # 9. 验证集 AUC
    valid_pred = best_model.predict(dvalid)
    valid_auc = roc_auc_score(y_valid, valid_pred)
    print(f"验证集 AUC: {valid_auc:.6f}")

    print("✅ 模型训练完成，已保存最佳模型 -> best_model.pkl")

if __name__ == "__main__":
    main()