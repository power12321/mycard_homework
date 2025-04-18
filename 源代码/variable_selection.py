# -*- coding: utf-8 -*-
"""
variable_selection.py

功能：
1. 读取 train_woe.csv，使用随机森林进行特征重要度排序（嵌入法）
2. 保留前 40 个最重要的特征
3. 输出 train_selected.csv / test_selected.csv
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def main():
    print("========== [Variable Selection] ==========")

    # 1. 读取数据
    train_woe = pd.read_csv('train_woe.csv')
    test_woe = pd.read_csv('test_woe.csv')

    y_col = 'isDefault'
    X_cols = [col for col in train_woe.columns if col not in ['id', y_col]]
    X_train = train_woe[X_cols]
    y_train = train_woe[y_col]

    # 2. 训练随机森林评估变量重要性
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=1000,
        random_state=2023,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_

    # 3. 保留前 40 个最重要的特征
    top_k = 40
    sorted_idx = np.argsort(importances)[::-1]  # 从大到小排序
    selected_features = [X_cols[i] for i in sorted_idx[:top_k]]

    print(f"特征选择后数量: {len(selected_features)} / {len(X_cols)}")
    print("保留的前 40 个特征如下：")
    print(selected_features)

    # 4. 构造新数据集
    train_selected = train_woe[['id', y_col] + selected_features]
    if 'isDefault' in test_woe.columns:
        test_selected = test_woe[['id', y_col] + selected_features]
    else:
        test_selected = test_woe[['id'] + selected_features]

    # 5. 保存结果
    train_selected.to_csv('train_selected.csv', index=False)
    test_selected.to_csv('test_selected.csv', index=False)

    print("✅ 特征选择完成 -> train_selected.csv / test_selected.csv")


if __name__ == "__main__":
    main()
