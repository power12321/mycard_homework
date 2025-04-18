import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib
matplotlib.use('TkAgg')  # 或 'QtAgg'，依你的系统而定

def main():
    print("========== [Model Evaluation] ==========")
    # 1. 读取数据和最优模型
    data = pd.read_csv('train_selected.csv')
    with open('best_model.pkl', 'rb') as f:
        best_model = pickle.load(f)  # 读取模型

    y_col = 'isDefault'
    X_cols = [c for c in data.columns if c not in ['id', y_col]]

    X = data[X_cols]
    y = data[y_col]

    # 2. 使用与训练脚本相同的 random_state 划分数据，保证一致性
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=2023, stratify=y
    )

    # 3. 转换数据为 DMatrix 格式（如果使用的是 Booster 模型）
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    # 4. 预测并计算 AUC（使用 predict 方法）
    train_pred = best_model.predict(dtrain)
    valid_pred = best_model.predict(dvalid)

    train_auc = roc_auc_score(y_train, train_pred)
    valid_auc = roc_auc_score(y_valid, valid_pred)
    print(f"训练集 AUC: {train_auc:.6f}")
    print(f"验证集 AUC: {valid_auc:.6f}")

    # 5. 可视化验证集 ROC 曲线
    fpr, tpr, _ = roc_curve(y_valid, valid_pred)
    plt.figure()
    plt.plot(fpr, tpr, label=f'Valid ROC (AUC={valid_auc:.4f})')
    plt.plot([0,1],[0,1],'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Validation ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

    print("✅ 模型评估和ROC曲线绘制完成")

if __name__ == "__main__":
    main()
