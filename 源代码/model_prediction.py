import pandas as pd
import pickle
import xgboost as xgb

def main():
    print("========== [Model Prediction on Test Set] ==========")
    # 1. 读取最优模型
    with open('best_model.pkl', 'rb') as f:
        best_model = pickle.load(f)

    # 2. 读取测试集数据
    test_df = pd.read_csv('test_selected.csv')

    # 3. 提取特征列（测试集可能没有 isDefault，但若有也不使用）
    y_col = 'isDefault'
    X_cols = [col for col in test_df.columns if col not in ['id', y_col]]
    X_test = test_df[X_cols]

    # 4. 将测试数据转换为 DMatrix 格式
    dtest = xgb.DMatrix(X_test)

    # 5. 预测违约概率 (isDefault=1)
    prob_default = best_model.predict(dtest)

    # 6. 保存为 CSV
    submission = pd.DataFrame({
        'id': test_df['id'],
        'isDefault': prob_default
    })
    submission.to_csv('submission.csv', index=False)
    print("✅ 预测完成 -> submission.csv (含 id 和 违约概率)")

if __name__ == "__main__":
    main()
