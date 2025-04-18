import pandas as pd
import numpy as np
import pickle
import xgboost as xgb


# 基于XGBoost预测概率生成评分卡
def generate_score(prob_default, base_score=600, pdo=50):
    # 计算概率对应的对数几率 (log(odds))
    odds = prob_default / (1 - prob_default)
    log_odds = np.log(odds)

    # 计算得分
    score = base_score - (pdo * log_odds)

    return score


# 主函数
def main():
    print("========== [生成评分] ==========")

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

    # 6. 根据预测概率生成评分
    scores = [generate_score(prob) for prob in prob_default]

    # 7. 保存结果为 CSV，将 id 和对应的评分保存
    submission = pd.DataFrame({
        'id': test_df['id'],
        'score': scores
    })
    submission.to_csv('scores.csv', index=False)
    print("✅ 评分已生成并保存为 scores.csv")


if __name__ == "__main__":
    main()
