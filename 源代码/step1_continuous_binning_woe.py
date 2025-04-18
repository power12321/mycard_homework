# -*- coding: utf-8 -*-
"""
step1_continuous_binning_woe.py

功能：
1. 对 train_preprocessed.csv 中的连续变量 (continuous_cols) 做最优分箱 + WOE
2. 同样对 test_preprocessed.csv 中相同列做 WOE 转换
3. 替换原连续列为 XXX_woe，保留其余列 (离散/类别/id/isDefault)
4. 输出 train_step1.csv / test_step1.csv
"""

import pandas as pd
import scorecardpy as sc
import pickle

def main():
    print("========== [STEP1] 连续变量: 分箱 + WOE ==========")

    # 1. 读取预处理后的数据
    train_df = pd.read_csv('train_preprocessed.csv')  # 含 isDefault
    test_df = pd.read_csv('test_preprocessed.csv')    # 无 isDefault

    y_col = 'isDefault'  # 标签

    # 2. 连续变量列表 (需分箱 + WOE)
    continuous_cols = [
        'loanAmnt', 'interestRate', 'installment', 'employmentTitle',
        'annualIncome', 'purpose', 'postCode', 'regionCode', 'dti',
        'delinquency_2years', 'ficoRangeLow', 'ficoRangeHigh', 'openAcc',
        'pubRec', 'pubRecBankruptcies', 'revolBal', 'revolUtil', 'totalAcc',
        'title', 'n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9',
        'n10', 'n13', 'n14', 'issueDate', 'earliesCreditLine'
    ]

    # 3. 使用 scorecardpy 对连续变量做最优分箱
    #    只在训练集上做 (有标签 isDefault)
    bins_cont = sc.woebin(
        train_df,
        y=y_col,
        x=continuous_cols,
        min_perc_fine_bin=0.01,
        min_perc_coarse_bin=0.02,
        stop_limit=0.01
    )

    # 4. 对训练集做 WOE 映射
    train_woe_cont = sc.woebin_ply(train_df, bins_cont, to='woe')

    # 5. 对测试集做 WOE 映射 (测试集无 isDefault，不影响)
    test_woe_cont = sc.woebin_ply(test_df, bins_cont, to='woe')

    # 6. 将训练集替换：把原连续字段替换为 xxx_woe
    #    其余字段 (离散/类别 + id + isDefault) 保留原状
    #    注意 sc.woebin_ply 会删除旧字段并生成 xxx_woe
    #    所以 train_woe_cont 已经只有 _woe 列 + 其余未映射列
    #    我们可以直接用 train_woe_cont，但要确保离散列还在
    #    -> sc.woebin_ply 删除了 continuous_cols 原字段，只保留了 _woe

    # 7. 保存 step1 的中间结果
    train_woe_cont.to_csv('train_step1.csv', index=False)
    test_woe_cont.to_csv('test_step1.csv', index=False)

    # 8. 保存连续变量分箱规则
    with open('bins_step1_continuous.pkl', 'wb') as f:
        pickle.dump(bins_cont, f)

    print("✅ [STEP1] 连续变量分箱+WOE完成 -> train_step1.csv / test_step1.csv")

if __name__ == "__main__":
    main()
