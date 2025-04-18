# -*- coding: utf-8 -*-
"""
step2_discrete_category_woe.py

功能：
1. 读取 step1 中间结果
2. 对离散/类别变量只做 WOE (不分箱)
3. 替换原离散/类别列为 xxx_woe
4. 输出最终 train_woe.csv / test_woe.csv
   - 训练集包含所有 _woe 列 + id + isDefault
   - 测试集包含所有 _woe 列 + id
"""

import pandas as pd
import scorecardpy as sc
import pickle


def main():
    print("========== [STEP2] 离散/类别变量: 只做 WOE ==========")

    # 1. 读取 step1 结果
    train_step1 = pd.read_csv('train_step1.csv')  # 有 isDefault
    test_step1 = pd.read_csv('test_step1.csv')  # 无 isDefault

    y_col = 'isDefault'

    # 2. 定义离散+类别变量 (剩余没做分箱的)
    discrete_cols = [
        'term', 'homeOwnership', 'verificationStatus', 'initialListStatus',
        'employmentLength'
    ]
    category_cols = ['grade', 'subGrade']

    undo_cols = ['applicationType']
    dc_cols = discrete_cols + category_cols

    # 在 step1 中，这些字段还保持原值
    # 训练集: 有 isDefault, 测试集: 无

    # 3. 从 train_step1 中取出待编码字段 (dc_cols + isDefault)
    #    注意: 测试集没 isDefault, sc.woebin 只能在训练集做
    train_dc = train_step1[dc_cols + [y_col]]

    # 4. 对离散/类别变量做 WOE (bin_num_limit足够大即可避免多次合并)
    bins_dc_cat = sc.woebin(
        train_dc,
        y=y_col,
        x=dc_cols,
        bin_num_limit=100,  # 避免过度合并
        stop_limit=0.01
    )

    # 5. 应用 WOE 到训练集 & 测试集
    #    测试集没有 isDefault，不会影响
    train_dc_woe = sc.woebin_ply(train_dc, bins_dc_cat, to='woe')
    test_dc_woe = sc.woebin_ply(test_step1[dc_cols], bins_dc_cat, to='woe')

    # 6. 在 train_step1 / test_step1 中替换掉原字段
    train_final = train_step1.copy()
    test_final = test_step1.copy()

    # 删掉原来的离散/类别列
    train_final.drop(columns=dc_cols, inplace=True)
    test_final.drop(columns=dc_cols, inplace=True)

    # 新增 WOE 列
    for col in dc_cols:
        train_final[col + '_woe'] = train_dc_woe[col + '_woe']
        test_final[col + '_woe'] = test_dc_woe[col + '_woe']

    # 7. 最终只保留 [所有 _woe] + [id] + [isDefault(仅训练)]
    # 获取所有 _woe 列
    woe_cols = [c for c in train_final.columns if c.endswith('_woe')]

    # 训练集保留 id, isDefault
    train_final = train_final[woe_cols + ['id', 'isDefault'] + undo_cols]
    # 测试集保留 id
    test_final = test_final[woe_cols + ['id'] + undo_cols]

    # 8. 输出最终结果
    train_final.to_csv('train_woe.csv', index=False)
    test_final.to_csv('test_woe.csv', index=False)

    # 9. 保存 bins
    with open('bins_step2_discrete_category.pkl', 'wb') as f:
        pickle.dump(bins_dc_cat, f)

    print("✅ [STEP2] 离散/类别WOE完成 -> train_woe.csv / test_woe.csv")


if __name__ == "__main__":
    main()
