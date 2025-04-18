# -*- coding: utf-8 -*-
"""
data_preprocessing.py

功能：
1. 读取 train.csv 和 testA.csv
2. 删除字段 policyCode,'n11','n12'
3. 将 issueDate 和 earliesCreditLine 转换为距最早时间的月数（覆盖原字段）
4. employmentLength 映射为数字（<1年→0，1年→1，10+年→10），覆盖原字段
5. 进行缺失值填充与异常值截断
6. 输出 train_preprocessed.csv 和 test_preprocessed.csv
"""

import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

def transform_fields(df, reference_dates):
    def clean_employment_length(x):
        if pd.isna(x):
            return np.nan
        if '<' in x:
            return 0
        if '10+' in x:
            return 10
        try:
            return int(str(x).strip().split()[0])
        except:
            return np.nan

    if 'employmentLength' in df.columns:
        df['employmentLength'] = df['employmentLength'].apply(clean_employment_length)

    if 'issueDate' in df.columns:
        df['issueDate'] = pd.to_datetime(df['issueDate'], errors='coerce')
        min_issue_date = reference_dates['min_issueDate']
        df['issueDate'] = df['issueDate'].apply(
            lambda x: (x.year - min_issue_date.year) * 12 + (x.month - min_issue_date.month) if pd.notna(x) else np.nan
        )

    def convert_credit_line(x):
        try:
            dt = pd.to_datetime(x, format='%b-%Y')
            min_dt = reference_dates['min_earliesCreditLine']
            return (dt.year - min_dt.year) * 12 + (dt.month - min_dt.month)
        except:
            return np.nan

    if 'earliesCreditLine' in df.columns:
        df['earliesCreditLine'] = df['earliesCreditLine'].apply(convert_credit_line)

    return df

def handle_missing_outliers(df):
    continuous_cols = ['loanAmnt', 'interestRate', 'installment', 'employmentTitle', 'annualIncome', 'purpose' ,'postCode',
                   'regionCode', 'dti', 'delinquency_2years', 'ficoRangeLow', 'ficoRangeHigh', 'openAcc', 'pubRec',
                   'pubRecBankruptcies' ,'revolBal', 'revolUtil', 'totalAcc', 'title', 'n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7',
                   'n8', 'n9', 'n10', 'n13', 'n14',
                   'issueDate', 'earliesCreditLine']

    discrete_cols = ['term', 'homeOwnership', 'verificationStatus', 'initialListStatus',
                    'employmentLength']

    category_cols = ['grade', 'subGrade']
    undo_cols = [ 'applicationType']
    # 对连续型变量进行缺失值填充：使用中位数填充
    for col in continuous_cols:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)
    #对离散型变量进行缺失值填充：使用众数填充
    for col in discrete_cols:
        if col in df.columns and df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)
    # 对类别型变量进行缺失值填充：使用字符串 'missing' 填充
    for col in category_cols:
        if col in df.columns and df[col].isnull().any():
            df[col].fillna('missing', inplace=True)
    # 对所有数值型列进行异常值处理：将数据裁剪到 [1%, 99%] 分位数范围内
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[col] = np.clip(df[col], lower, upper)

    return df

def main():
    print("========== [Data Preprocessing] ==========")
    train_df = pd.read_csv('train.csv', dtype={'id': str})
    test_df = pd.read_csv('testA.csv', dtype={'id': str})

    # ✅ 删除不需要的字段
    drop_cols = ['policyCode','n11','n12']
    train_df.drop(columns=[col for col in drop_cols if col in train_df.columns], inplace=True)
    test_df.drop(columns=[col for col in drop_cols if col in test_df.columns], inplace=True)

    # 时间字段基准
    reference_dates = {
        'min_issueDate': pd.to_datetime(train_df['issueDate'], errors='coerce').min(),
        'min_earliesCreditLine': pd.to_datetime(train_df['earliesCreditLine'], format='%b-%Y', errors='coerce').min()
    }

    train_df = transform_fields(train_df, reference_dates)
    test_df = transform_fields(test_df, reference_dates)

    train_df = handle_missing_outliers(train_df)
    test_df = handle_missing_outliers(test_df)

    train_df['id'] = train_df['id'].astype(str)
    test_df['id'] = test_df['id'].astype(str)

    train_df.to_csv('train_preprocessed.csv', index=False)
    test_df.to_csv('test_preprocessed.csv', index=False)

    print("✅ 数据预处理完成 -> train_preprocessed.csv / test_preprocessed.csv")

if __name__ == "__main__":
    main()
