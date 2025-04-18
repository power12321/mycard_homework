import pandas as pd

# 加载 CSV 文件
file_path = 'train.csv'  # 在Jupyter中路径应为相对路径或完整路径，确保文件已上传
data = pd.read_csv(file_path)

# 数据类型分析
print("数据类型分析：")
print(data.dtypes)

# 分析每一列的唯一值
print("\n每一列的唯一值分布：")
for col in data.columns:
    print(f"\n{col}的唯一值分布：")
    print(data[col].value_counts())

# 描述性统计
print("\n描述性统计：")
print(data.describe(include='all'))

# 检查缺失值
print("\n缺失值分析：")
print(data.isnull().sum())
