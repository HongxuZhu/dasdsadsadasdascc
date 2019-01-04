import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score

df = pd.read_csv('data/jinnan_round1_train_20181227.csv', encoding='gbk')

columns = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10',
           'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20',
           'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'B1', 'B2',
           'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13',
           'B14']
'''
df_dummy = pd.get_dummies(df, columns=columns)
for col in df_dummy.columns:
    unique_col = df_dummy[col].unique()
    print(col, ' -> ', len(unique_col), ' -> ', unique_col)
    # print(col, ' -> ', len(unique_col))
'''
# print(df.columns)
df = df.replace(np.nan, 'MissVal')
reg = svm.SVR()
enc = OneHotEncoder()

# X = df[columns]
# Y = df['收率']
# enc.fit(X)
# OnehotX = enc.transform(X)
# print(OnehotX)


train = pd.read_csv('data/jinnan_round1_train_20181227.csv', encoding='gbk')
test = pd.read_csv('data/jinnan_round1_testA_20181227.csv', encoding='gbk')
target = train['收率']
del train['收率']
data = pd.concat([train, test], axis=0, ignore_index=True)
del data['样本id']
data = data.replace(np.nan, 'MissVal')

df_dummy = pd.get_dummies(data)
cate_columns = [f for f in data.columns if f != '样本id']
for f in cate_columns:
    data[f] = data[f].map(dict(zip(data[f].unique(), range(0, data[f].nunique()))))
enc.fit(data)
data = enc.transform(data)
train = data[:train.shape[0]]
test = data[train.shape[0]:]

scores = cross_val_score(reg, train, target, scoring='mse', cv=5, return_train_score=False)
print(scores)
