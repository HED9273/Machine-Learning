# 低方差过滤

import numpy as np

# 构造特征
a = np.random.randn(100) # randn()函数生成正态分布的随机数
print(np.var(a)) #var()函数计算方差

# b = np.random.randn(100)*0.1 # 构造特征b
b = np.random.normal(5,0.1,size=100) # normal()函数生成正态分布的随机数,参数：均值，标准差，大小
print(np.var(b))

#构造特征向量（输入数据x）
X = np.vstack((a,b)).T # vstack()函数将多个数组堆叠起来
print(X)
print(X.shape)
#低方差过滤
from sklearn.feature_selection import VarianceThreshold
vt=VarianceThreshold(0.01)# 0.01为方差阈值
X_filtered = vt.fit_transform(X)# fit_transform()方法, 将数据进行低方差过滤
print(X_filtered)
print(X_filtered.shape)

