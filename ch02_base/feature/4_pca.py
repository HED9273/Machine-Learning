# 主成份分析

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 生成数据

X = np.random.randn(1000, 3) # 随机生成1000行3列的矩阵, randn()作用是生成一个服从正太分布的随机数
# print( X)
# print(X.shape)

# 使用PCA进行降维，将3维数据降为2维
pca = PCA(n_components=2) # n_components=2作用是设置主成分的个数
X_pca = pca.fit_transform(X) # fit_transform()作用是训练模型并返回降维后的数据
# print(X_pca.shape)

# 可视化
# 转换前的3维数据可视化
fig = plt.figure(figsize=(12,10))
# figure()作用是创建一个figure对象, figsize=(12,4)作用是设置figure的大小
# figure()对象是一个画布，画布上可以放置多个子图，子图可以放置多个子图元素
ax1 = fig.add_subplot(221, projection='3d') # 添加一个子图，并设置子图的投影为3D
ax1.scatter(X[:,0], X[:,1], X[:,2], c="g") # scatter()作用是绘制散点图
ax1.set_title('Before PCA(3D)')
ax1.set_xlabel('Feature1') # 设置X轴标签
ax1.set_ylabel('Feature2') # 设置Y轴标签
ax1.set_zlabel('Feature3') # 设置Z轴标签
# 转换后的2维数据可视化
ax2 = fig.add_subplot(222)
# 添加网格线
ax2.grid(zorder=1) # 添为 ax2.grid() 添加了 zorder=1 参数，使网格线位于底层
ax2.scatter(X_pca[:,0], X_pca[:,1], c="r",zorder=2) #为 ax2.scatter() 添加了 zorder=2 参数，使散点图位于网格线上层
ax2.set_title('After PCA(2D)')
ax2.set_xlabel('Principal Component1')
ax2.set_ylabel('Principal Component2')

# 手动构建线性相关的三组特征数据
n = 1000
# 定义两个主成分方向向量
pc1 = np.random.normal(0,1,size=n) # normal()作用是生成一个服从正太分布的随机数,参数：均值，标准差，大小
pc2 = np.random.normal(0,0.2,size=n)
# 定义不重要的第三主成分(噪声)
noise = np.random.normal(0,0.05,size=n)

# 构建三个特征的输入数据x
x = np.vstack((pc1+pc2,pc1-pc2,pc2+noise)).T # vstack()作用是将多个数组堆叠起来
print(x.shape)

# 使用PCA将3维数据降为2维
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x) # fit_transform()作用是训练模型并返回降维后的数据
print(x_pca.shape)

# 可视化
# 降维前的数据展示
# fig = plt.figure(figsize=(12,4))
x1 = fig.add_subplot(223, projection='3d')
x1.scatter(x[:,0], x[:,1], x[:,2], c="g")
x1.set_title('Before PCA(3D)')
x1.set_xlabel('Feature1')
x1.set_ylabel('Feature2')
x1.set_zlabel('Feature3')
# 降维后的数据展示
x2 = fig.add_subplot(224)
x2.grid(zorder=1)
x2.scatter(x_pca[:,0], x_pca[:,1], c="r",zorder=2)
x2.set_title('After PCA(2D)')
x2.set_xlabel('Principal Component1')
x2.set_ylabel('Principal Component2')
plt.show()