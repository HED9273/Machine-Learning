import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression # 线性回归模型
from sklearn.preprocessing import PolynomialFeatures # 构建多项式特征
from sklearn.model_selection import train_test_split #划分训练集和测试集
from sklearn.metrics import mean_squared_error # 均方误差损失函数

"""
1.生成数据或者获取数据
2.划分训练集和测试集(验证集)
3.定义模型（线性回归模型）
4.训练模型
5.预测结果，计算误差（损失）
"""
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 1.生成数据
X = np.linspace(-3, 3, 300).reshape(-1, 1)
# linspace() 函数用于生成等差数列, 参数为起始值，结束值，数量
# reshape(-1, 1) 将数组 reshape 成一个 2D 数组
y = np.sin(X)+np.random.uniform(low=-0.5, high=0.5, size=300).reshape(-1, 1)
# +np.random.uniform(low=-0.5, high=0.5, size=300) 添加噪声
# .uniform() 函数用于生成一个指定范围内的数, 参数为起始值，结束值，数量

# print(X.shape)
# print(y.shape)

# 画出散点图(3个子图)
fig,ax = plt.subplots(1,3,figsize=(15,4))
ax[0].scatter(X,y,color='y')
ax[1].scatter(X,y,color='y')
ax[2].scatter(X,y,color='y')
# plt.show()

# 2.划分训练集和测试集(验证集)
trainX,testX,trainY,testY=train_test_split(X,y,test_size=0.2,random_state=42) # test_size=0.2 测试集占比20%

#   3.定义模型（线性回归模型）
model = LinearRegression() # 创建线性回归模型

# 一，欠拟合（直线）
x_train1 = trainX
x_test1 = testX

# 4.训练模型
model.fit(x_train1, trainY)

# 打印查看模型参数
print(model.coef_) # 模型参数, 斜率
print(model.intercept_) # 模型截距, 截距

#5.预测结果，计算误差（损失）
y_pred1 = model.predict(x_test1) #  预测结果
test_loss1 = mean_squared_error(testY, y_pred1) # 计算测试集均方误差损失函数
train_loss1 = mean_squared_error(trainY, model.predict(x_train1)) # 计算训练集均方误差损失函数


# 画出拟合曲线，并写出训练误差和测试误差
ax[0].plot(X,model.predict(X),color='r') # 画出拟合曲线，参数X为自变量，model.predict(X)为因变量
ax[0].text(-3,1,f'测试误差：{test_loss1:.4f}')
ax[0].text(-3,1.3,f'训练误差：{train_loss1:.4f}')
# plt.show()

# 二，恰好拟合（5次多项式）
poly5 = PolynomialFeatures(degree=5) # 创建5次多项式特征
x_train2 = poly5.fit_transform(trainX)
x_test2 = poly5.transform(testX)
print(x_train2.shape)
print(x_test2.shape)
# 训练模型
model.fit(x_train2, trainY)
# 预测结果，计算误差（损失）
y_pred2 = model.predict(x_test2)
test_loss2 = mean_squared_error(testY, y_pred2)
train_loss2 = mean_squared_error(trainY, model.predict(x_train2))

# 画出拟合曲线，并写出训练误差和测试误差
ax[1].plot(X,model.predict(poly5.fit_transform(X)),color='r')
ax[1].text(-3,1,f'测试误差：{test_loss2:.4f}')
ax[1].text(-3,1.3,f'训练误差：{train_loss2:.4f}')
# plt.show()

#三，过拟合（20次多项式）
poly20 = PolynomialFeatures(degree=20)
x_train3 = poly20.fit_transform(trainX)
x_test3 = poly20.transform(testX)
print(x_train3.shape)
print(x_test3.shape)
# 训练模型
model.fit(x_train3, trainY)
# 预测结果，计算误差（损失）
y_pred3 = model.predict(x_test3)
test_loss3 = mean_squared_error(testY, y_pred3)
train_loss3 = mean_squared_error(trainY, model.predict(x_train3))

# 画出拟合曲线，并写出训练误差和测试误差
ax[2].plot(X,model.predict(poly20.fit_transform(X)),color='r')
ax[2].text(-3,1,f'测试误差：{test_loss3:.4f}')
ax[2].text(-3,1.3,f'训练误差：{train_loss3:.4f}')
plt.show()