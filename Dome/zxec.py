# 利用sklearn实现线性回归进行房价预测
from sklearn.datasets import fetch_openml
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# 波士顿房价数据集已移至OpenML平台
boston = fetch_openml(name="boston", version=1)
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['MEDV'] = boston.target  # 添加房价目标列

print(f"波士顿房价数据集形状: {df.shape}")
print(f"特征名称: {boston.feature_names}")
print(f"目标变量: MEDV (房屋中位数价格，单位: 1000美元)")
# 将数据集存入excel文件
# df.to_excel('波士顿房价数据集.xlsx')

# 准备数据
X = df[['LSTAT']]  # 选择一个特征进行简单线性回归（例如低收入人群比例）
y = df['MEDV']     # 目标变量

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建并训练线性回归模型（基于最小二乘法）
model = LinearRegression()
model.fit(X_train, y_train) # 训练模型

# 预测
y_pred = model.predict(X_test) # 预测

# 计算模型性能指标
mse = mean_squared_error(y_test, y_pred) #  均方误差
r2 = r2_score(y_test, y_pred) # 决定系数

print(f"线性回归系数: {model.coef_[0]:.4f}")
print(f"截距: {model.intercept_:.4f}")
print(f"均方误差(MSE): {mse:.4f}")
print(f"决定系数(R²): {r2:.4f}")

# 创建可视化图表
# plt.figure(figsize=(12, 8))
#
# # 绘制训练数据点
# plt.scatter(X_train, y_train, alpha=0.5, color='blue', label='训练数据')
#
# # 绘制测试数据点
# plt.scatter(X_test, y_test, alpha=0.5, color='red', label='测试数据')

# # 绘制回归线
# X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
# y_line = model.predict(X_line)
# plt.plot(X_line, y_line, color='green', linewidth=2, label=f'回归线: y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}')
#
# plt.xlabel('LSTAT (低收入人群比例 %)')
# plt.ylabel('MEDV (房价中位数, 千美元)')
# plt.title(f'波士顿房价预测 - 线性回归(最小二乘法)\nR² = {r2:.4f}')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()

# 残差分析图
# plt.figure(figsize=(12, 5))

# # 残差 vs 预测值
# plt.subplot(1, 2, 1)
# residuals = y_test - y_pred
# plt.scatter(y_pred, residuals, alpha=0.5)
# plt.axhline(y=0, color='red', linestyle='--')
# plt.xlabel('预测值')
# plt.ylabel('残差')
# plt.title('残差分析')
# plt.grid(True, alpha=0.3)
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 预测值 vs 实际值
# plt.subplot(1, 2, 2) #
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('预测值 vs 实际值')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

