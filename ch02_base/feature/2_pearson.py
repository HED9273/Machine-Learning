# 皮尔逊相关系数
import pandas as pd

advertising = pd.read_csv('../../data/advertising.csv')
# TV,Radio,Newspaper 不同的广告投放指标 Sales 最终得到的广告的收益
# print(advertising.head()) # 查看数据 head()作用是显示数据集的前几行
# print(advertising.describe()) # 描述数据集, describe()作用是显示数据集的统计信息
# print(advertising.shape) # 查看数据集的行数和列数 shape()作用是显示数据集的行数和列数
# print(advertising.corr()) # 计算相关系数 , corr()作用是计算数据集的相关系数
#数据预处理
#去掉第一列ID
advertising.drop(advertising.columns[0], axis=1, inplace=True)
# .columns[0] 表示第一列
# drop()作用是删除数据集的列或行, inplace=True作用是删除数据集的列或行,axis=1作用是删除数据集的列,axis=0作用是删除数据集的行

# 去掉空值
advertising.dropna(inplace=True)# dropna()作用是删除数据集的空值
# 提取特征和标签(目标值)
X = advertising.drop('Sales', axis=1) # 删除数据集的列
y = advertising['Sales']

# print(X.shape)
# print(y.shape)

# 计算皮尔逊相关系数
# print(X.corrwith(y, method='pearson')) # pearson()作用是计算皮尔逊相关系数,corrwith()作用是计算两个变量之间的皮尔逊相关系数
corr_matrix = advertising.corr(method='pearson')
print(corr_matrix) # corr()作用是计算皮尔逊相关系数
# corr()和corrwith()的区别是
# corr()作用是计算皮尔逊相关系数,corrwith()作用是计算两个变量之间的皮尔逊相关系数

# 将相关系数矩阵画出热力图
import seaborn as sns # seaborn 是一个专门用来做统计数据可视化的库
import matplotlib.pyplot as plt

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
# heatmap()作用是绘制热力图, annot=True作用是显示数据,
# cmap='coolwarm'作用是设置颜色,fmt='.2f'作用是设置数据格式,.2f表示保留两位小数
plt.title('Feature Correlation Matrix') # 标题中文意思是，特征相关矩阵
plt.show()
