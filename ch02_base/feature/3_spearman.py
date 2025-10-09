# 斯皮尔曼相关系数
import pandas as pd

# 定义数据
# 每周学习时长
x = [[5],[8],[10],[12],[15],[3],[7],[9],[14],[6]]
# 数学考试成绩
y = [55,65,70,75,85,50,60,72,80,58]

x=pd.DataFrame(x) # 将列表转换为DataFrame对象,DataFrame对象可以进行行和列的索引
y=pd.Series(y) # 将列表转换为Series对象,Series对象可以进行行索引

# print(x.shape)
# print(y.shape)
# 计算斯皮尔曼相关系数
print(x.corrwith(y,method='spearman'))