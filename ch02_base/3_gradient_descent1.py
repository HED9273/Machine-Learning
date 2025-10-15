import numpy as np
import matplotlib.pyplot as plt

# 定义目标函数
def f(x):
    return (x+3)**2-5
# 定义梯度函数 (导函数)
def gradient(x):
    return 2*(x+3)
# 用列表保存点的变化轨迹
x_list=[]
y_list=[]

# 定义超参数和x的初始值
alpha=0.1
x=1
# 重复迭代100次
for i in range(100):
    y = f(x)
    # 记录当前点的坐标
    x_list.append(x)
    y_list.append(y)
    print(f"x={x},f(x)={y}")
    # 计算梯度
    grad = gradient(x)
    # 更新参数
    x = x - alpha*grad
# 画图
x = np.arange(-1,1,0.01)
plt.plot(x,f(x))
plt.plot(x_list,y_list,"r")
plt.scatter(x_list,y_list,c="r")
plt.show()
