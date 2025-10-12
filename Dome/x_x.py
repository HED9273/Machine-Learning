# coding: utf-8
# x**x的图像

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,10,100) # 生成100个数据点
y = x**x
# linewidth=3 线宽
plt.plot(x,y,'ro-',mec='k',linewidth=3) # 绘制图像
plt.show()