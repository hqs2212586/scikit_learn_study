
import numpy as np
import os
import warnings
import matplotlib
import matplotlib.pyplot as plt

# 画图参数设置
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# 过滤警告
warnings.filterwarnings('ignore')

"""构造样本"""
# 通过rand函数可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1
X = 2 * np.random.rand(100, 1)
# 构造线性方程，加入随机抖动
# numpy.random.randn()是从标准正态分布中返回一个或多个样本值
# 1.当函数括号内没有参数时，返回一个浮点数；
# 2.当函数括号内有一个参数时，返回秩为1的数组，不能表示向量和矩阵
# 3.当函数括号内有两个及以上参数时，返回对应维度的数组，能表示向量或矩阵。np.random.randn(行,列)
# 4.np.random.standard_normal()函数与np.random.randn类似，但是输入参数为元组（tuple）
y = 3*X + 4 + np.random.randn(100, 1)

plt.plot(X, y, 'b.')     # b指定为蓝色,.指定线条格式
plt.xlabel('X_1')
plt.ylabel('y')

# 设置x轴为0-2，y轴为0-15
plt.axis([0, 2, 0, 15])
plt.show()

"""线性回归方程实现"""
# numpy.c_:按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等。
# numpy.r_:按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等。
# ones()返回一个全1的n维数组，同样也有三个参数：shape（用来指定返回数组的大小）、dtype（数组元素的类型）、order（是否以内存中的C或Fortran连续（行或列）顺序存储多维数据）。后两个参数都是可选的，一般只需设定第一个参数。
X_b = np.c_[(np.ones((100, 1)), X)]
# np.linalg.inv:矩阵求逆
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print(theta_best)
'''
[[3.99844262]
 [3.09461187]]
'''

# 测试数据
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
# 预测结果
y_predict = X_new_b.dot(theta_best)
print(y_predict)

plt.plot(X_new, y_predict, 'r--')   # 指定红色和线条
plt.plot(X, y, 'b.')     # 指定蓝色和点
plt.axis([0, 2, 0, 15])
plt.show()

# sklearn线性回归实现
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()    # 线性回归实例化
lin_reg.fit(X, y)     # 拟合线性模型
print(lin_reg.intercept_)   # intercept_:线性模型中的独立项
print(lin_reg.coef_)        # coef_:线性回归的估计系数
"""
[3.92151171]
[[2.98627461]]
"""



