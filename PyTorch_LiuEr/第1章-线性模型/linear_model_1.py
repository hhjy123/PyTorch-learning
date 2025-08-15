"""
核心概念：
  训练(x, y) -> 测试(x)
  过拟合：训练集表现好但测试集表现差，泛化能力差
  泛化能力：模型对未见数据的识别能力
  解决方案：将训练集分为训练子集和验证集
  模型公式：y = x*w + b (本演示中 b=0)
  损失函数：均方误差(Mean Squared Error, MSE) loss = (x*w - y)**2
"""
import numpy as np
import matplotlib.pyplot as plt

# ===== 训练数据 =====
# 输入特征
x_data = [1.0, 2.0, 3.0]
# 对应标签
y_data = [2.0, 4.0, 6.0]

# ===== 模型定义 =====
def forward(x):
    """
    前向传播函数：计算预测值
    参数:
        x: 输入特征值
    返回:
        x*w: 模型预测值
    """
    return x * w

def loss(x, y):
    """
    损失函数：计算单个样本的均方误差
    参数:
        x: 输入特征值
        y: 真实标签值
    返回:
        (y_pred - y)**2: 单个样本的损失值
    """
    y_pred = forward(x)  # 获取预测值
    return (y_pred - y) ** 2  # 计算平方误差


# ===== 模型训练与分析 =====
# 存储不同权重下的MSE值
w_list = []  # 权重值列表
mse_list = []  # 对应MSE值列表

# 遍历权重值范围(0.0到4.0，步长0.1)
for w in np.arange(0.0, 4.1, 0.1):
    print(f'当前权重 w = {w:.1f}')
    l_sum = 0  # 当前权重下的损失总和

    # 遍历所有训练样本
    # zip()函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
    for x_val, y_val in zip(x_data, y_data):
        # 计算预测值
        y_pred_val = forward(x_val)
        # 计算损失值
        loss_val = loss(x_val, y_val)
        # 累加损失
        l_sum += loss_val
        # 打印单个样本计算结果
        print(f'\t输入: {x_val}, 输出: {y_val}, 预测: {y_pred_val:.2f}, 损失: {loss_val:.2f}')

    # 计算平均损失(MSE)
    mse = l_sum / 3
    print(f'均方误差(MSE): {mse:.4f}\n')

    # 记录当前权重和对应的MSE
    w_list.append(w)
    mse_list.append(mse)

# ===== 结果可视化 =====
plt.plot(w_list, mse_list)
plt.xlabel('W')
plt.ylabel('Loss')
plt.show()