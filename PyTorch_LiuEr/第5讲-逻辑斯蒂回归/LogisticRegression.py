"""
PyTorch逻辑回归完整实现：二分类问题解决方案

本代码展示了使用PyTorch实现逻辑回归模型来解决二分类问题，包括：
1. 二分类数据准备
2. 逻辑回归模型定义
3. Sigmoid激活函数应用
4. 二元交叉熵损失函数
5. 训练循环与优化
6. 决策边界可视化
"""

import torch
from torch import nn
import torch.nn.functional as F  # 包含常用激活函数和损失函数
import numpy as np
import matplotlib.pyplot as plt

# ===== 数据准备 =====
# 创建特征张量 (3x1矩阵): 学习时间(小时)
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
# 创建标签张量 (3x1矩阵): 考试结果(0=未通过, 1=通过)
y_data = torch.Tensor([[0], [0], [1]])
"""
数据说明:
  学习1小时 → 未通过(0)
  学习2小时 → 未通过(0)
  学习3小时 → 通过(1)
问题类型: 二分类问题(通过/未通过)
"""

# ===== 模型定义 =====
class LogisticRegression(nn.Module):
    """
    逻辑回归模型

    继承自nn.Module，包含：
    - 一个线性层 (nn.Linear)
    - Sigmoid激活函数 (将输出转换为概率)
    """
    def __init__(self):
        """
        初始化函数：定义网络层
        """
        super(LogisticRegression, self).__init__()  # 调用父类构造函数
        # 创建全连接层: 输入特征数=1, 输出特征数=1
        self.linear = nn.Linear(1, 1)
        """
        参数说明:
          in_features=1: 输入维度(学习时间)
          out_features=1: 输出维度(通过概率)
        自动初始化:
          weight: 权重参数
          bias: 偏置参数
        """

    def forward(self, x):
        """
        前向传播函数：定义计算流程
        参数:
            x: 输入数据张量
        返回:
            y_pred: 通过概率 (0~1之间)
        """
        # 步骤1: 线性变换 z = w*x + b
        linear_output = self.linear(x)

        # 步骤2: 应用Sigmoid激活函数 σ(z) = 1/(1+e^{-z})
        y_pred = F.sigmoid(linear_output)  # 将输出转换为概率值

        return y_pred

# 创建模型实例
module = LogisticRegression()
"""
模型输出说明:
  输出值在0-1之间，表示通过考试的概率
  例如: 输出0.7表示70%的通过概率
"""

# ===== 损失函数 =====
# 创建二元交叉熵损失函数
criterion = nn.BCELoss(size_average=False)
"""
二元交叉熵损失(BCELoss):
  公式: L = -[y*log(p) + (1-y)*log(1-p)]
  其中:
    y: 真实标签(0或1)
    p: 预测概率(0~1)
  
参数说明:
  size_average=False: 返回所有样本的损失总和
  若为True: 返回损失的平均值
"""

# ===== 优化器 =====
# 创建随机梯度下降优化器
optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
"""
参数说明:
  module.parameters(): 获取模型所有可训练参数
  lr=0.01: 学习率(控制更新步长)
"""

# ===== 训练循环 =====
for epoch in range(1000):  # 1000个训练周期
    # 前向传播: 计算预测概率
    y_pred = module(x_data)

    # 计算损失: 比较预测概率和真实标签
    loss = criterion(y_pred, y_data)

    # 打印训练进度(每100个epoch打印一次)
    if epoch % 100 == 0:
        print(f'epoch: {epoch}, loss: {loss.item():.4f}')

    # 梯度清零: 清空上一次计算的梯度值
    optimizer.zero_grad()

    # 反向传播: 计算损失关于参数的梯度
    loss.backward()

    # 参数更新: 使用优化器更新权重和偏置
    optimizer.step()

# ===== 决策边界可视化 =====
# 生成测试数据: 0到10小时的200个点
x = np.linspace(0, 10, 200)
# 转换为PyTorch张量 (200x1矩阵)
x_t = torch.Tensor(x).view(200, 1)
# 使用训练好的模型预测通过概率
y_t = module(x_t)
# 转换为NumPy数组
y = y_t.data.numpy()

# 创建绘图
plt.figure(figsize=(10, 6))
# 绘制预测概率曲线
plt.plot(x, y, 'b-', label='Predicted Probability')
# 绘制决策边界线 (概率=0.5)
plt.plot([0, 10], [0.5, 0.5], 'r--', label='Decision Boundary (p=0.5)')
# 添加原始数据点
plt.scatter([1, 2], [0, 0], c='red', s=100, label='Fail (0 hour)')
plt.scatter([3], [1], c='green', s=100, label='Pass (3 hours)')

# 添加标签和标题
plt.xlabel('Study Hours', fontsize=12)
plt.ylabel('Probability of Passing', fontsize=12)
plt.title('Logistic Regression Decision Boundary', fontsize=14)
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.7)

# 显示图表
plt.show()

"""
逻辑回归核心概念详解:
=========================================================
1. 逻辑回归模型:
   - 线性部分: z = w*x + b
   - Sigmoid函数: p = 1/(1+e^{-z})
   - 输出解释: 样本属于正类(通过)的概率

2. Sigmoid函数特性:
   - 将任意实数映射到(0,1)区间
   - 当z=0时，p=0.5
   - 函数呈S形曲线

3. 决策边界:
   - 当p >= 0.5时，预测为正类(通过)
   - 当p < 0.5时，预测为负类(未通过)
   - 决策边界对应p=0.5的位置

4. 二元交叉熵损失:
   - 专门为二分类问题设计
   - 当预测概率接近真实标签时，损失值小
   - 当预测概率远离真实标签时，损失值大

5. 训练过程分析:
   - 初始损失较高(随机参数)
   - 随着训练进行，损失逐渐降低
   - 最终模型能区分不同学习时间的结果

6. 可视化解读:
   - 蓝色曲线: 不同学习时间对应的通过概率
   - 红色虚线: 决策边界(p=0.5)
   - 红点: 实际未通过的样本
   - 绿点: 实际通过的样本
   - 曲线与决策边界的交点: 预测的转折点

7. 模型参数解读:
   - 权重w: 控制曲线的陡峭程度
   - 偏置b: 控制决策边界的位置

8. 决策边界计算:
   当p=0.5时:
     0.5 = 1/(1+e^{-(w*x+b)})
     => 1 = 1 + e^{-(w*x+b)}
     => w*x + b = 0
     决策边界: x = -b/w
=========================================================
"""