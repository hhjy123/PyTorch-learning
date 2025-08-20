"""
糖尿病预测：多层神经网络二分类模型

本代码使用PyTorch实现一个多层神经网络来预测糖尿病风险，包括：
1. 糖尿病数据集加载与预处理
2. 三层全连接神经网络模型设计
3. Sigmoid激活函数应用
4. 二元交叉熵损失函数
5. 训练过程与损失可视化
"""

import numpy as np
import torch
from matplotlib import pyplot as plt

# ===== 数据加载与预处理 =====
# 加载糖尿病数据集
# 数据来源: Pima Indians Diabetes Database
# 数据集特征: 8个医学特征
# 目标变量: 是否患有糖尿病 (0=未患病, 1=患病)
xy = np.loadtxt("../数据/diabetes.csv.gz", delimiter=",", dtype=np.float32)
"""
数据集说明:
  特征(8个): 
    1. 怀孕次数 
    2. 口服葡萄糖耐量试验中2小时的血浆葡萄糖浓度
    3. 舒张压 (mm Hg)
    4. 三头肌皮褶厚度 (mm)
    5. 2小时血清胰岛素 (mu U/ml)
    6. 体重指数 (kg/m^2)
    7. 糖尿病家系功能
    8. 年龄 (岁)
  标签: 是否患有糖尿病 (0或1)
"""

# 创建特征张量 (所有行, 前8列)
x_data = torch.from_numpy(xy[:, :-1])
# 创建标签张量 (所有行, 最后一列，保持二维形式)
y_data = torch.from_numpy(xy[:, [-1]])


# ===== 神经网络模型定义 =====
class Net(torch.nn.Module):
    """
    三层全连接神经网络

    网络结构:
      输入层: 8个神经元 (对应8个特征)
      隐藏层1: 6个神经元
      隐藏层2: 4个神经元
      输出层: 1个神经元 (二分类概率)
    """

    def __init__(self):
        """
        初始化函数：定义网络层和激活函数
        """
        super(Net, self).__init__()  # 调用父类构造函数

        # 第一层: 8个输入特征 -> 6个神经元
        self.linear1 = torch.nn.Linear(8, 6)
        # 第二层: 6个神经元 -> 4个神经元
        self.linear2 = torch.nn.Linear(6, 4)
        # 第三层: 4个神经元 -> 1个输出
        self.linear3 = torch.nn.Linear(4, 1)

        # Sigmoid激活函数 (将输出转换为概率)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        """
        前向传播函数：定义数据流向
        参数:
            x: 输入特征张量
        返回:
            糖尿病患病概率 (0~1之间)
        """
        # 第一层: 线性变换 + Sigmoid激活
        x = self.sigmoid(self.linear1(x))
        # 第二层: 线性变换 + Sigmoid激活
        x = self.sigmoid(self.linear2(x))
        # 第三层: 线性变换 + Sigmoid激活
        x = self.sigmoid(self.linear3(x))
        return x


# 创建模型实例
net = Net()
"""
模型输出说明:
  输出值在0-1之间，表示患糖尿病的概率
  例如: 输出0.7表示70%的概率患有糖尿病
"""

# ===== 损失函数与优化器 =====
# 二元交叉熵损失函数
criterion = torch.nn.BCELoss(size_average=False)
"""
BCELoss (Binary Cross Entropy Loss):
  适用于二分类问题
  公式: L = -[y*log(p) + (1-y)*log(1-p)]
  参数说明:
    size_average=False: 返回所有样本的损失总和
    若为True: 返回损失的平均值
"""

# 随机梯度下降优化器
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
"""
参数说明:
  net.parameters(): 获取模型所有可训练参数
  lr=0.1: 学习率(相对较大的学习率，因为数据集较小)
  注意: 较大的学习率可能导致训练不稳定
"""

# ===== 训练循环 =====
# 用于记录训练过程的列表
epoch_list = []  # 记录训练轮次
loss_list = []  # 记录每个轮次的损失值

for epoch in range(100):  # 100个训练周期
    # 前向传播: 计算预测概率
    y_pred = net(x_data)

    # 计算损失: 比较预测概率和真实标签
    loss = criterion(y_pred, y_data)

    # 打印当前损失
    print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')

    # 记录当前轮次和损失
    epoch_list.append(epoch)
    loss_list.append(loss.item())

    # 梯度清零
    optimizer.zero_grad()

    # 反向传播: 计算梯度
    loss.backward()

    # 参数更新
    optimizer.step()

# ===== 训练过程可视化 =====
plt.figure(figsize=(10, 6))
# 绘制损失曲线
plt.plot(epoch_list, loss_list, 'b-')
# 添加标签和标题
plt.title('Training Loss Curve', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
# 添加网格线
plt.grid(True, linestyle='--', alpha=0.7)
# 显示图表
plt.show()

"""
糖尿病预测模型详解:
=========================================================
1. 数据集特性:
   - 样本数: 约768个
   - 特征数: 8个医学指标
   - 类别不平衡: 健康样本多于患病样本
   - 数据来源: Pima Indians Diabetes Database

2. 网络结构分析:
   - 输入层: 8个神经元 (对应8个特征)
   - 隐藏层1: 6个神经元 (降维)
   - 隐藏层2: 4个神经元 (进一步提取特征)
   - 输出层: 1个神经元 (概率输出)
   - 激活函数: Sigmoid (每层后应用)

3. 激活函数选择:
   - 隐藏层和输出层均使用Sigmoid
   - 优点: 输出范围(0,1)，适合概率预测
   - 缺点: 深层网络可能存在梯度消失问题

4. 损失函数:
   - BCELoss: 专门用于二分类问题
   - 使用总损失而非平均损失(size_average=False)

5. 优化器设置:
   - SGD: 随机梯度下降
   - 学习率0.1: 相对较大，适合小数据集
   - 风险: 可能导致训练不稳定或震荡

6. 训练过程分析:
   - 损失曲线应呈现下降趋势
   - 理想情况: 平滑收敛到较低损失值
   - 可能出现问题: 
        * 损失震荡(学习率过大)
        * 损失不下降(网络结构或学习率问题)

7. 模型评估建议:
   - 需要划分训练集和测试集(当前代码未划分)
   - 评估指标: 准确率、精确率、召回率、F1分数、AUC-ROC
   - 处理类别不平衡: 使用加权损失或过采样/欠采样

8. 改进方向:
   A. 数据预处理:
        - 特征标准化/归一化
        - 处理缺失值(如果存在)
        - 处理类别不平衡

   B. 网络结构:
        - 增加批量归一化层
        - 使用ReLU激活函数(隐藏层)
        - 添加Dropout层防止过拟合

   C. 训练策略:
        - 划分验证集进行早停
        - 使用学习率调度器
        - 尝试不同优化器(如Adam)

   D. 评估方法:
        - 使用k折交叉验证
        - 计算混淆矩阵和分类报告
=========================================================
"""