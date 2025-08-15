# 训练数据：输入特征和对应标签
x_data = [1.0, 2.0, 3.0]  # 输入特征 (x值)
y_data = [2.0, 4.0, 6.0]  # 标签 (对应的y值)
# 真实关系：y = 2x，理想权重应为w=2.0

# 初始化权重参数
w = 1.0  # 初始权重值设为1.0

# 前向传播函数：计算预测值
def forward(x):
    """
    线性模型前向传播
    参数:
        x: 输入特征值
    返回:
        x * w: 预测值
    """
    return x * w

# 损失函数：计算单个样本的损失
def loss(x, y):
    """
    计算单个样本的平方误差损失
    参数:
        x: 输入特征值
        y: 真实标签值
    返回:
        (y_pred - y)**2: 单个样本的损失值
    """
    y_pred = forward(x)  # 计算预测值
    return (y_pred - y) ** 2  # 返回平方误差

# 梯度计算函数：计算单个样本的梯度
def gradient(x, y):
    """
    计算单个样本的梯度
    参数:
        x: 输入特征值
        y: 真实标签值
    返回:
        单个样本对权重w的梯度
    梯度推导:
        损失函数: L = (wx - y)²
        关于w的导数: ∂L/∂w = 2x(wx - y)
    """
    return 2 * x * (x * w - y)

# 训练前使用初始模型进行预测
print('Predict (before training)', 4, forward(4))
# 输出: Predict (before training) 4 4.0
# 解释: 输入x=4时，预测值 = 4 * 1.0 = 4.0

# 训练循环：100个epoch
for epoch in range(100):
    # 遍历每个样本 (随机梯度下降)
    for x, y in zip(x_data, y_data):
        # 计算当前样本的梯度
        grad = gradient(x, y)

        # 更新权重: w = w - 学习率 * 梯度
        w -= 0.01 * grad

        # 打印当前样本的梯度信息
        print('\tgrad:', x, y, grad)

        # 计算当前样本的损失
        l = loss(x, y)

        # 打印训练进度
        print('epoch:', epoch, 'w =', w, 'loss =', l)

# 训练后使用优化后的模型进行预测
print('Predict (after training)', 4, forward(4))

"""
SGD的优缺点
优点：
1.计算效率高：特别适合大数据集
2.在线学习：可处理流式数据
3.逃离局部最优：随机性有助于跳出局部最小值
4.正则化效果：噪声有助于防止过拟合
缺点：
1.收敛不稳定：损失值波动大
2.学习率敏感：需要精心调整学习率
3.可能不收敛：在最小值附近震荡
4.并行困难：顺序处理样本
"""