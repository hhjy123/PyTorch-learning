"""
线性回归模型实现：梯度下降算法详解

本代码展示了使用梯度下降算法优化线性回归模型的全过程，重点演示了：
1. 梯度下降的核心原理：w = w - η(∂cost/∂w)
2. 学习率(η)的作用和影响
3. 梯度下降的局限性：局部最优和鞍点问题
"""

from matplotlib import pyplot as plt

# ===== 训练数据 =====
# 输入特征值 (x)
x_data = [1.0, 2.0, 3.0]
# 对应标签值 (y)
y_data = [2.0, 4.0, 6.0]  # 数据关系为 y = 2x，理想权重应为2.0

# ===== 模型参数 =====
w = 1.0  # 初始化权重值（从1.0开始）

# ===== 模型函数 =====
def forward(x):
    """
    前向传播函数：计算预测值
    参数:
        x: 输入特征
    返回:
        x * w: 模型预测值
    """
    return x * w


def cost(xs, ys):
    """
    计算均方误差损失函数
    参数:
        xs: 输入特征列表
        ys: 对应标签列表
    返回:
        平均损失值 (MSE)
    """
    total_cost = 0
    # 遍历所有数据点
    for x, y in zip(xs, ys):
        y_pred = forward(x)  # 计算预测值
        # 累加平方误差
        total_cost += (y_pred - y) ** 2
    # 返回平均损失
    return total_cost / len(xs)


def gradient(xs, ys):
    """
    计算损失函数关于权重w的梯度
    参数:
        xs: 输入特征列表
        ys: 对应标签列表
    返回:
        平均梯度值
    """
    total_grad = 0
    # 遍历所有数据点
    for x, y in zip(xs, ys):
        """
        梯度推导：
        损失函数: L = (wx - y)²
        关于w的导数: ∂L/∂w = 2x(wx - y)
        """
        # 计算并累加每个样本的梯度贡献
        total_grad += 2 * x * (x * w - y)
    # 返回平均梯度
    return total_grad / len(xs)

# ===== 训练过程 =====
# 用于绘制损失曲线的列表
epoch_list = []  # 记录训练轮次
cost_list = []  # 记录每个轮次的损失值

# 训练前预测
print('Predict (before training)', 4, forward(4))  # 输出: 4 * 1.0 = 4.0

# 训练循环 (100轮)
for epoch in range(100):
    # 计算当前损失值
    cost_val = cost(x_data, y_data)
    # 计算当前梯度
    grad_val = gradient(x_data, y_data)

    # 梯度下降更新权重: w = w - η * ∂cost/∂w
    # η = 0.01 (学习率)
    w -= 0.01 * grad_val

    # 打印训练进度
    print(f'Epoch: {epoch}, w = {w:.6f}, loss = {cost_val:.6f}, grad = {grad_val:.6f}')

    # 记录当前轮次和损失值
    epoch_list.append(epoch)
    cost_list.append(cost_val)

# 训练后预测
print('Predict (after training)', 4, forward(4))  # 理想值应为8.0

# ===== 可视化训练过程 =====
plt.plot(epoch_list,cost_list)
plt.ylabel('cost')
plt.xlabel('epoch')
plt.show()

"""
梯度下降算法详解:
=========================================================
1. 核心更新公式:
    w = w - η * (∂cost/∂w)

    - w: 待优化的权重参数
    - η: 学习率(learning rate)，控制更新步长
    - ∂cost/∂w: 损失函数关于w的梯度(导数)

2. 学习率(η)的重要性:
    - η太小: 收敛速度慢，需要更多轮次
    - η太大: 可能跳过最优解，甚至发散
    - 本例中: η = 0.01 (适中值)

3. 梯度下降的局限性:
    A. 局部最优问题:
        - 只能找到当前位置附近的极小值点
        - 无法保证找到全局最优解
        - 本例中: 损失函数是凸函数，只有全局最优

    B. 鞍点问题:
        - 梯度为零的非最优点
        - 在鞍点处无法进行有效迭代
        - 本例中: 没有鞍点问题

    C. 其他问题:
        - 可能收敛到平坦区域
        - 在高维空间中容易遇到病态条件问题

4. 梯度计算:
    数学推导:
        cost = 1/N * Σ(wx_i - y_i)²
        ∂cost/∂w = 2/N * Σx_i(wx_i - y_i)

    代码实现:
        grad = (2/N) * Σ[x_i * (w*x_i - y_i)]

5. 损失函数:
    - 均方误差(Mean Squared Error, MSE)
    - 公式: cost = 1/N * Σ(y_pred_i - y_true_i)²
    - 特性: 连续可导，适合梯度下降

6. 训练过程分析:
    - 初始状态: w=1.0, loss≈4.67
    - 训练结束: w≈2.0, loss≈0
    - 损失曲线: 单调递减，平滑收敛

7. 实际应用建议:
    - 学习率调整: 可尝试动态学习率(如指数衰减)
    - 批量大小: 本例使用全批量，大数据集可用小批量
    - 早停机制: 当loss不再下降时停止训练
    - 梯度裁剪: 防止梯度爆炸
=========================================================
"""