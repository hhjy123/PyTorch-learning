import torch
from torch import nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

# 设置批处理大小
batch_size = 64

# 定义数据预处理流程：
# 1. 将图像转换为张量
# 2. 对图像进行标准化处理（使用MNIST数据集的均值和标准差）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载MNIST训练数据集
train_dataset = datasets.MNIST(root='../数据/MNIST_DATA/',  # 数据集存储路径
                               train=True,  # 使用训练集
                               download=True,  # 如果不存在则自动下载
                               transform=transform)  # 应用定义的数据变换

# 创建训练数据加载器
train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,  # 每批加载64个样本
                          shuffle=True)  # 打乱数据顺序

# 加载MNIST测试数据集
test_dataset = datasets.MNIST(root='../数据/MNIST_DATA/',
                              train=False,  # 使用测试集
                              download=True,  # 如果不存在则自动下载
                              transform=transform)  # 应用相同的数据变换

# 创建测试数据加载器
test_loader = DataLoader(test_dataset,
                         batch_size=batch_size,
                         shuffle=False)  # 测试集不需要打乱顺序


# 定义全连接神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义5个全连接层
        self.l1 = nn.Linear(784, 512)  # 输入层（28x28=784像素）-> 隐藏层1（512神经元）
        self.l2 = nn.Linear(512, 256)  # 隐藏层1 -> 隐藏层2（256神经元）
        self.l3 = nn.Linear(256, 128)  # 隐藏层2 -> 隐藏层3（128神经元）
        self.l4 = nn.Linear(128, 64)  # 隐藏层3 -> 隐藏层4（64神经元）
        self.l5 = nn.Linear(64, 10)  # 隐藏层4 -> 输出层（10个数字类别）

    def forward(self, x):
        x = x.view(-1, 784)  # 将图像展平为784维向量（保持批量维度）
        x = F.relu(self.l1(x))  # 通过第一层并使用ReLU激活
        x = F.relu(self.l2(x))  # 通过第二层并使用ReLU激活
        x = F.relu(self.l3(x))  # 通过第三层并使用ReLU激活
        x = F.relu(self.l4(x))  # 通过第四层并使用ReLU激活
        return self.l5(x)  # 输出层不使用激活函数（CrossEntropyLoss包含Softmax）


# 实例化神经网络
net = Net()

# 定义损失函数（交叉熵损失，适用于多分类问题）
criterion = nn.CrossEntropyLoss()

# 定义优化器（带动量的随机梯度下降）
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)


# 训练函数
def train(epoch):
    running_loss = 0.0  # 累计损失值
    # 遍历训练数据加载器中的批次
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, labels = data  # 解包数据（输入图像和对应标签）

        optimizer.zero_grad()  # 清零梯度缓存

        outputs = net(inputs)  # 前向传播（获取预测值）
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播（计算梯度）
        optimizer.step()  # 更新权重

        running_loss += loss.item()  # 累加损失值

        # 每300个批次打印一次训练状态
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0  # 重置累计损失


# 测试函数
def test():
    correct = 0  # 正确预测的样本数
    total = 0  # 总样本数
    with torch.no_grad():  # 禁用梯度计算（节省内存）
        for data in test_loader:
            images, labels = data
            outputs = net(images)  # 前向传播

            # 获取预测结果（选择概率最高的类别）
            _, predicted = torch.max(outputs.data, dim=1)

            total += labels.size(0)  # 更新总样本数
            correct += (predicted == labels).sum().item()  # 更新正确预测数

    # 打印测试准确率
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


# 主程序入口
if __name__ == '__main__':
    # 进行10轮训练
    for epoch in range(10):
        train(epoch)  # 训练模型
        test()  # 在测试集上评估模型