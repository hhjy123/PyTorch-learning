# 导入torchvision库，用于计算机视觉相关任务
import torchvision
# 导入DataLoader类，用于批量加载数据
from torch.utils.data import DataLoader
# 导入SummaryWriter类，用于将数据写入TensorBoard
from torch.utils.tensorboard import SummaryWriter

# 准备测试数据集
# 使用CIFAR10数据集，参数说明：
# root：数据集存储路径
# train=False：使用测试集（非训练集）
# transform：将PIL图像转换为PyTorch张量
# download=True：如果本地不存在则自动下载
test_data = torchvision.datasets.CIFAR10(
    root='./CIFAR10_DATA',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)
"""
DataLoader核心功能解析
1. 自动批处理（Batching）
   • 功能：将多个数据样本组合成一个批次（Batch），减少内存开销并利用硬件并行计算。
   • 参数：batch_size=N（例如 batch_size=32 表示每个批次含 32 个样本）
2. 数据打乱（Shuffling）
   • 功能：每个训练周期（Epoch）开始时随机重排数据顺序，避免模型学习到样本顺序特征。
   • 参数：shuffle=True（训练集通常启用，测试集禁用）。
3. 并行数据加载（Multiprocessing）
   • 功能：利用多进程预加载数据，避免 GPU/CPU 等待 I/O 阻塞。
   • 参数：num_workers=K（K 为子进程数，一般设为 CPU 核数）。
4. 内存管理优化
   • 功能：
     ◦ pin_memory=True：将数据直接加载到 GPU 的锁页内存（Pinned Memory），加速 CPU→GPU 传输。
     ◦ drop_last=  True：丢弃最后一个不完整的批次（当样本数无法被 batch_size 整除时）。
"""
# 创建测试数据加载器（第一次定义，但会被下面的第二次定义覆盖）
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

# 重新定义测试数据加载器（覆盖前一个定义），参数说明：
# dataset：使用的数据集
# batch_size=64：每批加载64张图像
# shuffle=True：打乱数据顺序
# num_workers=0：使用主进程加载数据（无子进程）
# drop_last=False：保留最后不足批量的样本
test_loader = DataLoader(
    dataset=test_data,
    batch_size=64,
    shuffle=True,
    num_workers=0,
    drop_last=False
)

# 获取测试数据集中的第一张图像及其标签
img, target = test_data[0]
# 打印图像张量的形状（通道数×高度×宽度）
print(img.shape)  # 输出：torch.Size([3, 32, 32])
# 打印图像的类别标签（0-9之间的整数）
print(target)     # 输出：3（代表猫类别）

# 创建SummaryWriter实例，日志将保存在"TensorBoard"目录
writer = SummaryWriter("TensorBoard")

# 进行2轮数据遍历（epoch循环）
for epoch in range(2):
    step = 0  # 初始化步数计数器
    # 遍历测试数据加载器中的所有批次
    for data in test_loader:
        # 解包批次数据：图像张量和标签
        imgs, targets = data
        # 将当前批次图像添加到TensorBoard
        # 'Epoch：{}'：标量名称包含当前epoch
        # imgs：图像张量（64×3×32×32）
        # step：当前批次的步数索引
        writer.add_images('Epoch：{}'.format(epoch), imgs, step)
        step += 1  # 更新步数计数器

# 关闭SummaryWriter，确保所有数据写入磁盘
writer.close()