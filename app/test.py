import torch
from torchvision import datasets, transforms

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
])

# 加载MNIST训练数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 查看第一个样本的标签
first_label = train_dataset.train_labels[0]
print(f"First label type: {type(first_label)}")  # 输出应为：<class 'numpy.int64'>