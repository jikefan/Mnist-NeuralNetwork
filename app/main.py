import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# 定义自定义数据集类
# 定义自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        sample_feature = self.features[idx]
        sample_label = self.labels[idx]
        
        if self.transform:
            sample_feature = self.transform(sample_feature)
            
        return torch.from_numpy(sample_feature).float(), torch.tensor(sample_label).long()  # 返回特征张量与标签张量（标签通常为LongTensor类型）

# 使用交叉熵损失函数
criterion = nn.CrossEntropyLoss()

class SimpleFullyConnetedNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_class):
        super(SimpleFullyConnetedNetwork, self).__init__()
        
        # 第一层：输入层到隐藏层的全连接层
        self.fc1 = nn.Linear(input_size, hidden_size)
        # 非线性激活函数
        self.relu = nn.ReLU()
        # 临时增加一个隐藏层
        self.fc11 = nn.Linear(hidden_size, hidden_size)
        self.fc12 = nn.Linear(hidden_size, hidden_size)
        
        # 第二层：隐藏层到输出层的全连接层
        self.fc2 = nn.Linear(hidden_size, num_class)
        # 加一个softmax
        self.softmax = nn.Softmax()

    def forward(self, x):
        # 前向传播
        x = x.view(-1, 1 * 28 * 28)
        out = self.fc1(x)
        out = self.relu(out)
        
        out = self.fc11(out)
        out = self.relu(out)
        
        out = self.fc12(out)
        out = self.relu(out)
        
        out = self.fc2(out)
        out = self.softmax(out)
        
        return out

def get_data_path():
    # 获取脚本的主模块文件路径
    main_script_path = sys.modules['__main__'].__file__
    # 获取该文件所在的目录
    project_root = os.path.dirname(os.path.abspath(main_script_path)) # type: ignore 忽略Pylance的报红
    # 数据所在路径
    data_path = os.path.join(project_root, "../data")
    
    return data_path


def print_img(arr):
    for row in arr:
        for c in row:
            if c[0] > 0:
                print('O', end='')
            else:
                print(' ', end='')
        print()


def show_img(arr):
    """
    绘制手写体图片
    """
    image_display = np.where(arr > 0, 1, 0).astype(np.uint8) * 255
    # 绘制图像
    plt.imshow(arr, cmap='gray', interpolation='nearest')
    plt.axis('off')  # 移除坐标轴
    plt.show()


def print_label(y_arr):
    """
    打印分类标签值
    """
    i = np.where(np.array(y_arr)==1)
    print(f'分类是: {i[0][0]}')


def main():
    print('Hello Neural Network!')
    
    test_set_filename = 'testset.npz'
    train_set_filename = 'trainset.npz'
    
    data_path = get_data_path()
    
    test_set_path = os.path.join(data_path, test_set_filename)
    train_set_path = os.path.join(data_path, train_set_filename)
    
    test_set = np.load(test_set_path)
    train_set = np.load(train_set_path)
    
    # print(test_set['X'][0].flatten())
    # print(test_set['Y'][0])
    # print(len(train_set['X']))
    
    # print(len((np.random.randn(784, 200) * 0.01)[0]))
    
    model = SimpleFullyConnetedNetwork(input_size=784, hidden_size=256, num_class=10)
    # 使用随机梯度下降优化器
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # 打印模型结构
    print(model)

    # 每次训练的样本数量
    batch_size = 100
    # 训练循环
    num_epochs = 60
    # 创建自定义数据集对象
    dataset = CustomDataset(train_set['X'] / 255.0, train_set['Y'])
    # 设置DataLoader参数，例如批量大小、是否shuffle等
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # 记录开始时间
    start_time = time.time()
    for epoch in range(num_epochs):
        loss = None
        # 现在可以通过DataLoader迭代获取批次数据
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # 设置模型为训练模式
            model.train(True)
            optimizer.zero_grad()  # 清零梯度缓存

            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 反向传播梯度
            loss.backward()
            
            # 更新权重
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}', end='\t')
        torch.save(model.state_dict(), 'simple_fc_model.pth')
        
        model.eval()  # 设置模型为评估模式
        test_dataset = CustomDataset(test_set['X'] / 255.0, test_set['Y'])
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        correct_count, total_count = 0, 0
        with torch.no_grad():  # 关闭梯度计算以提高性能
            for batch_idx, (inputs, targets) in enumerate(test_dataloader):
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)  # 获取每个样本的最大概率对应的类别
                total_count += targets.size(0)
                correct_count += (predicted == targets).sum().item()
            accuracy = correct_count / total_count
            print(f'准确度为: {accuracy} \t [{correct_count} / {total_count}]', end='\t')
            end_time = time.time()
            cpu_time = end_time - start_time
            print(f"当前训练耗费：{cpu_time:.6f} 秒")
    


if __name__ == "__main__":
    main()
