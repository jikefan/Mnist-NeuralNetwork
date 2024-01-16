import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from main import SimpleFullyConnetedNetwork, show_img, get_data_path


def test(index):
    if index > 10000:
        print('index大于测试数据集10000')
        return
    # 加载模型
    model = SimpleFullyConnetedNetwork(input_size=784, hidden_size=256, num_class=10)
    model.load_state_dict(torch.load('simple_fc_model.pth'))
    model.eval()  # 如果需要做预测，设置模型为评估模式
    
    test_set_filename = 'testset.npz'
    
    data_path = get_data_path()
    
    test_set_path = os.path.join(data_path, test_set_filename)
    
    test_set = np.load(test_set_path)

    label = torch.argmax(torch.Tensor(test_set['Y'][index])).item()
    
    sample_data = torch.Tensor(test_set['X'][index] / 255.0)
    
    show_img(sample_data)
    
    output = model(sample_data)
    
    prediction = torch.argmax(output, dim = 1).item()
    
    print(f'预测数字是: {prediction}')
    print(f'标签值是: {label}')
    if prediction == label:
        print('预测正确!')

if __name__ == '__main__':
    test(5)