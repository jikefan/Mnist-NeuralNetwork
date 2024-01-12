import os
import sys
import numpy as np


def get_data_path():
    # 获取脚本的主模块文件路径
    main_script_path = sys.modules['__main__'].__file__
    # 获取该文件所在的目录
    project_root = os.path.dirname(os.path.abspath(main_script_path)) # type: ignore 忽略Pylance的报红
    # 数据所在路径
    data_path = os.path.join(project_root, "../data")
    
    return data_path

def main():
    print('Hello Neural Network!')
    
    test_set_filename = 'testset.npz'
    train_set_filename = 'trainset.npz'
    
    data_path = get_data_path()
    
    test_set_path = os.path.join(data_path, test_set_filename)
    train_set_path = os.path.join(data_path, train_set_filename)
    
    test_set = np.load(test_set_path)
    train_set = np.load(train_set_path)
    
    print(test_set)



if __name__ == "__main__":
    main()
