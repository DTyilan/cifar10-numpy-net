import numpy as np
import pickle
import os


import os
import pickle
import numpy as np


def load_cifar10_batch(filepath):
    """
    从单个 CIFAR-10 batch 文件中加载图像和标签。

    参数:
        filepath (str): CIFAR-10 数据 batch 文件路径。

    返回:
        X (np.ndarray): 图像数据，形状为 (10000, 32, 32, 3)，类型为 float。
        Y (np.ndarray): 标签数据，形状为 (10000,)。
    """
    with open(filepath, 'rb') as file:
        data_dict = pickle.load(file, encoding='bytes')
        X = data_dict[b'data']
        Y = data_dict[b'labels']

        # 变形并转换为 float 类型
        X = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32)
        Y = np.array(Y)

    return X, Y


def load_cifar10(root_dir):
    """
    加载 CIFAR-10 数据集（训练集+测试集）。

    参数:
        root_dir (str): 包含 CIFAR-10 批量数据文件的文件夹路径。

    返回:
        X_train (np.ndarray): 训练图像数据。
        y_train (np.ndarray): 训练标签数据。
        X_test (np.ndarray): 测试图像数据。
        y_test (np.ndarray): 测试标签数据。
    """
    X_train_list = []
    y_train_list = []

    for i in range(1, 6):
        batch_file = os.path.join(root_dir, f'data_batch_{i}')
        X_batch, y_batch = load_cifar10_batch(batch_file)
        X_train_list.append(X_batch)
        y_train_list.append(y_batch)

    X_train = np.concatenate(X_train_list)
    y_train = np.concatenate(y_train_list)

    X_test, y_test = load_cifar10_batch(os.path.join(root_dir, 'test_batch'))

    return X_train, y_train, X_test, y_test
