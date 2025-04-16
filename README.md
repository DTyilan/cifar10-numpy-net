# CIFAR-10 图像分类 - 基于 NumPy 实现的多层感知机（MLP）

本项目实现了一个用于 CIFAR-10 图像分类任务的多层感知机（MLP）模型，采用纯 NumPy 手动实现前向传播与反向传播，以便深入理解神经网络结构与梯度计算过程。

## 🧠 特性
- 手动实现前向传播与反向传播算法
- 支持多层神经网络结构
- 支持 ReLU 和 Sigmoid 激活函数
- 可视化权重与训练过程
- 支持训练集、验证集、测试集评估

## 📁 数据集准备
本项目使用 [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) 数据集，请按以下步骤操作：

1. 下载 CIFAR-10 数据集（Python 版本）。
2. 解压后找到名为 `cifar-10-batches-py`的文件夹。
3. 将该文件夹放置在项目根目录下（即与 `main.py` 同级）。

目录结构示例：

```
project/
├── main.py
├── model.py
├── dataset.py
├── visualize.py
├── cifar-10-batches-py/
│   ├── data_batch_1
│   ├── ...
│   └── test_batch
```

## 🚀 如何运行

确保数据准备完成后，在项目根目录下运行：

```bash
python main.py

训练过程将输出训练集、验证集、测试集的准确率。训练完成后，会生成loss曲线与accuracy曲线。

如需进行权重可视化，可运行：

```bash
python visualize.py
可视化图像将保存在当前目录下。

## 📦 环境要求

- Python 3.8 及以上版本
- NumPy 1.24.3 及以上版本
- Matplotlib 3.7.5 及以上版本
- `pickle`（Python 内置模块）


## 📝 说明

本项目旨在帮助学习者深入理解神经网络内部机制，特别是权重更新与误差反向传播过程。相比于使用深度学习框架（如 PyTorch、TensorFlow），本项目采用手工实现方式更适合教学与算法原理学习。
Happy coding! 🎉
