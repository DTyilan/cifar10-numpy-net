import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


def visualize_weights(params, save_dir='ckpts'):
    """
    可视化权重矩阵并保存为图片。

    参数:
        params (dict): 包含 'W1', 'W2', 'W3' 三个权重矩阵的字典。
        save_dir (str): 图片保存目录，默认为 'ckpts'。
    """
    os.makedirs(save_dir, exist_ok=True)

    #  可视化第一层权重 W1: 输入层 -> 隐藏层1 
    W1 = params['W1']  
    num_filters = W1.shape[1]
    filter_size = int(np.sqrt(W1.shape[0] / 3)) 
    grid_size = int(np.ceil(np.sqrt(num_filters)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            ax = axes[i, j]
            if idx < num_filters:
                filt = W1[:, idx].reshape(
                    3, filter_size, filter_size).transpose(1, 2, 0)
                filt = (filt - filt.min()) / (filt.max() - filt.min() + 1e-6)
                ax.imshow(filt)
            ax.axis('off')

    plt.suptitle('W1: Input to Hidden Layer 1')
    plt.savefig(os.path.join(save_dir, 'W1_visualization.png'))
    plt.close()

    # 可视化第二层权重 W2: 隐藏层1 -> 隐藏层2 
    W2 = params['W2']  

    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.imshow(W2.T, aspect='auto', cmap='viridis')
    ax.set_title('W2: Hidden Layer 1 to Hidden Layer 2')
    ax.set_xlabel('Hidden Units in Layer 1')
    ax.set_ylabel('Hidden Units in Layer 2')
    plt.colorbar(cax, ax=ax, label='Weight Value')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'W2_heatmap.png'))
    plt.close()

    #  可视化第三层权重 W3: 隐藏层2 -> 输出层 
    W3 = params['W3']  
    hidden2_dim, num_classes = W3.shape

    fig, axes = plt.subplots(num_classes, 1, figsize=(8, 2 * num_classes))
    if num_classes == 1:
        axes = [axes]  

    for i in range(num_classes):
        weights = W3[:, i]
        weights = (weights - weights.min()) / \
            (weights.max() - weights.min() + 1e-6)
        axes[i].bar(np.arange(hidden2_dim), weights)
        axes[i].set_title(f'Class {i}')
        axes[i].set_ylabel('Weight')
        axes[i].set_xlim([0, hidden2_dim])

    plt.suptitle('W3: Hidden Layer 2 to Output Layer')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'W3_barplots.png'))
    plt.close()


if __name__ == "__main__":
    model_path = 'ckpts/best_params.pkl'
    with open(model_path, 'rb') as f:
        best_params = pickle.load(f)

    visualize_weights(best_params)
