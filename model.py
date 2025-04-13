import numpy as np
import warnings


class MLP:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, activation='relu'):
        # 初始化权重和偏置
        self.params = {
            'W1': 0.001 * np.random.randn(input_size, hidden_size1),
            'b1': np.zeros((1, hidden_size1)),
            'W2': 0.001 * np.random.randn(hidden_size1, hidden_size2),
            'b2': np.zeros((1, hidden_size2)),
            'W3': 0.001 * np.random.randn(hidden_size2, output_size),
            'b3': np.zeros((1, output_size)),
        }
        self.activation = activation

    def fun_activation(self, X):
        # 激活函数：支持 ReLU 和 Sigmoid
        if self.activation == 'relu':
            return np.maximum(0, X)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-X))
        else:
            warnings.warn(
                f"Unsupported activation function '{self.activation}', defaulting to ReLU.", UserWarning)
            return np.maximum(0, X)

    def forward(self, X):
        # 前向传播过程
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        z1 = X @ W1 + b1
        a1 = self.fun_activation(z1)

        z2 = a1 @ W2 + b2
        a2 = self.fun_activation(z2)

        scores = a2 @ W3 + b3

        # 缓存中间变量用于反向传播
        self.cache = {
            'X': X, 'z1': z1, 'a1': a1,
            'z2': z2, 'a2': a2
        }
        return scores

    def loss(self, X, y, reg):
        # softmax 计算交叉熵损失和正则项
        num_train = X.shape[0]
        scores = self.forward(X)
        scores -= np.max(scores, axis=1, keepdims=True)  # 防止数值不稳定

        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        correct_logprobs = -np.log(probs[np.arange(num_train), y])
        data_loss = np.sum(correct_logprobs) / num_train

        reg_loss = 0.5 * reg * (
            np.sum(self.params['W1'] ** 2) +
            np.sum(self.params['W2'] ** 2) +
            np.sum(self.params['W3'] ** 2)
        )

        loss = data_loss + reg_loss
        return loss, probs

    def activation_grad(self, X):
        # 激活函数的导数
        if self.activation == 'relu':
            return (X > 0).astype(float)
        elif self.activation == 'sigmoid':
            sig = 1 / (1 + np.exp(-X))
            return sig * (1 - sig)
        else:
            return (X > 0).astype(float)

    def backward(self, y, probs, reg):
        # 反向传播过程
        num_train = y.shape[0]
        dscores = probs.copy()
        dscores[np.arange(num_train), y] -= 1
        dscores /= num_train

        # 第三层梯度
        dW3 = self.cache['a2'].T @ dscores + reg * self.params['W3']
        db3 = np.sum(dscores, axis=0, keepdims=True)

        # 第二层梯度
        da2 = dscores @ self.params['W3'].T
        dz2 = da2 * self.activation_grad(self.cache['z2'])
        dW2 = self.cache['a1'].T @ dz2 + reg * self.params['W2']
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # 第一层梯度
        da1 = dz2 @ self.params['W2'].T
        dz1 = da1 * self.activation_grad(self.cache['z1'])
        dW1 = self.cache['X'].T @ dz1 + reg * self.params['W1']
        db1 = np.sum(dz1, axis=0, keepdims=True)

        grads = {
            'W1': dW1, 'b1': db1,
            'W2': dW2, 'b2': db2,
            'W3': dW3, 'b3': db3
        }
        return grads
