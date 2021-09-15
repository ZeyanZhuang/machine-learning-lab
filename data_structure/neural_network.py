import numpy as np


class Layer:
    def __init__(self, input_dim, output_dim):
        # 维度
        self.input_dim = input_dim
        self.output_dim = output_dim
        # 权重
        self.weights = np.random.randn(input_dim, output_dim)
        # 偏置值
        self.bias = np.random.randn(output_dim)
        # 输出
        self.output = None
        # 梯度
        self.delta = np.zeros(output_dim, dtype=np.float64)

    def update_params(self, lr_rate, inputs):
        """更新参数
        
        Parameters:
        ----------
        lr_rate: 学习率
        inputs: 上一层的输入 (反向传播的时候是 w[i, j] -= \delta[j] * input[i])
        """
        # 梯度
        grad = np.matmul(inputs.reshape(self.input_dim, 1), self.delta.reshape(1, self.output_dim))
        self.weights -= lr_rate * grad
        self.bias -= lr_rate * self.delta


def sigmoid_derivative(y):
    ''' sig_moid 的导数'''
    return y * (1.0 - y)


def sigmoid(x):
    '''sig_moid 激活函数'''
    return 1 / (1 + np.exp(-x))


class NeuralNetwork:
    def __init__(self, dim_list):
        # 网络
        self.layers = []
        # 初始化
        for i in range(len(dim_list) - 1):
            layer = Layer(dim_list[i], dim_list[i + 1])
            self.layers.append(layer)

    def fun_leayer(self, x, layer: Layer):
        '''计算一层的输出

        Parameters:
        ----------
        x: 输入
        layer: 网络
        '''
        y = np.dot(x, layer.weights) + layer.bias
        return y

    def forward(self, x):
        """前馈

        Parameters:
        ----------
        x: 一个数据， np.array 类型
        """
        in_data = x
        for layer in self.layers:
            # 每层前馈
            y = self.fun_leayer(in_data, layer)
            # 过激活函数
            y = sigmoid(y)
            layer.output = y
            in_data = y
        return self.layers[-1].output

    def back_propagation(self, y_hat):
        """反向传播算法

        Parameters:
        ----------
        y_hat: 神经网络输出的最终目标， np.array 类型
        """
        for i in reversed(range(len(self.layers))):
            if i == len(self.layers) - 1:
                error = self.layers[i].output - y_hat
            else:
                error = np.dot(self.layers[i + 1].weights, self.layers[i + 1].delta)
            error = error * sigmoid_derivative(self.layers[i].output)
            self.layers[i].delta += error

    def zero_grad(self):
        """ 梯度清零 """
        for layer in self.layers:
            layer.delta = np.zeros(layer.output_dim)

    def upgrade_params(self, x, lr_rate):
        """更新参数

        Parameters:
        ----------
        x: 输入的那个数据（对应一次前馈）
        lr_rate : 学习率
        """
        for i, layer in enumerate(self.layers):
            if i == 0:
                # 第一层， input就是数据 x
                inputs = x
            else:
                inputs = self.layers[i - 1].output
            # 每层的参数更新
            layer.update_params(lr_rate=lr_rate, inputs=inputs)

    def train(self, epochs, data, y, lr_rate):
        """训练

        Parameters:
        ----------
        epochs: 训练次数
        data: 数据
        y: 输出 (对应 label 的独热码)
        lr_rate : 学习率
        """
        # 梯度清零
        self.zero_grad()
        for i in range(epochs):
            for x, y_hat in zip(data, y):
                # 前馈
                self.forward(x)
                # BP 算法
                self.back_propagation(y_hat)
                # 更新参数(这里也可以实现一个更新的步长)
                self.upgrade_params(x=x, lr_rate=lr_rate)
                # 梯度清零
                self.zero_grad()

    def predict(self, x):
        """ 预测 """
        for layer in self.layers:
            x = sigmoid(self.fun_leayer(x, layer))
        # 返回概率最大的类别
        return np.argmax(x)

    def validation(self, data_vali, y_vali):
        """ 验证


        Parameters:
        ----------
        data_vali: 验证的数据
        y_vali: 输出 (对应 label )

        Returns:
        -------
        acc_rate: 准确率
        """
        y_pred_list = []
        for x in data_vali:
            y_pred = self.predict(x)
            y_pred_list.append(y_pred)
        # print(y_pred_list)
        total_size = data_vali.shape[0]
        count_correct = 0
        for x, y in zip(y_pred_list, y_vali):
            if x == y:
                count_correct += 1
        acc_rate = count_correct / total_size
        return acc_rate