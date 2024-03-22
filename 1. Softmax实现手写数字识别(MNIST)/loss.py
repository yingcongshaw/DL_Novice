import numpy as np

# 定义了一个非常小的常数EPS，用于表示浮点数计算中的误差范围
EPS = 1e-11

class SoftmaxCrossEntropyLoss(object):

    def __init__(self, num_input, num_output, trainable=True):
        """
        Apply a linear transformation to the incoming data: y = Wx + b
        Args:
            num_input: size of each input sample
            num_output: size of each output sample
            trainable: whether if this layer is trainable
        """

        self.num_input = num_input
        self.num_output = num_output
        self.trainable = trainable
        self.XavierInit()

        self.y_softmax=[]
        self.y_labels = []
        self.Input = []
        self.grad_W=[]
        self.grad_b=[]

    def forward(self, Input, labels):
        """
          Inputs: (minibatch)
          - Input: (batch_size, 784)
          - labels: the ground truth label, shape (batch_size, )
        """

        # 输入数据Input重新整形为二维数组
        self.Input= Input.reshape(-1, self.num_input)
        # 创建一个大小为self.num_output的单位矩阵，然后通过[labels]将矩阵的行索引转换为对应的One-hot编码。
        self.y_labels = np.eye(self.num_output)[labels]

        # softmax
        y_raw =np.dot(self.Input, self.W) + self.b
        y_exp=np.exp(y_raw)
        y_sum=np.sum(y_exp, axis=1,keepdims=True)
        self.y_softmax = y_exp/y_sum

        # 计算交叉熵
        e = -np.log(np.sum(self.y_softmax* self.y_labels, axis=1))
        loss = np.average(e)
        acc = (self.y_softmax.argmax(axis=1)==labels).mean()
        return loss, acc

    def gradient_computing(self):

        # 计算损失函数的导数（delta）
        delta  =  self.y_labels - self.y_softmax
        # 计算权重（W）的梯度
        self.grad_W = -np.dot(self.Input.T, delta ) / len(self.Input)
        # 用于计算 delta 沿着每一列的平均值，即每个神经元的偏置。
        self.grad_b = -np.average(delta , axis=0)


    def XavierInit(self):
        """
        初始化神经网络的权重（W）和偏置（b）的值
        """
        # 初始化权重，高斯分布（MSRA）
        raw_std = (2 / (self.num_input + self.num_output))**0.5
        # 将原始标准差乘以 2 的平方根，以得到一个较大的初始标准差
        init_std = raw_std * (2**0.5)
        # 生成一个服从正态分布的随机数组，其均值为 0，标准差为 init_std 
        self.W = np.random.normal(0, init_std, (self.num_input, self.num_output))
        self.b = np.random.normal(0, init_std, (1, self.num_output))