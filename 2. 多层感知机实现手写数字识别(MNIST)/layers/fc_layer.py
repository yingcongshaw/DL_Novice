""" 全连接层 """

import numpy as np

class FCLayer():
	def __init__(self, num_input, num_output, actFunction='relu', trainable=True):
		"""
		对输入进行线性变换: y = Wx + b
		参数简介:
			num_input: 输入大小
			num_output: 输出大小
			actFunction: 激活函数类型(无需修改)
			trainable: 是否具有可训练的参数
		"""
		self.num_input = num_input
		self.num_output = num_output
		self.trainable = trainable
		self.actFunction = actFunction
		assert actFunction in ['relu', 'sigmoid']

		self.XavierInit()

		self.grad_W = np.zeros((num_input, num_output))
		self.grad_b = np.zeros((1, num_output))

		self.W_v = 0
		self.b_v =0


	def forward(self, Input):
		# 对输入计算Wx+b并返回结果.
		self.Input = Input
		return np.dot(Input, self.W) + self.b


	def backward(self, delta):
		# 输入的delta由下一层计算得到
		# 根据delta计算梯度
		self.grad_W = np.dot(self.Input.T, delta)/self.Input.shape[0]
		self.grad_b = np.average(delta, axis=0)
		return np.dot(delta, self.W.T)


	def XavierInit(self):
        # 初始化权重，高斯分布（MSRA）
		raw_std = (2 / (self.num_input + self.num_output))**0.5
		if 'relu' == self.actFunction:
			init_std = raw_std * (2**0.5)
		elif 'sigmoid' == self.actFunction:
			init_std = raw_std
		else:
			init_std = raw_std # * 4

		self.W = np.random.normal(0, init_std, (self.num_input, self.num_output))
		self.b = np.random.normal(0, init_std, (1, self.num_output))
