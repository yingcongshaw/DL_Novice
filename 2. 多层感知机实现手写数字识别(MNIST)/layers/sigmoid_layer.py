""" Sigmoid 激活层 """

import numpy as np

class SigmoidLayer():
	def __init__(self):
		"""
		Sigmoid激活函数: f(x) = 1/(1+exp(-x))
		"""
		self.trainable = False

	def forward(self, Input):
		# 对输入应用Sigmoid激活函数并返回结果
		self.sigmoid =  1 / (1+np.exp(-Input))
		return self.sigmoid
		
	def backward(self, delta):
		# 根据delta计算梯度
	    return delta * self.sigmoid *(1-self.sigmoid)
