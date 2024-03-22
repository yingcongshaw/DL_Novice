""" ReLU激活层 """

import numpy as np

class ReLULayer():
	def __init__(self):
		"""
		ReLU激活函数: relu(x) = max(x, 0)
		"""
		self.trainable = False # 没有可训练的参数

	def forward(self, Input):
		# 对输入应用ReLU激活函数并返回结果
		self.Input = Input
		return np.where(Input>0,Input,0)

	def backward(self, delta):
		# 根据delta计算梯度
		delta[self.Input<0]=0
		return delta
