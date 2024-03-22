""" SGD优化器 """

import numpy as np

class SGD():
	def __init__(self, learningRate, weightDecay, momentum=0.):
		self.learningRate = learningRate
		self.weightDecay = weightDecay
		self.momentum =momentum
		
	# 一步反向传播，逐层更新参数
	def step(self, model):
		layers = model.layerList
		for layer in layers:
			if layer.trainable:

				# 使用layer.grad_W和layer.grad_b计算diff_W and diff_b.
				if self.momentum:
					layer.W_v = self.momentum * layer.W_v - self.learningRate * (layer.grad_W + self.weightDecay * layer.W)
					layer.b_v = self.momentum * layer.b_v - self.learningRate * layer.grad_b
					layer.diff_W = layer.W_v
					layer.diff_b = layer.b_v
				else:
					# Weight update without momentum
					layer.diff_W = -self.learningRate * (layer.grad_W + self.weightDecay * layer.W)
					layer.diff_b = -self.learningRate * layer.grad_b

				# Weight update
				layer.W += layer.diff_W
				layer.b += layer.diff_b
