""" 定义网络 """

class Network():
	def __init__(self):
		self.layerList = []
		self.numLayer = 0

	def add(self, layer):
		self.numLayer += 1
		self.layerList.append(layer)

	def forward(self, x):
		# 逐层前向计算
		for i in range(self.numLayer):
			x = self.layerList[i].forward(x)
		return x

	def backward(self, delta):
		# 逐层后向计算
		for i in reversed(range(self.numLayer)): # 逆向遍历
			delta = self.layerList[i].backward(delta)
