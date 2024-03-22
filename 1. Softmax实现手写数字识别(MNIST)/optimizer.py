import numpy as np

class SGD(object):
    def __init__(self, model, learning_rate, momentum=0.0):
        self.model = model
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.W_v=0
        self.b_v=0

    def step(self):
        """One updating step, update weights"""

        layer = self.model
        
        if layer.trainable:
            #Weight update with momentum
            if self.momentum:
                self.W_v = self.momentum * self.W_v - self.learning_rate*layer.grad_W
                self.b_v = self.momentum * self.b_v - self.learning_rate*layer.grad_b
                layer.W += self.momentum * self.W_v
                layer.b += self.momentum * self.b_v            
            #Weight update without momentum
            else:
                layer.W -= self.learning_rate * layer.grad_W
                layer.b -= self.learning_rate * layer.grad_b
