import numpy as np
def CrossProductMatrix(a):
	return np.array([[0, -a[2], a[1]], \
					 [a[2], 0, -a[0]], \
					 [-a[1], a[0], 0]])

class RunningMax():
	def __init__(self, size):
		self.max = np.zeros(size)
	def Update(self, a):
		self.max = np.maximum(self.max, a)
	def CurrentMax(self):
		return self.max