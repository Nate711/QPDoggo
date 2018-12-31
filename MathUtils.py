import numpy as np
def CrossProductMatrix(a):
	return np.array([[0, -a[2], a[1]], \
					 [a[2], 0, -a[0]], \
					 [-a[1], a[0], 0]])