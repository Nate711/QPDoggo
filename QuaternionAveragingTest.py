import numpy as np
import quaternion

def quaternionAverage(Q, w_m):
	"""
	Compute the average quaternion of a set of quaternions

	For more information, see:
	Markley, F. Landis, et al. "Averaging quaternions."
	Journal of Guidance, Control, and Dynamics 30.4 (2007): 1193-1197.
	"""
	B = Q @ np.diag(np.sqrt(w_m))
	A = B @ B.T
	# A = Q @ np.diag(w_m) @ Q.T

	return quaternion.normalize(np.linalg.eigh(A)[1][:, -1])

N = 100

Q = np.zeros((4,N))
q_nom = np.array([1, 0, 0, 0])


w_m = 1/N * np.ones((N))

for i in range(N):
	Q[:,i] = quaternion.prod(q_nom, quaternion.exp(np.random.normal(0, 0.1, (3))))


q_avg = quaternionAverage(Q, w_m)
print("Quaternion average: ", q_avg)
print("Angle: ", np.linalg.norm(quaternion.log(q_avg)))
