import numpy as np
from StateEstimator import UKFStateEstimator
from WooferConfig import WOOFER_CONFIG
from FallingWoofer import FallingWoofer
import quaternion
import pdb

dt = 0.01

L = 13

x0 = np.zeros(L)
x0[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
x0[10:13] = np.array([1, 1, 1])

accel_sigma = 0.1
gyro_sigma 	= 0.1

x0_est = np.zeros(L)

x0_est[3:7] = np.array([1.0, 0.0, 0.0, 0.0])# quaternion.exp(phi)
x0_est[10:13] = np.array([0.8, 1.2, 0.9])

ukf_state_est 	= UKFStateEstimator(x0_est, dt)
model 			= FallingWoofer(x0, dt, accel_sigma, gyro_sigma)

N = 500

true_state = np.zeros((L, N+1))
true_state[:,0] = x0

state_est = np.zeros((L, N+1))
state_est[:,0] = x0_est

# u = WOOFER_CONFIG.MASS * np.array([0, 0, 9.81/4]*4)
u = np.array([0, 0, 0]*4)

true_state_i = x0
for i in range(1, N):
	true_state_i = model.update(u)
	true_state_i[3:7] = quaternion.normalize(true_state_i[3:7])
	true_state[:,i] = true_state_i
	z_meas = model.meas(true_state_i, u)

	# pdb.set_trace()

	state_est_i = ukf_state_est.update(z_meas, u)

	state_est[0:3, i] = state_est_i['p']
	state_est[3:7, i] = state_est_i['q']
	state_est[7:10, i] = state_est_i['p_d']
	state_est[10:13, i] = state_est_i['w']
	# state_est[13:16, i] = state_est_i['b_a']
	# state_est[16:19, i] = state_est_i['b_g']

	q_true = true_state_i[3:7]
	q_est = state_est_i['q']

	q_d = quaternion.prod(quaternion.inv(q_est), q_true)
	phi = np.linalg.norm(quaternion.log(q_d))
	phi = np.abs((phi + np.pi) % (2 * np.pi) - np.pi)

	print(phi)

np.savez('falling_brick_state_log', true_state)
np.savez('falling_brick_state_est_log', state_est)
