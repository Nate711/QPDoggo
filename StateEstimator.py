import WooferDynamics
import numpy as np
import quaternion

from WooferConfig import WOOFER_CONFIG

class StateEstimator():
	"""
	Interface for generic state estimator
	"""
	def update(self, sim):
		"""
		Get data from the simulation and return
		an estimate of the robot state
		"""
		raise NotImplementedError

class MuJoCoStateEstimator(StateEstimator):
	"""
	Grab state data directly from the simulation
	"""
	def __init__(self):
		pass
	def update(self, sim, u):
		"""
		Grab the state directly from the MuJoCo sim, aka, cheat.
		"""

		xyz 		= WooferDynamics.position(sim)
		v_xyz 		= WooferDynamics.velocity(sim)
		quat_orien 	= WooferDynamics.orientation(sim)
		ang_vel 	= WooferDynamics.angular_velocity(sim)
		joints 		= WooferDynamics.joints(sim)
		joint_vel 	= WooferDynamics.joint_vel(sim)
		# rpy 		= rotations.quat2euler(quat_orien)

		state_est = {"p":xyz, "p_d":v_xyz, "q":quat_orien, "w":ang_vel, "j":joints, "j_d":joint_vel}

		return state_est

class UKFStateEstimator(StateEstimator):
	"""
	UKF based state estimation with quaternions for orientation
	"""
	def __init__(self, x0, dt): #, contact_estimator):
		# self.contact_estimator = contact_estimator

		self.x = x0
		self.alpha = 0.15
		self.L = np.size(self.x)
		self.P = 0.1 * np.eye(self.L - 1)
		self.J = WOOFER_CONFIG.INERTIA
		self.dt = dt
		self.J_inv = np.linalg.inv(self.J)

		self.Q = 0.01 * np.eye(self.L-1)
		self.R = 0.01 * np.eye(6)

		# leg offsets
		self.r_fr = np.array([WOOFER_CONFIG.LEG_FB, 	-WOOFER_CONFIG.LEG_LR, 0])
		self.r_fl = np.array([WOOFER_CONFIG.LEG_FB, 	 WOOFER_CONFIG.LEG_LR, 0])
		self.r_br = np.array([-WOOFER_CONFIG.LEG_FB, 	-WOOFER_CONFIG.LEG_LR, 0])
		self.r_bl = np.array([-WOOFER_CONFIG.LEG_FB, 	 WOOFER_CONFIG.LEG_LR, 0])

	def update(self, z_meas, u):#sim, u):
		# z_meas = self.getSensorMeasurements(sim)
		# print("Condition number: ", np.linalg.cond(self.P))
		# print("Sensor measurement: ", z_meas)
		# print("Control: %s", u)
		# print("Estimated: ", self.x)

		X_x = self.calcSigmas()

		# number of sigma points generated
		n = X_x.shape[1]

		# weighting vectors for mean/covariance calculation
		w_m = 1/(n)*np.ones((n))
		w_c = 1/(self.alpha**2) * w_m

		x_p = np.zeros((self.L, n))

		# propogate sigma points through dynamics via RK4 integration
		for k in range(n):
			k1 = self.dt*self.dynamics(X_x[:,k], u)
			k2 = self.dt*self.dynamics(X_x[:,k]+k1/2, u)
			k3 = self.dt*self.dynamics(X_x[:,k]+k2/2, u)
			k4 = self.dt*self.dynamics(X_x[:,k]+k3, u)
			x_p[:,k] = X_x[:,k] + (k1+2*k2+2*k3+k4)/6
			x_p[3:7,k] = quaternion.normalize(x_p[3:7,k])

		x_bar = np.zeros((self.L))

		x_bar[0:3] = x_p[0:3,:] @ w_m
		x_bar[3:7] = self.quaternionAverage(x_p[3:7,:], w_m)
		x_bar[7:(self.L)] = x_p[7:(self.L),:] @ w_m

		z_p = np.zeros((z_meas.size, n))

		# propogate predicted states through measurement model
		for k in range(n):
			z_p[:,k] = self.meas(x_p[:,k], u)
		#
		z_bar = z_p @ w_m
		#
		dX = np.zeros((self.L-1, n))
		#

		dX[0:3,:] = x_p[0:3,:] - np.tile(x_bar[0:3][np.newaxis].T, (1, n))
		dX[3:6,:] = self.calcQuatDiff(x_p[3:7, :], self.x[3:7])
		dX[6:(self.L-1),:] = x_p[7:self.L,:] - np.tile(x_bar[7:self.L][np.newaxis].T, (1, n))

		dZ = z_p - np.tile(z_bar[np.newaxis].T, (1, n))
		#
		# calculate Pxx
		Pxx = self.Q + self.calcCovariance(dX, dX, w_c)

		# calculate Pxz
		Pxz = self.calcCovariance(dX, dZ, w_c)

		# calculate Pzz
		Pzz = self.calcCovariance(dZ, dZ, w_c)

		# Innovation
		nu = z_meas - z_bar

		# print("Nu: ", nu)

		#Innovation Covariance
		S = Pzz + self.R

		# Kalman Gain
		K = Pxz @ np.linalg.inv(S)

		x_update = K @ nu

		q_prev = self.x[3:7]

		# multiplicative quaternion update
		self.x[0:3] = x_bar[0:3] + x_update[0:3]
		self.x[3:7] = quaternion.prod(x_bar[3:7], quaternion.exp(x_update[3:6]))
		self.x[3:7] = np.sign(q_prev.T @ self.x[3:7]) * self.x[3:7]

		self.x[7:(self.L)] = x_bar[7:(self.L)] + x_update[6:(self.L-1)]

		self.P = Pxx - K @ S @ K.T

		# cheating here for now:
		joints 		= np.zeros(12)#WooferDynamics.joints(sim)
		joint_vel 	= np.zeros(12)#WooferDynamics.joint_vel(sim)

		state_est = {"p":self.x[0:3], "p_d":self.x[7:10], "q":self.x[3:7], "w":self.x[10:13], \
						"j":joints, "j_d":joint_vel, "b_a":self.x[13:16], "b_g":self.x[16:19]}

		return state_est

	def getSensorMeasurements(self, sim):
		accelerometer_sensor = WooferDynamics.accel_sensor(sim)
		gyro_sensor = WooferDynamics.gyro_sensor(sim)
		joint_sensor = WooferDynamics.joint_sensor(sim)

		# z_meas = block([accelerometer_sensor, gyro_sensor, joint_sensor])
		z_meas = np.block([accelerometer_sensor, gyro_sensor])

		return z_meas

	def dynamics(self, x, u):
		xdot = np.zeros((self.L))

		q = x[3:7]
		v = x[7:10]
		om = x[10:13]

		# velocity
		xdot[0:3] = v

		# quaternion kinematics
		xdot[3:7] = 0.5*quaternion.prod(q, quaternion.fromVector(om))

		# acceleration

		# sum foot forces in the world frame
		force = u[0:3] + u[3:6] + u[6:9] + u[9:12]
		xdot[7:10] = 1/WOOFER_CONFIG.MASS * (force) + np.array([0,0,-9.81])

		# angular acceleration

		# go from world to body
		f_fr_body = quaternion.vectorRotation(quaternion.inv(x[3:7]), quaternion.fromVector(u[0:3]))[1:4]
		f_fl_body = quaternion.vectorRotation(quaternion.inv(x[3:7]), quaternion.fromVector(u[3:6]))[1:4]
		f_br_body = quaternion.vectorRotation(quaternion.inv(x[3:7]), quaternion.fromVector(u[6:9]))[1:4]
		f_bl_body = quaternion.vectorRotation(quaternion.inv(x[3:7]), quaternion.fromVector(u[9:12]))[1:4]

		# calculate torques
		torque = np.cross(self.r_fr, f_fr_body) + np.cross(self.r_fl, f_fl_body) + np.cross(self.r_br, f_br_body) + np.cross(self.r_bl, f_bl_body)

		xdot[10:13] = self.J_inv @ (torque - np.cross(om, self.J @ om))

		# foot positions
		# xdot[13:25] = np.zeros((12,1))

		# sensor bias
		xdot[13:16] = np.zeros((3))
		xdot[16:19] = np.zeros((3))

		return xdot

	def meas(self, x, u):
		z = np.zeros((6))

		a_w = 1/WOOFER_CONFIG.MASS * (u[0:3] + u[3:6] + u[6:9] + u[9:12])

		q_a_w = quaternion.fromVector(a_w)

		q_a_b = quaternion.vectorRotation(quaternion.inv(x[3:7]), q_a_w)


		# rotated accelerometer measurement:
		z[0:3] = q_a_b[1:4] + x[13:16]

		# angular velocity measurement
		z[3:6] = x[10:13] + x[16:19]

		return z

	def calcSigmas(self):
		"""
		Calculate the sigma points through a Cholesky decomposition
		"""
		n = 2*(self.L-1)

		X_x = np.zeros((self.L, n))

		A = self.alpha * np.linalg.cholesky(self.P).T

		for i in range(self.L-1):
			# do simple additive sigma point calculation
			X_x[0:3, i] = self.x[0:3] + A[0:3, i]
			X_x[0:3, i+(self.L-1)] = self.x[0:3] - A[0:3, i]

			X_x[7:(self.L), i] = self.x[7:(self.L)] + A[6:(self.L), i]
			X_x[7:(self.L), i+(self.L-1)] = self.x[7:(self.L)] - A[6:(self.L), i]

			# do quaternion multiplicative update
			phi = A[3:6,i]
			dq_plus = quaternion.exp(phi)
			dq_neg = quaternion.exp(-phi)

			X_x[3:7, i] = quaternion.prod(self.x[3:7], dq_plus)
			X_x[3:7, i+(self.L-1)] = quaternion.prod(self.x[3:7], dq_neg)

		return X_x

	def quaternionAverage(self, Q, w_m):
		"""
		Compute the average quaternion of a set of quaternions

		For more information, see:
		Markley, F. Landis, et al. "Averaging quaternions."
		Journal of Guidance, Control, and Dynamics 30.4 (2007): 1193-1197.
		"""
		A = w_m[0] * Q @ Q.T

		return quaternion.normalize(np.linalg.eigh(A)[1][:, -1])

	def calcQuatDiff(self, q_p, q_bar):
		"""
		Calculate the three parameter quaternion difference from a matrix
		of quaternions
		"""
		q_bar_inv = quaternion.inv(q_bar)

		n = np.size(q_p[0,:])

		dQ = np.zeros((3, n))
		for i in range(n):
			dq_i = quaternion.prod(q_bar_inv, q_p[:,i])
			dQ[:, i] = quaternion.log(dq_i)

		return dQ

	def calcCovariance(self, D1, D2, w_c):
		"""
		Calculate covariance given two difference matrices
		"""
		return D1 @ np.diag(w_c) @ D2.T
