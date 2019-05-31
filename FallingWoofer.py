import quaternion
import numpy as np
from WooferConfig import WOOFER_CONFIG

class FallingWoofer():
	def __init__(self, x0, dt, accel_sigma, gyro_sigma):
		self.J = WOOFER_CONFIG.INERTIA
		self.J_inv = np.linalg.inv(self.J)
		self.r_fr = np.array([WOOFER_CONFIG.LEG_FB, 	-WOOFER_CONFIG.LEG_LR, 0])
		self.r_fl = np.array([WOOFER_CONFIG.LEG_FB, 	 WOOFER_CONFIG.LEG_LR, 0])
		self.r_br = np.array([-WOOFER_CONFIG.LEG_FB, 	-WOOFER_CONFIG.LEG_LR, 0])
		self.r_bl = np.array([-WOOFER_CONFIG.LEG_FB, 	 WOOFER_CONFIG.LEG_LR, 0])

		self.x = x0

		self.dt = dt
		self.accel_sigma = accel_sigma
		self.gyro_sigma = gyro_sigma

	def update(self, u):
		# print("True: ", self.x)
		k1 = self.dt*self.dynamics(self.x, u)
		k2 = self.dt*self.dynamics(self.x+k1/2.0, u)
		k3 = self.dt*self.dynamics(self.x+k2/2.0, u)
		k4 = self.dt*self.dynamics(self.x+k3, u)
		self.x = self.x + (k1+2.0*k2+2.0*k3+k4)/6.0
		self.x[3:7] = quaternion.normalize(self.x[3:7])

		return self.x

	def dynamics(self, x, u):
		xdot = np.zeros(np.shape(x))

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

		# sensor bias
		# xdot[13:16] = np.random.normal(0, 0, (3))
		# xdot[16:19] = np.random.normal(0, 0, (3))

		return xdot

	def meas(self, x, u):
		z = np.zeros((6))

		a_w = 1/WOOFER_CONFIG.MASS * (u[0:3] + u[3:6] + u[6:9] + u[9:12])

		q_a_w = quaternion.fromVector(a_w)

		q_a_b = quaternion.vectorRotation(quaternion.inv(x[3:7]), q_a_w)


		# rotated accelerometer measurement:
		z[0:3] = q_a_b[1:4] + np.random.normal(0, self.accel_sigma, (3)) # + x[13:16]

		# angular velocity measurement
		z[3:6] = x[10:13] + np.random.normal(0, self.gyro_sigma, (3)) # + x[16:19]

		return z
