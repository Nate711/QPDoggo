import numpy as np

class WooferConfig:
	def __init__(self):
		# Robot joint limits
		self.MAX_JOINT_TORQUE = 8
		self.MAX_LEG_FORCE = 133
		self.REVOLUTE_RANGE = 3
		self.PRISMATIC_RANGE = 0.18

		# Robot geometry
		self.LEG_FB = 0.23					# front-back distance from center line to leg axis
		self.LEG_LR = 0.175 					# left-right distance from center line to leg plane
		self.LEG_L  = 0.32
		self.ABDUCTION_OFFSET = 0				# distance from abduction axis to leg

		# Robot inertia params
		self.MASS = 8.0 # kg

		(self.L,self.W,self.T) = (0.66, 0.176, 0.092)
		Ix = self.MASS/12 * (self.W**2 + self.T**2)
		Iy = self.MASS/12 * (self.L**2 + self.T**2)
		Iz = self.MASS/12 * (self.L**2 + self.W**2)
		self.INERTIA = np.zeros((3,3))
		self.INERTIA[0,0] = Ix
		self.INERTIA[1,1] = Iy
		self.INERTIA[2,2] = Iz

# Software stuff
class QPConfig:
	def __init__(self):
		self.ALPHA 	= 1e-3 	# penalty on the 2norm of foot forces
		self.GAMMA 	= 10 	# scale factor to penalize deviations in angular acceleration compared to linear accelerations
		self.MU 	= 1.0	# friction coefficient

class SwingControllerConfig:
	def __init__(self):
		self.STEP_HEIGHT 	= 0.08 # [m]
		self.KP				= 500 # [torque or force / m]

class GaitPlannerConfig:
	def __init__(self):
		self.STEP_LENGTH 	= 0.20 # [m]
		self.D 				= 0.6 # [s]

QP_CONFIG = QPConfig()
WOOFER_CONFIG = WooferConfig()
SWING_CONTROLLER_CONFIG = SwingControllerConfig()
GAIT_PLANNER_CONFIG = GaitPlannerConfig()