import numpy as np

class WooferConfig:
	def __init__(self):
		# Robot joint limits
		self.MAX_JOINT_TORQUE = 12
		self.MAX_LEG_FORCE = 133
		self.REVOLUTE_RANGE = 3
		self.PRISMATIC_RANGE = 0.18

		# Robot geometry
		self.LEG_FB = 0.23					# front-back distance from center line to leg axis
		self.LEG_LR = 0.175 					# left-right distance from center line to leg plane
		self.LEG_L  = 0.32
		self.ABDUCTION_OFFSET = 0				# distance from abduction axis to leg

		self.FOOT_RADIUS = 0.02

		# Robot inertia params
		self.MASS = 6.135 # kg

		(self.L,self.W,self.T) = (0.66, 0.176, 0.092)
		Ix = self.MASS/12 * (self.W**2 + self.T**2)
		Iy = self.MASS/12 * (self.L**2 + self.T**2)
		Iz = self.MASS/12 * (self.L**2 + self.W**2)
		self.INERTIA = np.zeros((3,3))
		self.INERTIA[0,0] = Ix
		self.INERTIA[1,1] = Iy
		self.INERTIA[2,2] = Iz

		self.JOINT_NOISE 	= 0.5	# Nm, 1 sigma of gaussian noise
		self.LATENCY 		= 2		# ms of sense->control latency
		self.UPDATE_PERIOD	= 2		# ms between control updates

class EnvironmentConfig:
	def __init__(self):
		self.MU 			= 1.5 	# coeff friction
		self.SIM_STEPS 		= 10000 # simulation steps to take
		self.DT 			= 0.001 # timestep [s]

# Software stuff
class QPConfig:
	def __init__(self):
		self.ALPHA 	= 1e-3 	# penalty on the 2norm of foot forces
		self.BETA	= 1e-1	# penalty coefficient for the difference in foot forces between time steps
		self.GAMMA 	= 200 	# scale factor to penalize deviations in angular acceleration compared to linear accelerations
		self.MU 	= 1.0	# friction coefficient

class SwingControllerConfig:
	def __init__(self):
		self.STEP_HEIGHT 	= 0.08 # [m]
		self.KP				= np.array([400,400,2000]) # [Nm/rad, Nm/rad, N/m]

class GaitPlannerConfig:
	def __init__(self):
		self.STEP_LENGTH 	= 0.2 # [m]
		self.D 				= 0.6 # [s]

WOOFER_CONFIG = WooferConfig()
QP_CONFIG = QPConfig()
SWING_CONTROLLER_CONFIG = SwingControllerConfig()
GAIT_PLANNER_CONFIG = GaitPlannerConfig()
ENVIRONMENT_CONFIG = EnvironmentConfig()
