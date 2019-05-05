import WooferDynamics 		

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
	def update(self, sim):
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