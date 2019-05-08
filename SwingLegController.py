import numpy as np
import WooferDynamics
from math import pi, sin

class SwingLegController:
	def update(self, state, contacts, step_phase):
		raise NotImplementedError

class ZeroSwingLegController(SwingLegController):
	"""
	Placeholder class
	"""
	def update(self, state, contacts, step_phase):
		return np.zeros(12)

class PDSwingLegController(SwingLegController):
	"""
	Computes joint torques to bring the swing legs to their new locations
	"""

	def trajectory(self, state, step_phase, step_locs, p_step_locs, active_feet, swing_config):
		"""
		sin2-based parametric trajectory planner

		STEP_HEIGHT: maximum ground clearance during the foot step (in meters)
		"""

		# smooth interpolation between 0 and 1 (with respect to time)
		incrementer = (sin(step_phase * pi/2.0))**2

		# ground plane interpolation between previous step locations and new step locations
		ground_plane_foot_reference = step_locs * incrementer + (1-incrementer) * p_step_locs

		# foot heights
		foot_heights = np.zeros(12)
		foot_heights[[2,5,8,11]] = swing_config.STEP_HEIGHT * (sin(step_phase * pi))**2

		# combine ground plane interp and foot heights
		swing_foot_reference_p = ground_plane_foot_reference + foot_heights

		# print(feet_world_NED)
		# print(swing_foot_reference_p)
		# print(" ")

		return swing_foot_reference_p

	def update(self, state, step_phase, step_locs, p_step_locs, active_feet, woof_config, swing_config):
		reference_positions = self.trajectory(state, step_phase, step_locs, p_step_locs, active_feet, swing_config)

		feet_world_NED = WooferDynamics.FootLocationsWorld(state)

		errors = reference_positions - feet_world_NED
		foot_forces = swing_config.KP * errors

		leg_torques = np.zeros(12)
		for i in range(4):
			if active_feet[i] == 0:
				specific_foot_force = foot_forces[3*i:3*i+3]
				specific_leg_joints = state['j'][3*i:3*i+3]
				leg_torques[3*i:3*i+3] = WooferDynamics.FootForceToJointTorques(specific_foot_force, specific_leg_joints, state['q'], abaduction_offset = 0)

		# print(leg_torques, reference_positions)
		# print(" ")
		return leg_torques, reference_positions




