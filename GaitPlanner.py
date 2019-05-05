import numpy as np
from math import pi, cos, sin

class GaitPlanner:
	"""
	Takes desired velocity and current state and outputs foot placements, CoM trajectory, and gait phase.
	"""
	def update(self, state, contacts, t):
		"""
		Output CoM trajectory and foot placements
		"""
		raise NotImplementedError
		return (None, None, None, None) # Foot placements, CoM ref position, body ref orientation (euler), active_feet, phase

class StandingPlanner(GaitPlanner):
	"""
	Stands still!
	"""
	def update(self, state, contacts, t):
		freq 		= 	1.0
		phase 		= 	t * 2 * pi * freq
		p_ref 		= 	np.array([	sin(phase)*0.00,
									cos(phase)*0.00,
									sin(phase)*0.00 + 0.32])
		rpy_ref		= 	np.array([	sin(phase)*15*pi/180.0,
									cos(phase)*25*pi/180.0 + 0*pi/180,
									sin(phase)*0*pi/180.0])

		# Boolean representation of which feet the QP controller treats as in contact with the ground. 
		# A 1 in the ith slot means that the ith leg is in contact with the ground
		active_feet = np.array([1,1,1,1])

		return (None, p_ref, rpy_ref, active_feet, phase)
