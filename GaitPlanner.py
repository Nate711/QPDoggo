import numpy as np
from math import pi, cos, sin

"""
Note:
Order of legs is: FR, FL, BR, BL
"""

class GaitPlanner:
	"""
	Takes desired velocity and current state and outputs foot placements, CoM trajectory, and gait phase.
	"""
	def update(self, state, contacts, t):
		"""
		Output CoM trajectory and foot placements
		"""
		raise NotImplementedError
		return (None, None, None, None, None, None) # Foot placements, CoM ref position, body ref orientation (euler), active_feet, phase

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

		return (None, None, p_ref, rpy_ref, active_feet, phase)

class StepPlanner(GaitPlanner):
	"""
	Plans two walking-trot steps forward. During the first half of the stride, 
	the front-left and back-right legs are planned to be in stance. During the
	second half of the stride, the front-right and back-left legs are planned
	to be in stance. After the full stride, all legs are planned for stance.
	"""
	
	def update(self, state, contacts, t, woof_config, gait_config):
		"""
		STEP_LENGTH: step length in meters
		D: duration of the total two-step move in seconds

		phase: 0->0.5 & step_phase 0->1: moving FR and BL forward
		phase: 0.5 -> 1 & step_phase 0->1: moving FL and BR forward

		Note that the foot placements vector is only used as reference positions for the swing controller
		which means that the reference foot placements are meaningless for legs in contact
		"""

		# stride starts at phase = 0 and ends at phase = 1
		phase = t/gait_config.D

		foot_locs = np.array([ 	 woof_config.LEG_FB, -woof_config.LEG_LR, 0,
								 woof_config.LEG_FB,  woof_config.LEG_LR, 0,
								-woof_config.LEG_FB, -woof_config.LEG_LR, 0,
								-woof_config.LEG_FB,  woof_config.LEG_LR, 0])
		p_foot_locs = foot_locs

		active_feet = np.array([1,1,1,1])

		step_phase = 0

		if phase >= 0 and phase < 0.5:
			# Move FR and BL forward
			foot_locs = np.array([ 	 woof_config.LEG_FB + gait_config.STEP_LENGTH,	-woof_config.LEG_LR, 0,
									 woof_config.LEG_FB, 				 	 		 woof_config.LEG_LR, 0,
								 	-woof_config.LEG_FB, 							-woof_config.LEG_LR, 0,
									-woof_config.LEG_FB + gait_config.STEP_LENGTH,   woof_config.LEG_LR, 0])
			
			p_foot_locs = np.array([ woof_config.LEG_FB, -woof_config.LEG_LR, 0,
									 woof_config.LEG_FB,  woof_config.LEG_LR, 0,
								 	-woof_config.LEG_FB, -woof_config.LEG_LR, 0,
									-woof_config.LEG_FB,  woof_config.LEG_LR, 0])
			active_feet = np.array([0,1,1,0])
			
			step_phase = phase * 2.0
			
		elif phase >= 0.5 and phase < 1.0:
			# Move FL and BR forward
			foot_locs = np.array([	 woof_config.LEG_FB + gait_config.STEP_LENGTH,	-woof_config.LEG_LR, 0,
									 woof_config.LEG_FB + gait_config.STEP_LENGTH,	 woof_config.LEG_LR, 0,
								 	-woof_config.LEG_FB + gait_config.STEP_LENGTH, 	-woof_config.LEG_LR, 0,
									-woof_config.LEG_FB + gait_config.STEP_LENGTH,	 woof_config.LEG_LR, 0])
			p_foot_locs = np.array([ woof_config.LEG_FB + gait_config.STEP_LENGTH, 	-woof_config.LEG_LR, 0,
									 woof_config.LEG_FB, 				 	 		 woof_config.LEG_LR, 0,
									-woof_config.LEG_FB, 							-woof_config.LEG_LR, 0,
									-woof_config.LEG_FB + gait_config.STEP_LENGTH,	 woof_config.LEG_LR, 0])
			active_feet = np.array([1,0,0,1])

			step_phase = (phase - 0.5) * 2.0

		elif phase >= 1.0:
			# All feet are forward
			foot_locs = np.array([ 	 woof_config.LEG_FB + gait_config.STEP_LENGTH, -woof_config.LEG_LR, 0,
									 woof_config.LEG_FB + gait_config.STEP_LENGTH,  woof_config.LEG_LR, 0,
									-woof_config.LEG_FB + gait_config.STEP_LENGTH, -woof_config.LEG_LR, 0,
									-woof_config.LEG_FB + gait_config.STEP_LENGTH,  woof_config.LEG_LR, 0])
			p_foot_locs = foot_locs

			active_feet = np.array([1,1,1,1])

			step_phase = 0.0

		# Want body to be level during the step
		rpy_ref = np.array([0,0,0])

		# Want the body to move forward one step length
		CoM_x = gait_config.STEP_LENGTH * np.clip(phase,0,1) 
		p_ref = np.array([CoM_x, 0, woof_config.LEG_L])

		# print("foot placements:")
		# print(foot_locs)

		return (foot_locs, p_foot_locs, p_ref, rpy_ref, active_feet, phase, step_phase)
