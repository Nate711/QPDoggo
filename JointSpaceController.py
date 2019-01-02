import numpy as np
from BasicController import PropController
import math

class JointSpaceController:
	def __init__(self, max_revolute_torque, max_prismatic_force):
		self.max_joint_torque 	= max_revolute_torque
		self.max_ext_force 		= max_prismatic_force
		self.max_forces = np.zeros(12)
		self.kp_joint = 0
		self.kp_ext = 0
		self.freq = 0
		self.angular_amp = 0
		self.extension_amp = 0

	def InitTrot(self, freq = 2, angular_amp = 0, extension_amp = 0, kp_joint = 60, kp_ext = 400):
		self.angular_amp 	= angular_amp
		self.extension_amp 	= extension_amp
		self.kp_joint 		= kp_joint
		self.kp_ext 		= kp_ext
		self.freq 			= freq

	def Update(self, qpos_joints, t):
		"""
		Joint Space PD to control a trot
		"""

		if self.freq == 0:
			return np.nan

		###### Generate gait signals #####
		kp 		= np.array([self.kp_joint*2, self.kp_joint*2, self.kp_ext]*4)
		phase 	= t * self.freq * 2*math.pi

		phase0 	= math.sin(phase) 				* self.angular_amp
		phase1 	= math.sin(phase + math.pi) 	* self.angular_amp
		phase2 	= math.sin(phase + math.pi/2) 	* self.extension_amp
		phase3 	= math.sin(phase + 3*math.pi/2) * self.extension_amp

		# The format is 4*[abad, theta, radial]
		refpos = np.array([	phase0*0.0, phase0, phase3, 
							phase1*0.0, phase1, phase2,
							phase1*0.0, phase1, phase2,
							phase0*0.0, phase0, phase3])


		# # Override the gait and hold the legs still for debugging purposes
		# # kp 		= np.array([kp_joint, kp_joint, kp_ext]*4)
		# # refpos 	= np.array([0, math.pi/2, math.pi/4]*4)

		######### JOINT PID ##########
		pid_output = PropController(qpos_joints, refpos, kp)
		pid_output = np.clip(pid_output, 	[-self.max_joint_torque, -self.max_joint_torque, -self.max_ext_force]*4, 
											[ self.max_joint_torque,  self.max_joint_torque,  self.max_ext_force]*4)

		return pid_output, phase, refpos
