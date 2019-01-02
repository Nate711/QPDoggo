import numpy as np
from BasicController import PropController
import math

class JointSpaceController:
	def __init__(self, max_revolute_torque, max_prismatic_force, kp_joint, kp_ext):
		self.max_joint_torque 	= max_revolute_torque
		self.max_ext_force 		= max_prismatic_force
		self.max_forces = np.zeros(12)
		self.kp_joint = kp_joint
		self.kp_ext = kp_ext

	def Update(self, refpos, qpos_joints):
		"""
		Joint Space PD to control a trot
		"""

		###### Generate gait signals #####
		kp 		= np.array([self.kp_joint*2, self.kp_joint*2, self.kp_ext]*4)

		# # Override the gait and hold the legs still for debugging purposes
		# # kp 		= np.array([kp_joint, kp_joint, kp_ext]*4)
		# # refpos 	= np.array([0, math.pi/2, math.pi/4]*4)

		######### JOINT PID ##########
		pid_output = PropController(qpos_joints, refpos, kp)
		pid_output = np.clip(pid_output, 	[-self.max_joint_torque, -self.max_joint_torque, -self.max_ext_force]*4, 
											[ self.max_joint_torque,  self.max_joint_torque,  self.max_ext_force]*4)

		return pid_output

class TrotPDController:
	def __init__(self, jsp=JointSpaceController(12,133,50,500), freq=2, angular_amp=0, extension_amp=0):
		self.jsp 			= jsp
		self.freq 			= freq
		self.angular_amp 	= angular_amp
		self.extension_amp 	= extension_amp

	def Update(self, qpos_joints, t):
		"""
		Joint Space PD to control a trot
		"""

		###### Generate gait signals #####
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

		pid_output = self.jsp.Update(refpos, qpos_joints)

		return pid_output, phase, refpos


