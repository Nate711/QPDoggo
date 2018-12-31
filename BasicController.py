import numpy as np
def PropController(qpos,refpos,kp):
	"""
	Computes the output for a proportional controller

	Note: the damping for a PD controller should be added to the joint in the xml file to increase
	the stability of the simulation
	"""
	return kp*(refpos-qpos)