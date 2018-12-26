"""
Overview:
Applies joint-space PD controllers

Note: Free joints have 7 numbers: 3D position followed by 4D unit quaternion.
"""

def PropController(qpos,kp,refpos):
	return kp*(refpos-qpos)

from mujoco_py import load_model_from_path, MjSim, MjViewer, functions
import numpy as np
np.set_printoptions(suppress=True)
import woofer_xml_parser
import rotations
import math

model = load_model_from_path("woofer_out.xml")
print(model)
sim = MjSim(model)
viewer = MjViewer(sim)


radial_inds = np.array([2,5,8,11])
abad_inds = np.array([0,3,6,9])
fb_inds = np.array([1,4,7,10])

# radial, then ab/ad, then extension
joint_inds = np.concatenate([radial_inds, abad_inds, fb_inds])

joint_qpos_inds = joint_inds + 7	# 7 variables for free joint position
joint_qvel_inds = joint_inds + 6	# 6 variables for free joint velocity

max_forces = np.zeros(12)

for i in range(10000):
	qpos = sim.data.qpos
	qvel = sim.data.qvel

	# sim.data.qpos[0:3] = np.array([0,0,0.6])
	# sim.data.qpos[3:7] = np.array([1,0,0,0])

	qpos_joints = qpos[joint_qpos_inds]
	qvel_joints = qvel[joint_qvel_inds]

	xyz = qpos[0:3]
	orientation_quat = qpos[3:7]	
	rotmat = rotations.quat2mat(orientation_quat)
	rpy = rotations.mat2euler(rotmat)


	kp = np.concatenate([500*np.ones(4), 50*np.ones(8)])

	time = i/1000.0
	freq = 2.0
	phase = time * 2*math.pi*freq

	angular_amp = 0.2
	extension_amp = 0.1

	phase0 = math.sin(phase) * angular_amp
	phase1 = math.sin(phase + math.pi) * angular_amp

	phase2 = math.sin(phase + math.pi/2) * extension_amp
	phase3 = math.sin(phase + 3*math.pi/2) * extension_amp

	refpos = np.array([phase3, phase2, phase2, phase3] + [0]*4 + [phase0, phase1, phase1, phase0])

	pid_output = PropController(qpos_joints,kp,refpos)
	pid_output = np.clip(pid_output, [-133]*4 + [-6]*8, [133]*4 + [6]*8)

	max_forces = np.maximum(max_forces, np.abs(pid_output))

	sim.data.ctrl[joint_inds] = pid_output

	if i % 100 == 0:
		# input("Press Enter to continue...")
		# print("Cartesian: %s"%xyz)
		# print("Rotation matrix: %s"%rotmat)
		# print("Euler angles: %s"%rpy)
		# print("Control outs: %s"%pid_output)
		print("Max: %s"%max_forces)

	sim.step()
	viewer.render()









