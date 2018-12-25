"""
Free joints have 7 numbers: 3D position followed by 4D unit quaternion.
"""
from mujoco_py import load_model_from_path, MjSim, MjViewer, functions
import numpy as np
np.set_printoptions(suppress=True)
import woofer_xml_parser
import rotations

model = load_model_from_path("woofer_out.xml")
print(model)
sim = MjSim(model)
viewer = MjViewer(sim)

for i in range(3000):
	qpos = sim.data.qpos

	xyz = qpos[0:3]
	orientation_quat = qpos[3:7]
	rotmat = rotations.quat2mat(orientation_quat)
	rpy = rotations.mat2euler(rotmat)

	if i % 100 == 0:
		# input("Press Enter to continue...")
		print("Cartesian: %s"%xyz)
		print("Rotation matrix: %s"%rotmat)
		print("Euler angles: %s"%rpy)

	radial_inds = np.array([2,5,8,11])
	abad_inds = np.array([0,3,6,9])
	fb_inds = np.array([1,4,7,10])

	sim.data.ctrl[radial_inds] = -200
	sim.data.ctrl[fb_inds] = 0.1
	sim.data.ctrl[abad_inds] = 0.1

	sim.step()
	viewer.render()





