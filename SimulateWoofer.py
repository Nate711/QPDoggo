"""
Overview:
Applies joint-space PD controllers

Note: Free joints have 7 numbers: 3D position followed by 4D unit quaternion.

Tuning alpha:
1Hz circle

alpha = 10
Max gen. torques: [ 2.47340165  2.4093495  31.23974859  2.48783854  2.50385205 36.68769709
  2.35802917  2.50993243 37.28364324  2.20897474  2.41542804 31.00640439]

alpha = 0.1
Max gen. torques: [ 2.36847743  1.94332827 30.43882154  1.87009629  2.00477958 35.72317222
  1.57899464  1.62752591 34.96297803  2.31617442  2.01736958 30.62738061]

alpha = 0.001
Max gen. torques: [ 2.35190582  1.92108034 30.49324448  1.88638873  2.01311611 36.11915157
  1.54473214  1.60009785 35.03724632  2.28538479  2.00016686 30.64702714]

alpha = 0.00001
Max gen. torques: [ 2.35490409  1.92427229 30.4785506   1.89554137  1.99377022 34.53321875
  1.58081161  1.57831604 34.31487551  2.51154226  2.01617002 30.79013023]

The torques might be increasing because at larger alphas, there is a larger tracking error -> bigger reference accelerations

TODO: make inverse jacob
"""

# Public modules
from mujoco_py import load_model_from_path, MjSim, MjViewer, functions
import numpy as np
import math, time
import rotations

# Custom modules
from MathUtils 				import CrossProductMatrix
from WooferDynamics 		import *
from JointSpaceController 	import JointSpaceController
from BasicController 		import PropController
from QPBalanceController 	import QPBalanceController
import WooferXMLParser

# Tell Numpy not to print in decimal format rather than scientific
np.set_printoptions(suppress=True)

#### Parse the MJCF XML model file ####
WooferXMLParser.Parse()

#### Initailize MuJoCo ####
model = load_model_from_path("woofer_out.xml")
print(model)
sim = MjSim(model)
viewer = MjViewer(sim)


# Initialize variables to track maximum torques and forces
max_gen_torques = np.zeros(12)
max_forces = np.zeros(12)

# Initialize QP Balance Controller
qp_controller = QPBalanceController()
qp_controller.InitQPBalanceController(woofer_mass, woofer_inertia)


# Initialize Trot Controller
trot_controller = JointSpaceController(	max_revolute_torque 	= 12, 
										max_prismatic_force 	= 133)
trot_controller.InitTrot(	freq			= 2.5, 
							angular_amp		= 0.25, 
							extension_amp	= 0.1, 
							kp_joint		= 80, 
							kp_ext			= 500)

# Simulation params
timestep = 0.004
timespan = 15

for i in range(int(timespan/timestep)):
	####### EXTRACT JOINT POSITIONS ######
	qpos = sim.data.qpos
	qvel = sim.data.qvel

	qpos_joints = qpos[7:]
	qvel_joints = qvel[6:]

	t 			= 	i * timestep

	######## QP BALANCE ########
	
	freq 		= 	1.5
	phase 		= 	t * 2*math.pi*freq

	o_ref 		= 	np.array([	math.sin(phase)*0.00,
								math.cos(phase)*0.00,
								math.sin(phase)*0.00 + 0.25])
	rpy_ref		= 	np.array([	math.sin(phase)*0*math.pi/180.0,
								math.cos(phase)*15*math.pi/180.0+0*math.pi/180,
								math.sin(phase)*15*math.pi/180.0])

	(sim.data.ctrl[:], max_forces, max_torques) = qp_controller.Update(sim, o_ref, rpy_ref)

	######## Trot Controller ######

	# (sim.data.ctrl[:], max_forces) = trot_controller.Update(qpos_joints, t)

	####### SIM and RENDER ######

	if i % 1 == 50:
		print("Frame: %s"%i)
		print("Cartesian: %s"%xyz)
		print("Euler angles: %s"%rpy)
		print("Max gen. torques: %s"%max_gen_torques)
		print("Max forces: %s"%max_forces)
		print("ref wrench: %s"%ref_wrench)
		print("feet locations: %s"%feet_locations)
		print("contacts: %s"%contacts)
		print("qp feet forces: %s"%feet_forces)
		print("joint torques: %s"%joint_torques)
		print('\n')

	sim.step()
	viewer.render()
