"""
Overview:
Applies joint-space PD controllers

Note: 
1) Free joints have 7 numbers: 3D position followed by 4D unit quaternion.
2) Setting too high gains for the whole-body PID -> large desired accelerations -> 
higher foot forces -> chattering feet

TODO: 
1) change horizontal force limits into torque limits
2) penalize changes between forces

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

## Set controller
USE_QP = True
USE_JS = not USE_QP

# Public modules
from mujoco_py import load_model_from_path, MjSim, MjViewer, functions
import numpy as np
import math, time
import rotations

# Custom modules
from MathUtils 				import CrossProductMatrix, RunningMax
import WooferDynamics 		
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

# Initialize QP Balance Controller
qp_controller = QPBalanceController()
qp_controller.InitQPBalanceController(WooferDynamics.woofer_mass, WooferDynamics.woofer_inertia)
qp_trot_controller = JointSpaceController(max_revolute_torque = 12, max_prismatic_force = 133)
qp_trot_controller.InitTrot(	freq			= 2.0, 
								angular_amp		= 0.0, 
								extension_amp	= 0.1, 
								kp_joint		= 80, 
								kp_ext			= 200)


# Initialize Trot Controller
trot_controller = JointSpaceController(	max_revolute_torque 	= 12, 
										max_prismatic_force 	= 133)
trot_controller.InitTrot(	freq			= 2.5, 
							angular_amp		= 0.25, 
							extension_amp	= 0.1, 
							kp_joint		= 80, 
							kp_ext			= 500)

# Simulation params
timestep = 0.001
timespan = 2
i_range = int(timespan/timestep)

# Initialize variables to track maximum torques and forces
max_torques = RunningMax(12)
max_forces = RunningMax(12)
ref_wrench = 0
foot_forces = 0
contacts = 0
feet_locations = 0
active_feet = 0

torque_history = np.zeros((12,i_range))
force_history = np.zeros((12,i_range))
ref_wrench_history = np.zeros((6,i_range))
contacts_history = np.zeros((4,i_range))
active_feet_history = np.zeros((4,i_range))


for i in range(i_range):

	xyz 			= WooferDynamics.position(sim)
	v_xyz 			= WooferDynamics.velocity(sim)
	quat_orien 		= WooferDynamics.orientation(sim)
	ang_vel 		= WooferDynamics.angular_velocity(sim)
	joints 			= WooferDynamics.joints(sim)


	rpy = rotations.quat2euler(quat_orien)

	t 	= i * timestep

	if USE_QP:
		# sim.data.qpos[0:3] = np.array([0,0,0.5])
		# sim.data.qpos[3:7] = np.array([1,0,0,0])
		######## QP BALANCE ########
		
		freq 		= 	2
		phase 		= 	t * 2*math.pi*freq
		o_ref 		= 	np.array([	math.sin(phase)*0.00,
									math.cos(phase)*0.0,
									math.sin(phase)*0.0 + 0.32])
		rpy_ref		= 	np.array([	math.sin(phase)*0*math.pi/180.0,
									math.cos(phase)*10*math.pi/180.0+0*math.pi/180,
									math.sin(phase)*10*math.pi/180.0])

		contacts 		= WooferDynamics.FeetContacts(sim)
		feet_locations 	= WooferDynamics.LegForwardKinematics(quat_orien, joints)


		### Use trot controller to schedule swing phase feet ###
		(pd_torques,phase, refpos) = qp_trot_controller.Update(joints, t)

		active_feet = (refpos[[2,5,8,11]] <= 0)*1

		# override active feet for debug
		# active_feet = np.array([1,1,1,1])
		active_feet = contacts

		active_feet_exp = active_feet[[0,0,0,1,1,1,2,2,2,3,3,3]]


		(qp_torques, foot_forces, ref_wrench) = qp_controller.Update(	(xyz, v_xyz, quat_orien, ang_vel, joints), 
																	feet_locations, 
																	active_feet, 
																	o_ref, 
																	rpy_ref)
		# override torque for debugging
		torques = qp_torques

		# torques = active_feet_exp*qp_torques + (1-active_feet_exp)*pd_torques


		sim.data.ctrl[:] = torques

		max_forces.Update(foot_forces)
		max_torques.Update(torques)

	elif USE_JS:

		######## Trot Controller ######

		(torques,refpos,phase) = trot_controller.Update(joints, t)
		sim.data.ctrl[:] = torques

		max_torques.Update(torques)

	### Update histories ###
	torque_history[:,i] = torques
	force_history[:,i] = foot_forces
	ref_wrench_history[:,i] = ref_wrench
	contacts_history[:,i] = contacts
	active_feet_history[:,i] = active_feet

	####### SIM and RENDER ######

	if i % 50 == 0:

		print("Frame: %s"%i)
		print("Cartesian: %s"%xyz)
		print("Euler angles: %s"%rpy)
		print("Max gen. torques: %s"%max_torques.CurrentMax())
		print("Max forces: %s"%max_forces.CurrentMax())
		print("ref wrench: %s"%ref_wrench)
		print("feet locations: %s"%feet_locations)
		print("contacts: %s"%contacts)
		print("qp feet forces: %s"%foot_forces)
		print("joint torques: %s"%torques)
		print('\n')

	sim.step()
	viewer.render()

np.savez('SimulationData',th=torque_history, fh=force_history, rwh=ref_wrench_history, ch=contacts_history, afh=active_feet_history)

