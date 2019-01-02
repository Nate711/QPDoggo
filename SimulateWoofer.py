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
from JointSpaceController 	import JointSpaceController, TrotPDController
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
qp_trot_controller = TrotPDController(	jsp=JointSpaceController(12,133,30,300),
										freq = 2.0,
										angular_amp=0, extension_amp=0.1)
# Initialize Trot Controller
trot_controller = TrotPDController(	jsp=JointSpaceController(12,133,50,300),
									freq=1.5,
									angular_amp = 0.15, 
									extension_amp = 0.1)

# Simulation params
timestep = 0.001
timespan = 3
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
smooth_contacts_history = np.zeros((4,i_range))

contacts_smooth = np.ones(4)
contacts_alpha 	= 0.8

for i in range(i_range):
	xyz 			= WooferDynamics.position(sim)
	v_xyz 			= WooferDynamics.velocity(sim)
	quat_orien 		= WooferDynamics.orientation(sim)
	ang_vel 		= WooferDynamics.angular_velocity(sim)
	joints 			= WooferDynamics.joints(sim)
	rpy 			= rotations.quat2euler(quat_orien)

	t 	= i * timestep

	if USE_QP:
		# sim.data.qpos[0:3] = np.array([0,0,0.5])
		# sim.data.qpos[3:7] = np.array([1,0,0,0])

		######## QP BALANCE ########

		### Generate reference trajectory ###
		freq 		= 	1.0
		phase 		= 	t * 2*math.pi*freq
		o_ref 		= 	np.array([	math.sin(phase)*0.00,
									math.cos(phase)*0.0,
									math.sin(phase)*0.0 + 0.32])
		rpy_ref		= 	np.array([	math.sin(phase)*5*math.pi/180.0,
									math.cos(phase)*5*math.pi/180.0+0*math.pi/180,
									math.sin(phase)*0*math.pi/180.0])

		### Calculate contacts ###
		contacts 		= WooferDynamics.FeetContacts(sim)
		contacts_smooth = contacts_alpha*contacts_smooth + (1-contacts_alpha)*contacts
		# Using the smoothed contacts makes the slipping worse
		# contacts 		= 1*(contacts_smooth>0.5)

		### Calculate foot locations ###
		feet_locations 	= WooferDynamics.LegForwardKinematics(quat_orien, joints)

		### Use trot controller to schedule swing phase feet ###
		(pd_torques,phase, refpos) = qp_trot_controller.Update(joints, t)

		### Use dummy controller to move alternate legs upwards
		dummy_controller = JointSpaceController(12,133,30,100)
		(pd_torques) = dummy_controller.Update([0,0,0.05]*4, joints)

		### Calculate the feet that are in active stance
		active_feet = (refpos[[2,5,8,11]] <= 0.02)*1
		# active_feet = np.logical_or(active_feet, contacts)
		# disable trot controller
		# active_feet = contacts
		# override active feet for debug
		# active_feet = np.array([1,1,1,1])
		# causes more slip than if using actual contacts
		# active_feet = np.array([1,0,0,1])
		active_feet_exp = active_feet[[0,0,0,1,1,1,2,2,2,3,3,3]]

		### Solve for torques ###
		(qp_torques, foot_forces, ref_wrench) = qp_controller.Update(	(xyz, v_xyz, quat_orien, ang_vel, joints), 
																	feet_locations, 
																	active_feet, 
																	o_ref, 
																	rpy_ref)
		# override torque for debugging
		# torques = qp_torques

		torques = active_feet_exp*qp_torques + (1-active_feet_exp)*pd_torques
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
	smooth_contacts_history[:,i] = contacts_smooth

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

np.savez('SimulationData',	th=torque_history, 
							fh=force_history, 
							rwh=ref_wrench_history, 
							ch=contacts_history, 
							afh=active_feet_history,
							sch=smooth_contacts_history)

