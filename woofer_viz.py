"""
Overview:
Applies joint-space PD controllers

Note: Free joints have 7 numbers: 3D position followed by 4D unit quaternion.


TODO: make inverse jacob
"""
from mujoco_py import load_model_from_path, MjSim, MjViewer, functions
import numpy as np
np.set_printoptions(suppress=True)
import math
import woofer_xml_parser, woofer_qp, rotations
from math_utils import CrossProductMatrix

woofer_leg_fb = 0.23					# front-back distance from center line to leg axis
woofer_leg_lr = 0.175 					# left-right distance from center line to leg plane
woofer_leg_l  = 0.3
woofer_abduction_offset = 0

def PropController(qpos,refpos,kp):
	"""
	Computes the output for a proportional controller

	Note: the damping for a PD controller should be added to the joint in the xml file to increase
	the stability of the simulation
	"""
	return kp*(refpos-qpos)

def LegForwardKinematics(qpos):
	"""
	Gives the cartesian coordinates of the four feet

	qpos: Vector of generalized coordinates of the robot with entries:
		[0:3] 	= x,y,z
		[4:7] 	= unit quaternion orientation
		[7:10]	= ab/ad, forward/back, radial for front right leg
		[10:13]	= above for front left leg
		[13:16] = above for back right leg
		[16:19] = above for back left leg
	"""
	

	quat_orientation = qpos[3:7]
	joints = qpos[7:]

	def LegFK(abad, for_back, radial, handedness):
		hands = {"left":1, "right":-1}
		offset = hands[handedness]*(woofer_abduction_offset)
		leg_unrotated = np.array([0, offset, -woofer_leg_l + radial])
		R = rotations.euler2mat([abad, for_back, 0])	# rotation matrix for abduction, then forward back
		foot = np.dot(R,leg_unrotated)
		return foot
	
	# Get foot locations in local body coordinates
	# Right-handedness of frame means y is positive in the LEFT direction
	foot_fr = LegFK(joints[0], joints[1], joints[2], "right") + np.array([woofer_leg_fb, -woofer_leg_lr, 0])
	foot_fl = LegFK(joints[3], joints[4], joints[5], "left")  + np.array([woofer_leg_fb, woofer_leg_lr, 0])
	foot_br = LegFK(joints[6], joints[7], joints[8], "right") + np.array([-woofer_leg_fb, -woofer_leg_lr, 0])
	foot_bl = LegFK(joints[9], joints[10],joints[11],"left")  + np.array([-woofer_leg_fb, woofer_leg_lr, 0])

	# Transform into world coordinates (centered around robot COM)
	feet_col_stack = np.column_stack((foot_fr, foot_fl, foot_br, foot_bl))
	feet_world = np.dot(rotations.quat2mat(quat_orientation), feet_col_stack)
	return feet_world.T.reshape(12)

def LegJacobian(beta, theta, r, abaduction_offset = 0):
	"""
	Gives the jacobian (in the body frame) of a leg given the joint values

	beta	: ab/aduction angle
	theta	: forward/back angle
	r 		: radius of leg

	return	: Jacobian (J) where (x,y,z).T = J * (betad, thetad, rd).T
	"""

	i = np.array([1,0,0])
	j = np.array([0,1,0])
	k = np.array([0,0,1])

	# leg pointing straight down
	unrotated = np.array([0,abaduction_offset,-woofer_leg_l + r]) 	
	# vector from leg hub to foot in body coordinates
	p = np.dot(rotations.euler2mat([beta, theta, 0]), unrotated) 

	# directions of positive joint movement
	theta_axis 	= np.dot(rotations.euler2mat([beta, 0, 0]), j) 
	beta_axis 	= i
	radial_axis = k

	dpdbeta 	= np.cross(beta_axis, p)
	dpdtheta 	= np.cross(theta_axis, p)
	dpdr 		= np.dot(rotations.euler2mat([beta,theta,0]), radial_axis)

	return np.column_stack([dpdbeta, dpdtheta, dpdr])

def FeetContacts(_sim):
	contacts = np.zeros(4)
	for i in range(_sim.data.ncon):
		contact = _sim.data.contact[i]
		c1 = _sim.model.geom_id2name(contact.geom1)
		c2 = _sim.model.geom_id2name(contact.geom2)

		# print('contact', i)
		# print('dist', contact.dist)

		# print('geom1', contact.geom1, c1)
		# print('geom2', contact.geom2, c2)
		
		if c1 == 'floor':
			if c2 == 'fr':
				contacts[0] = 1
			if c2 == 'fl':
				contacts[1] = 1
			if c2 == 'br':
				contacts[2] = 1
			if c2 == 'bl':
				contacts[3] = 1
	return contacts



#### INITIALIZE MUJOCO ####
model = load_model_from_path("woofer_out.xml")
print(model)
sim = MjSim(model)
viewer = MjViewer(sim)

# abad_inds = np.array([0,3,6,9])
# fb_inds = np.array([1,4,7,10])
# radial_inds = np.array([2,5,8,11])


max_gen_torques = np.zeros(12)
max_forces = np.zeros(12)

# Robot params
max_joint_torque = 8
max_ext_force = 133
woofer_mass = 7.50 # kg

(w,l,t) = (0.176, 0.66, 0.92)
Ix = woofer_mass/12 * (w**2 + t**2)
Iy = woofer_mass/12 * (l**2 + t**2)
Iz = woofer_mass/12 * (l**2 + w**2)
woofer_inertia = np.zeros((3,3))
woofer_inertia[0,0] = Ix
woofer_inertia[1,1] = Iy
woofer_inertia[2,2] = Iz

# Joint space gait params
angular_amp = 0.25
extension_amp = 0.1
kp_joint = 50
kp_ext = 400
freq = 0.5


# Simulation params
timestep = 0.001
timespan = 10

for i in range(int(timespan/timestep)):
	####### EXTRACT JOINT POSITIONS ######
	qpos = sim.data.qpos
	qvel = sim.data.qvel

	# sim.data.qpos[0:3] = np.array([0,0,0.6])
	# sim.data.qpos[3:7] = np.array([1,0,0,0])

	qpos_joints = qpos[7:]
	qvel_joints = qvel[6:]


	# BODY ORIENTATION #
	xyz 				= qpos[0:3]
	orientation_quat 	= qpos[3:7]	
	rotmat 				= rotations.quat2mat(orientation_quat)
	rpy 				= rotations.mat2euler(rotmat)

	v_xyz 				= qvel[0:3]
	angular_vel 		= qvel[3:6]

	# Foot Kinematics
	feet_locations = LegForwardKinematics(qpos)

	# Feet contact
	contacts = FeetContacts(sim)

	######## QP BALANCE ########
	time 		= 	i * timestep
	freq 		= 	0.5
	phase 		= 	time * 2*math.pi*freq

	o_ref 		= 	np.array([	math.sin(phase)*0.00,
								math.cos(phase)*0.00,
								0.25])
	rpy_ref		= 	np.array([	math.sin(phase)*15*math.pi/180.0,
								math.cos(phase)*15*math.pi/180.0,
								0])

	# Whole body PID
	wn_cart		= 	20
	kp_cart 	= 	wn_cart**2 	# wn^2
	kd_cart		= 	2*wn_cart	# 2wn*zeta
	a_xyz 		= 	PropController(	xyz,	o_ref,	kp_cart) + \
				  	PropController(	v_xyz, 	0.0, 	kd_cart)
	a_xyz	    += 	np.array([0,0,9.81])
	f_xyz 		= 	woofer_mass * a_xyz
	
	wn_ang 		= 	20
	kp_ang 		= 	wn_ang**2
	kd_ang		= 	2*wn_ang
	a_rpy 		= 	PropController(	rpy,		rpy_ref,kp_ang) + \
					PropController(	angular_vel,0.0, 	kd_ang)
	tau_rpy		= 	np.dot(woofer_inertia, a_rpy)

	ref_wrench 	= np.concatenate([f_xyz, tau_rpy])

	feet_forces = woofer_qp.SolveFeetForces(feet_locations, contacts, ref_wrench, mu = 1.0, alpha = 0.005, verbose=0)

	# Transform global foot forces into generalized joint torques
	joint_torques = np.zeros(12)
	for f in range(4):
		foot_force_world = feet_forces[f*3 : f*3 + 3]

		# from world to body, the negative sign make it the force exerted by the body
		foot_force_body = - np.dot(rotmat.T, foot_force_world)
		(beta,theta,r) = tuple(qpos_joints[f*3 : f*3 + 3])
		joint_torques[f*3 : f*3 + 3] = np.dot(LegJacobian(beta, theta, r).T, foot_force_body)
	
	sim.data.ctrl[:] = joint_torques

	max_gen_torques = np.maximum(max_gen_torques, np.abs(joint_torques))
	max_forces 		= np.maximum(max_forces, np.abs(feet_forces))

	# ######## TROT GAIT #########
	# kp = np.concatenate([kp_ext*np.ones(4), kp_joint*np.ones(8)])

	# time = i * timestep
	# phase = time * 2*math.pi*freq

	# phase0 = math.sin(phase) 				* angular_amp
	# phase1 = math.sin(phase + math.pi) 	* angular_amp
	# phase2 = math.sin(phase + math.pi/2) 	* extension_amp
	# phase3 = math.sin(phase + 3*math.pi/2) * extension_amp

	# # 4 extension, 4 ab/ad, 4 forward/back
	# refpos = np.array([	phase3, 0, phase0, 
	# 						phase2, 0, phase1,
	# 						phase2, 0, phase1,
	# 						phase3, 0, phase0])

	
	# # Override the gait and hold the legs still for debugging purposes
	# # kp 		= np.array([kp_joint, kp_joint, kp_ext]*4)
	# # refpos 	= np.array([0, math.pi/2, math.pi/4]*4)

	# ######## JOINT PID ##########

	# pid_output = PropController(qpos_joints, refpos, kp)
	# pid_output = np.clip(pid_output, [-max_joint_torque, -max_joint_torque, -max_ext_force]*4, [max_joint_torque, max_joint_torque, max_ext_force]*4)

	# max_forces = np.maximum(max_forces, np.abs(pid_output))

	# sim.data.ctrl[:] = pid_output

	####### SIM and RENDER ######

	if i % 50 == 0:
		# input("Press Enter to continue...")
		print("Frame: %s"%i)
		print("Cartesian: %s"%xyz)
		# print("Rotation matrix: %s"%rotmat)
		print("Euler angles: %s"%rpy)
		# print("Control outs: %s"%pid_output)
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
