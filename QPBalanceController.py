from BasicController import PropController
import rotations
from WooferDynamics import *
from WooferQP import SolveFeetForces

class QPBalanceController:
	def __init__(self):
		self.max_forces = 0
		self.max_gen_torques = 0

	def InitQPBalanceController(self, woofer_mass, woofer_inertia):
		self.woofer_inertia = woofer_inertia
		self.woofer_mass = woofer_mass

	def Update(self, sim, o_ref, rpy_ref):
		########## Generate desired body accelerations ##########
		qpos = sim.data.qpos[:]
		qvel = sim.data.qvel[:]

		# BODY ORIENTATION #
		xyz 				= qpos[0:3]
		orientation_quat 	= qpos[3:7]	
		rotmat 				= rotations.quat2mat(orientation_quat)
		rpy 				= rotations.mat2euler(rotmat)

		v_xyz 				= qvel[0:3]
		angular_vel 		= qvel[3:6]

		qpos_joints = qpos[7:]

		# Foot Kinematics
		feet_locations = LegForwardKinematics(qpos)

		# Feet contact
		contacts = FeetContacts(sim)

		# Use a whole-body PD controller to generate desired body moments
		### Cartesian force ###
		wn_cart		= 20			# desired natural frequency
		kp_cart 	= wn_cart**2 	# wn^2
		kd_cart		= 2*wn_cart		# 2wn*zeta
		a_xyz 		= PropController(xyz,	o_ref,	kp_cart) + \
					  PropController(v_xyz, 0, 		kd_cart)
		a_xyz	   += np.array([0,0,9.81])
		f_xyz 		= woofer_mass * a_xyz
		
		### Angular moment ###
		wn_ang 		= 20
		kp_ang 		= wn_ang**2
		kd_ang		= 2*wn_ang
		a_rpy 		= PropController(	rpy,			rpy_ref,	kp_ang) + \
					  PropController(	angular_vel,	0.0, 		kd_ang)			  
		tau_rpy		= np.dot(woofer_inertia, a_rpy)

		### Combined force and moment ###
		ref_wrench 	= np.concatenate([f_xyz, tau_rpy])



		########## Solve for foot forces #########

		# Find foot forces to achieve the desired body moments
		feet_forces = SolveFeetForces(feet_locations, contacts, ref_wrench, mu = 1.0, alpha=1e-3, verbose=0)

		joint_torques = np.zeros(12)
		
		for f in range(4):
			# Extract the foot force vector for foot i
			foot_force_world = feet_forces[f*3 : f*3 + 3]

			# Transform from world to body frames, 
			# The negative sign makes it the force exerted by the body
			foot_force_body 				= - np.dot(rotmat.T, foot_force_world)
			(beta,theta,r) 					= tuple(qpos_joints[f*3 : f*3 + 3])
			
			# Transform from body frame forces into joint torques
			joint_torques[f*3 : f*3 + 3] 	= np.dot(LegJacobian(beta, theta, r).T, foot_force_body)
	

		self.max_gen_torques 	= np.maximum(self.max_gen_torques, 	np.abs(joint_torques))
		self.max_forces 		= np.maximum(self.max_forces, 		np.abs(feet_forces))

		return (joint_torques, self.max_forces, self.max_gen_torques)