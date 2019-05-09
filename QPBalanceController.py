from BasicController import PropController
import rotations
from WooferDynamics import *
from WooferQP import SolveFeetForces

class QPBalanceController:
	def __init__(self):
		self.max_forces = 0
		self.max_gen_torques = 0

	def Update(self,coordinates,
					feet_locations, 
					active_feet, 
					o_ref, 
					rpy_ref,
					f_prev, 
					woof_config,
					qp_config,
					verbose = 0):
		"""
		Run the QP Balance controller
		"""

		(xyz, v_xyz, orientation, w_rpy, qpos_joints) = coordinates

		########## Generate desired body accelerations ##########
		rpy = rotations.quat2euler(orientation)
		rotmat = rotations.quat2mat(orientation)

		# Foot Kinematics
		feet_locations = LegForwardKinematics(orientation, qpos_joints)

		# Feet contact
		contacts = active_feet

		# Use a whole-body PD controller to generate desired body moments
		### Cartesian force ###
		wn_cart		= 20			# desired natural frequency
		zeta_cart	= 0.8			# desired damping ratio
		kp_cart 	= wn_cart**2 	# wn^2
		kd_cart		= 2*wn_cart*zeta_cart		# 2wn*zeta
		a_xyz 		= PropController(	xyz,	o_ref,	kp_cart) + \
					  PropController(	v_xyz, 	0, 		kd_cart)
		a_xyz	   += np.array([0,0,9.81])
		f_xyz 		= woof_config.MASS * a_xyz
		
		### Angular moment ###
		wn_ang 		= 20
		zeta_ang 	= 0.8
		kp_ang 		= wn_ang**2
		kd_ang		= 2*wn_ang*zeta_ang
		a_rpy 		= PropController(	rpy,	rpy_ref,	kp_ang) + \
					  PropController(	w_rpy,	0.0, 		kd_ang)			  
		tau_rpy		= np.matmul(woof_config.INERTIA, a_rpy)

		### Combined force and moment ###
		ref_wrench 	= np.concatenate([f_xyz, tau_rpy])


		########## Solve for foot forces #########

		# Find foot forces to achieve the desired body moments
		feet_forces = SolveFeetForces(	feet_locations, 
										contacts, 
										ref_wrench, 
										f_prev, 
										mu = qp_config.MU, 
										alpha = qp_config.ALPHA, 
										beta = qp_config.BETA,
										gamma = qp_config.GAMMA, 
										verbose = verbose)

		joint_torques = np.zeros(12)
		
		for f in range(4):
			# Extract the foot force vector for foot i
			foot_force_world = feet_forces[f*3 : f*3 + 3]

			# Transform from world to body frames, 
			# The negative sign makes it the force exerted by the body
			foot_force_body 				= -np.dot(rotmat.T, foot_force_world)
			(beta,theta,r) 					= tuple(qpos_joints[f*3 : f*3 + 3])
			
			# Transform from body frame forces into joint torques
			joint_torques[f*3 : f*3 + 3] 	= np.dot(LegJacobian(beta, theta, r).T, foot_force_body)
	

		return (joint_torques, feet_forces, ref_wrench)