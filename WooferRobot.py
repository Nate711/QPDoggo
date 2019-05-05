import numpy as np
import math, time
import rotations

# Helper math functions
from MathUtils 				import CrossProductMatrix, RunningMax

# Provides kinematic functions among others
import WooferDynamics 		

from JointSpaceController 	import JointSpaceController, TrotPDController
from BasicController 		import PropController
from QPBalanceController 	import QPBalanceController
from StateEstimator import MuJoCoStateEstimator
from ContactEstimator import MuJoCoContactEstimator


class WooferRobot():
	"""
	This class represents the onboard Woofer software. 

	The primary input is the mujoco simulation data and the
	primary output is a set joint torques. 
	"""
	def __init__(self, state_estimator, contact_estimator, qp_controller, dt):
		"""
		Initialize object variables
		"""

		self.contact_estimator 	= contact_estimator
		self.state_estimator 	= state_estimator
		self.qp_controller 		= qp_controller # QP controller for calculating foot forces
		self.state 				= None
		self.contacts 			= None


		self.max_torques 		= RunningMax(12)
		self.max_forces 		= RunningMax(12)

		init_data_size = 1000
		self.data = {}
		self.data['torque_history'] 			= np.empty((12,init_data_size))
		self.data['force_history']				= np.empty((12,init_data_size))
		self.data['ref_wrench_history'] 		= np.empty((6,init_data_size))
		self.data['contacts_history'] 			= np.empty((4,init_data_size))
		self.data['active_feet_history'] 		= np.empty((4,init_data_size))

		self.dt = dt
		self.t = 0
		self.i = 0

	def step(self,sim):
		"""
		Get sensor data and update state estimate and contact estimate. Then calculate joint torques for locomotion.
		
		Details:
		Gait controller:
		Looks at phase variable to determine foot placements and COM trajectory
		
		QP: 
		Generates joint torques to achieve given desired CoM trajectory given stance feet

		Swing controller:
		Swing controller needs reference foot landing positions and phase
		"""

		# Update the state and contact estimators
		self.state 		= self.state_estimator.update(sim)
		self.contacts 	= self.contact_estimator.update(sim)

		################################################## TODO ##################################
		# TODO: take the gait planner out of this function
		### Generate reference trajectory ###
		freq 		= 	1.0
		phase 		= 	self.t * 2*math.pi*freq
		p_ref 		= 	np.array([	math.sin(phase)*0.00,
									math.cos(phase)*0.00,
									math.sin(phase)*0.00 + 0.32])
		rpy_ref		= 	np.array([	math.sin(phase)*15*math.pi/180.0,
									math.cos(phase)*25*math.pi/180.0+0*math.pi/180,
									math.sin(phase)*0*math.pi/180.0])

		################################################## TODO ##################################
		# TODO. Zero for now, but in the future the swing controller will provide these torques
		pd_torques = np.zeros(12)

		# Rearrange the state for the qp solver
		qp_state = (self.state['p'],self.state['p_d'],self.state['q'],self.state['w'],self.state['j'])

		# Use forward kinematics from the robot body to compute where the woofer feet are
		self.feet_locations = WooferDynamics.LegForwardKinematics(self.state['q'], self.state['j'])

		# Boolean representation of which feet the QP controller treats as in contact with the ground
		
		################################################## TODO ##################################
		# TODO: In the future the gait scheduler will set this variable
		self.active_feet = np.array([1,1,1,1])

		# Calculate foot forces using the QP solver
		(qp_torques, self.foot_forces, self.ref_wrench) = self.qp_controller.Update(qp_state, 
																					self.feet_locations, 
																					self.active_feet, 
																					p_ref, 
																					rpy_ref)
		# Expanded version of active feet
		active_feet_12 = self.active_feet[[0,0,0,1,1,1,2,2,2,3,3,3]] 

		# Mix the QP-generated torques and PD-generated torques to produce the final joint torques sent to the robot
		self.torques = active_feet_12 * qp_torques + (1 - active_feet_12) * pd_torques

		# Update our record of the maximum force/torque
		self.max_forces.Update(self.foot_forces)
		self.max_torques.Update(self.torques)

		# Log stuff
		self.log_data()

		# Step time forward
		self.t += self.dt
		self.i += 1

		return self.torques

	def log_data(self):
		"""
		Append data to logs
		""" 
		data_len = self.data['torque_history'].shape[1]
		if self.i > data_len - 1:
			self.data['torque_history'] 	= np.append(self.data['torque_history'], 		np.empty((12,1000)),axis=1)
			self.data['force_history'] 		= np.append(self.data['force_history'], 		np.empty((12,1000)),axis=1)
			self.data['ref_wrench_history'] = np.append(self.data['ref_wrench_history'], 	np.empty((6,1000)),axis=1)
			self.data['contacts_history'] 	= np.append(self.data['contacts_history'], 		np.empty((4,1000)),axis=1)
			self.data['active_feet_history']= np.append(self.data['active_feet_history'], 	np.empty((4,1000)),axis=1)

		self.max_forces.Update(self.foot_forces)
		self.max_torques.Update(self.torques)
		self.data['torque_history'][:,self.i] 		= self.torques
		self.data['force_history'][:,self.i] 		= self.foot_forces
		self.data['ref_wrench_history'][:,self.i] 	= self.ref_wrench
		self.data['contacts_history'][:,self.i] 	= self.contacts
		self.data['active_feet_history'][:,self.i] 	= self.active_feet

	def print_data(self):
		"""
		Print debug data
		"""
		print("Time: %s"			%self.t)
		print("Cartesian: %s"		%self.state['p'])
		print("Euler angles: %s"	%rotations.quat2euler(self.state['q']))
		print("Max gen. torques: %s"%self.max_torques.CurrentMax())
		print("Max forces: %s"		%self.max_forces.CurrentMax())
		print("ref wrench: %s"		%self.ref_wrench)
		print("feet locations: %s"	%self.feet_locations)
		print("contacts: %s"		%self.contacts)
		print("qp feet forces: %s"	%self.foot_forces)
		print("joint torques: %s"	%self.torques)
		print('\n')

	def save_logs(self):
		"""
		Save the log data to file
		"""
		print(self.data['torque_history'])
		np.savez('SimulationData',	th 	= self.data['torque_history'], 
									fh 	= self.data['force_history'], 
									rwh = self.data['ref_wrench_history'], 
									ch 	= self.data['contacts_history'], 
									afh = self.data['active_feet_history'])

def MakeWoofer():
	"""
	Create robot object
	"""
	mujoco_state_est 	= MuJoCoStateEstimator()
	mujoco_contact_est 	= MuJoCoContactEstimator()
	qp_controller	 	= QPBalanceController()
	qp_controller.InitQPBalanceController(WooferDynamics.woofer_mass, WooferDynamics.woofer_inertia)

	woofer = WooferRobot(mujoco_state_est, mujoco_contact_est, qp_controller, dt = 0.001)

	return woofer







