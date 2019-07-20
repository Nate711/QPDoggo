import numpy as np
import pickle
import math, time
import rotations

# Helper math functions
from MathUtils 				import CrossProductMatrix, RunningMax

# Provides kinematic functions among others
import WooferDynamics

from JointSpaceController 	import JointSpaceController, TrotPDController
from BasicController 		import PropController
from QPBalanceController 	import QPBalanceController
from StateEstimator 		import MuJoCoStateEstimator, UKFStateEstimator
from ContactEstimator 		import MuJoCoContactEstimator
from GaitPlanner 			import StandingPlanner, StepPlanner
from SwingLegController		import PDSwingLegController, ZeroSwingLegController

from WooferConfig import WOOFER_CONFIG, QP_CONFIG, SWING_CONTROLLER_CONFIG, GAIT_PLANNER_CONFIG

class WooferRobot():
	"""
	This class represents the onboard Woofer software.

	The primary input is the mujoco simulation data and the
	primary output is a set joint torques.
	"""
	def __init__(self, state_estimator, contact_estimator, qp_controller, gait_planner, swing_controller, dt):
		"""
		Initialize object variables
		"""

		self.contact_estimator 	= contact_estimator
		self.state_estimator 	= state_estimator
		self.qp_controller 		= qp_controller # QP controller for calculating foot forces
		self.gait_planner		= gait_planner
		self.swing_controller 	= swing_controller
		self.state 				= None
		self.contacts 			= None


		self.max_torques 		= RunningMax(12)
		self.max_forces 		= RunningMax(12)

		init_data_size = 10
		self.data = {}
		self.data['torque_history'] 			= np.zeros((12,init_data_size))
		self.data['force_history']				= np.zeros((12,init_data_size))
		self.data['ref_wrench_history'] 		= np.zeros((6,init_data_size))
		self.data['contacts_history'] 			= np.zeros((4,init_data_size))
		self.data['active_feet_history'] 		= np.zeros((4,init_data_size))
		self.data['swing_torque_history']		= np.zeros((12,init_data_size))
		self.data['swing_force_history']		= np.zeros((12,init_data_size))

		self.data['swing_trajectory']			= np.zeros((12,init_data_size))
		self.data['foot_positions']				= np.zeros((12,init_data_size))
		self.data['phase_history']				= np.zeros((1,init_data_size))
		self.data['step_phase_history']			= np.zeros((1,init_data_size))

		self.data['accelerometer_history']		= np.zeros((3,init_data_size))
		self.data['gyro_history']				= np.zeros((3,init_data_size))
		self.data['joint_pos_sensor_hist']		= np.zeros((12,init_data_size))
		self.data['joint_vel_sensor_hist']		= np.zeros((12,init_data_size))
		self.data['force_sensor_hist']			= np.zeros((12,init_data_size))
		self.data['state_history']				= np.zeros((13,init_data_size))
		#
		self.accelerometer_sensor 				= None
		self.gyro_sensor 						= None
		self.joint_pos_sensor					= None
		self.joint_vel_sensor					= None
		self.force_sensor_hist					= None

		self.dt = dt
		self.t = 0
		self.i = 0

		self.foot_forces = np.array([0,0,WOOFER_CONFIG.MASS*9.81/4]*4)
		self.state_arr = np.zeros(13)

		self.phase = 0
		self.step_phase = 0 # Increases from 0 to 1 and back to 0 every step
		self.step_locations = np.zeros(12)
		self.p_step_locations = np.zeros(12)

		self.swing_torques = np.zeros(12)
		self.swing_trajectory = np.zeros(12)
		self.foot_positions = np.zeros(12)

	def step(self, sim):
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
		################################### Sensor update ###################################
		self.accelerometer_sensor 	= WooferDynamics.accel_sensor(sim)
		self.gyro_sensor 			= WooferDynamics.gyro_sensor(sim)
		self.joint_pos_sensor 		= WooferDynamics.joint_pos_sensor(sim)
		self.joint_vel_sensor 		= WooferDynamics.joint_vel_sensor(sim)
		self.force_sensor			= WooferDynamics.force_sensor(sim)

		################################### Contact estimation ###################################
		self.contacts 	= self.contact_estimator.update(sim)

		################################### State estimation ###################################
		self.state 		= self.state_estimator.update(sim, self.contacts)

		if 0:
			################################### Gait planning ###################################
			(self.step_locations, self.p_step_locations, \
			 p_ref, rpy_ref, self.active_feet, self.phase, self.step_phase) = self.gait_planner.update(	self.state,
																										self.contacts,
																										self.t,
																										WOOFER_CONFIG,
																										GAIT_PLANNER_CONFIG)
			# print("phase: %s"%self.phase)

			################################### Swing leg control ###################################
			# TODO. Zero for now, but in the future the swing controller will provide these torques
			self.swing_torques, \
			self.swing_forces,\
			self.swing_trajectory, \
			self.foot_positions = self.swing_controller.update(	self.state,
																self.step_phase,
																self.step_locations,
																self.p_step_locations,
																self.active_feet,
																WOOFER_CONFIG,
																SWING_CONTROLLER_CONFIG)
		else:
			(unused1, unused2, \
			 p_ref, rpy_ref, self.active_feet, self.phase) = self.gait_planner.update(	self.state,
																						self.contacts,
																						self.t)

			self.swing_torques = np.zeros(12)
			self.swing_forces = np.zeros(12)
			self.swing_trajectory = np.zeros(12)

		################################### QP force control ###################################
		# Rearrange the state for the qp solver
		qp_state = (self.state['p'],
					self.state['p_d'],
					self.state['q'],
					self.state['w'],
					self.state['j'])

		self.state_arr = np.concatenate([self.state['p'], self.state['q'], self.state['p_d'], self.state['w']])

		# Use forward kinematics from the robot body to compute where the woofer feet are
		self.feet_locations = WooferDynamics.LegForwardKinematics(self.state['q'], self.state['j'])

		# Calculate foot forces using the QP solver
		(qp_torques, self.foot_forces, self.ref_wrench) = self.qp_controller.Update(qp_state,
																					self.feet_locations,
																					self.active_feet,
																					p_ref,
																					rpy_ref,
																					self.foot_forces,
																					WOOFER_CONFIG,
																					QP_CONFIG)

		# Expanded version of active feet
		active_feet_12 = self.active_feet[[0,0,0,1,1,1,2,2,2,3,3,3]]

		# Mix the QP-generated torques and PD-generated torques to produce the final joint torques sent to the robot
		self.torques = active_feet_12 * qp_torques + (1 - active_feet_12) * self.swing_torques

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
			for key in self.data.keys():
				self.data[key] = np.append(self.data[key], np.zeros((np.shape(self.data[key])[0],1000)),axis=1)


		self.max_forces.Update(self.foot_forces)
		self.max_torques.Update(self.torques)
		self.data['torque_history'][:,self.i] 		= self.torques
		self.data['force_history'][:,self.i] 		= self.foot_forces
		self.data['ref_wrench_history'][:,self.i] 	= self.ref_wrench
		self.data['contacts_history'][:,self.i] 	= self.contacts
		self.data['active_feet_history'][:,self.i] 	= self.active_feet
		self.data['swing_torque_history'][:,self.i]	= self.swing_torques
		self.data['swing_force_history'][:,self.i] 	= self.swing_forces
		self.data['swing_trajectory'][:,self.i]		= self.swing_trajectory
		self.data['foot_positions'][:,self.i]		= self.foot_positions
		self.data['phase_history'][:,self.i]		= self.phase
		self.data['step_phase_history'][:,self.i]	= self.step_phase
		self.data['accelerometer_history'][:,self.i]= self.accelerometer_sensor
		self.data['gyro_history'][:,self.i]			= self.gyro_sensor
		self.data['joint_pos_sensor_hist'][:,self.i]= self.joint_pos_sensor
		self.data['joint_vel_sensor_hist'][:,self.i]= self.joint_vel_sensor
		self.data['state_history'][:,self.i]		= self.state_arr
		self.data['force_sensor_hist'][:,self.i]	= self.force_sensor

	def print_data(self):
		"""
		Print debug data
		"""
		# print("Time: %s"			%self.t)
		# print("Cartesian: %s"		%self.state['p'])
		# print("Euler angles: %s"	%rotations.quat2euler(self.state['q']))
		# print("Max gen. torques: %s"%self.max_torques.CurrentMax())
		# print("Max forces: %s"		%self.max_forces.CurrentMax())
		# print("Reference wrench: %s"%self.ref_wrench)
		# print("feet locations: %s"	%self.feet_locations)
		# print("contacts: %s"		%self.contacts)
		# print("QP feet forces: %s"	%self.foot_forces)
		# print("Joint torques: %s"	%self.torques)
		# print("Accelerometer: %s"	%self.accelerometer_sensor)
		# print("Gyro: %s"			%self.gyro_sensor)
		# print("Joint Pos Sensor: %s"%self.joint_pos_sensor)
		# print("Joint Vel Sensor: %s"%self.joint_vel_sensor)
		print('\n')

	def save_logs(self):
		"""
		Save the log data to file
		"""
		np.savez('woofer_numpy_log',**self.data)
		# with open('woofer_logs.pickle', 'wb') as handle:
		# 	pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def MakeWoofer(dt = 0.001):
	"""
	Create robot object
	"""
	mujoco_contact_est 	= MuJoCoContactEstimator()
	mujoco_state_est 	= MuJoCoStateEstimator()

	x0 = np.zeros(10)
	x0[2] = 0.32
	x0[3:7] = np.array([1, 0, 0, 0])
	ukf_state_est 		= UKFStateEstimator(x0, dt)
	qp_controller	 	= QPBalanceController()
	gait_planner 		= StandingPlanner()
	# gait_planner 		= StepPlanner()
	swing_controller	= PDSwingLegController()

	# woofer = WooferRobot(mujoco_state_est, mujoco_contact_est, qp_controller, gait_planner, swing_controller, dt = dt)
	woofer = WooferRobot(ukf_state_est, mujoco_contact_est, qp_controller, gait_planner, swing_controller, dt = dt)

	return woofer
