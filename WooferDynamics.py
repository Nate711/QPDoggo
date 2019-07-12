import numpy as np
import rotations
from WooferConfig import WOOFER_CONFIG

def FootSelector(binary_foot_selector):
	return binary_foot_selector[[0,0,0,1,1,1,2,2,2,3,3,3]]
def CoordinateExpander(binary_coordinate_selector):
	return binary_coordinate_selector[[0,1,2,0,1,2,0,1,2,0,1,2]]

def LegForwardKinematics(quat_orientation, joints):
	"""
	Gives the North-East-Down (NED)-style coordinates of the four feet. NED coordinates are coordinates in a noninertial reference frame
	attached to the CoM of the robot. The axes of the NED frame are parallel to the x, y, and z axes in this simulation.

	quat_orientation:	unit quaternion for body orientation
	joints: 			joint angles in qpos ordering
	"""

	def LegFK(abad, for_back, radial, handedness):
		hands = {"left":1, "right":-1}
		offset = hands[handedness]*(WOOFER_CONFIG.ABDUCTION_OFFSET)
		leg_unrotated = np.array([0, offset, -WOOFER_CONFIG.LEG_L + radial])
		R = rotations.euler2mat([abad, for_back, 0])	# rotation matrix for abduction, then forward back
		foot = np.dot(R,leg_unrotated)
		return foot

	# Get foot locations in local body coordinates
	# Right-handedness of frame means y is positive in the LEFT direction
	foot_fr = LegFK(joints[0], joints[1], joints[2], "right") + np.array([WOOFER_CONFIG.LEG_FB, 	-WOOFER_CONFIG.LEG_LR, 0])
	foot_fl = LegFK(joints[3], joints[4], joints[5], "left")  + np.array([WOOFER_CONFIG.LEG_FB, 	 WOOFER_CONFIG.LEG_LR, 0])
	foot_br = LegFK(joints[6], joints[7], joints[8], "right") + np.array([-WOOFER_CONFIG.LEG_FB, 	-WOOFER_CONFIG.LEG_LR, 0])
	foot_bl = LegFK(joints[9], joints[10],joints[11],"left")  + np.array([-WOOFER_CONFIG.LEG_FB, 	 WOOFER_CONFIG.LEG_LR, 0])

	# Transform into world coordinates (centered around robot COM)
	feet_col_stack = np.column_stack((foot_fr, foot_fl, foot_br, foot_bl))
	feet_world = np.dot(rotations.quat2mat(quat_orientation), feet_col_stack)
	return feet_world.T.reshape(12)

def FootLocationsWorld(state):
	"""
	state: dictionary of state variables
	"""

	feet_locs_floating_NED		= LegForwardKinematics(state['q'], state['j'])
	feet_locs_world 			= feet_locs_floating_NED + state['p'][[0,1,2,0,1,2,0,1,2,0,1,2]]
	return feet_locs_world

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
	unrotated = np.array([0,abaduction_offset,-WOOFER_CONFIG.LEG_L + r])
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

def FootForceToJointTorques(F_world, leg_joints, body_quat, abaduction_offset = 0):
		"""
		Transforms a world-coordinate foot force into joint torques
		leg_joints: 3-vector
		"""

		# Transform into body coordinates
		F_body = np.matmul(rotations.quat2mat(body_quat).T, F_world)

		# Transform into joint torques
		(beta,theta,r) = tuple(leg_joints)
		joint_torques  = np.matmul(LegJacobian(beta, theta, r, abaduction_offset = abaduction_offset).T, F_body)
		return joint_torques

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

def position(sim):
	return sim.data.qpos[0:3]
def velocity(sim):
	return sim.data.qvel[0:3]
def orientation(sim):
	return sim.data.qpos[3:7]
def angular_velocity(sim):
	return sim.data.qvel[3:6]
def joints(sim):
	return sim.data.qpos[7:]
def joint_vel(sim):
	return sim.data.qvel[6:]
def accel_sensor(sim):
	return sim.data.sensordata[0:3]
def gyro_sensor(sim):
	return sim.data.sensordata[3:6]
def joint_pos_sensor(sim):
	return sim.data.sensordata[6:18]
def joint_vel_sensor(sim):
	return sim.data.sensordata[18:30]
def force_sensor(sim):
	return sim.data.sensordata[30:42]
