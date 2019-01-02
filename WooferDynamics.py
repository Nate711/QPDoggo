import numpy as np
import rotations

# Robot joint limits
max_revolute_torque = 8
max_prismatic_force = 133

# Robot geometry
woofer_leg_fb = 0.23					# front-back distance from center line to leg axis
woofer_leg_lr = 0.175 					# left-right distance from center line to leg plane
woofer_leg_l  = 0.32
woofer_abduction_offset = 0

# Robot inertia params
woofer_mass = 7.50 # kg

(w,l,t) = (0.176, 0.66, 0.92)
Ix = woofer_mass/12 * (w**2 + t**2)
Iy = woofer_mass/12 * (l**2 + t**2)
Iz = woofer_mass/12 * (l**2 + w**2)
woofer_inertia = np.zeros((3,3))
woofer_inertia[0,0] = Ix
woofer_inertia[1,1] = Iy
woofer_inertia[2,2] = Iz

def LegForwardKinematics(quat_orientation, joints):
	"""
	Gives the cartesian coordinates of the four feet

	quat_orientation:	unit quaternion for body orientation
	joints: 			joint angles in qpos ordering
	"""
	
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
