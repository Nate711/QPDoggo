import WooferDynamics 		
import rotations

class SwingLegController:
	def __init__(self):
		pass
	
	def GeneralTrackingController(self, q, qd, qdd, q_ref, qd_ref, Kp, Kd):
		"""
		Output torques to track a given trajectory using a combination of feed forward and PD. Operates on ONE trajectory only. 
		q:   trajectory xyz coordinates in the world frame
		qd:  trajectory velocity
		qdd: trajectory acceleration
		"""

		# Leg dynamics parameters
		H = 0.250 	# leg inertia [kg]
		C = 0.0 	# leg coriolis term 
		G = 0		# position dependent terms

		g = np.array([0,0,-9.81])
		F_ext = g

		# Feed forward force in xy world coordinates
		F_ff = H * qdd - F_ext
		F_pd = Kp * (q_ref - q) + Kd * (qd_ref - qd)
		F_world = F_ff + F_pd
		return F_world

	def ForceToJointTorques(sef, F_world, leg_joints, body_orientation):
		"""
		leg_joints: 3-vector
		"""

		# Transform into body coordinates
		F_body = rotations.quat2mat(body_orientation).T * F_world

		# Transform into joint torques
		(beta,theta,r) = tuple(leg_joints)
		joint_torques  = (WooferDynamics.LegJacobian(beta, theta, r, abaduction_offset = 0)).T * F_body
		
		return joint_torques

	def TrajectoryPlanner(self):
		"""
		Given the current foot 
		"""
		pass

	def RaibertHeuristic(self, state):
		"""
		Use raibert heuristics to plan footstep locations given the current velocity and desired velocity
		"""
		pass

