import osqp
import scipy.sparse as sparse
import numpy as np
import math

from MathUtils import CrossProductMatrix

def AccelerationMatrix(q_feet):
	"""
	Generates the matrix that maps from foot forces to body accelerations in x,y,z and pitch,roll,yaw
	From Equation (5) in the paper "High-slope terrain locomotion for torque-controlled quadruped robots"
	
	q_feet: 12 vector of the foot locations, in world coordinates centered around the robot COM
	
	Note: The 12-vector of foot forces are ground reaction forces, so for example, if the foot is pushing down,
	the z-component is >= 0
	"""
	A = np.zeros((6,12))
	A[0:3,0:3] = np.eye(3)
	A[0:3,3:6] = np.eye(3)
	A[0:3,6:9] = np.eye(3)
	A[0:3,9:12] = np.eye(3)
	A[3:6,0:3] = CrossProductMatrix(q_feet[0:3])
	A[3:6,3:6] = CrossProductMatrix(q_feet[3:6])
	A[3:6,6:9] = CrossProductMatrix(q_feet[6:9])
	A[3:6,9:12] = CrossProductMatrix(q_feet[9:12])
	return A


def SolveFeetForces(q_feet, feet_contact, reference_wrench, mu = 1.0, alpha = 0.1, beta = 0, gamma = 100.0, verbose=0):
	"""
	Use OSQP to solve for feet forces. 

	q_feet				: Location of feet relative to body COM in world coordinates 
								(yaw may be in local coordinates)
	reference_wrench	: reference force and torque on the body 
	mu 					: Friction coefficient
	alpha				: Weight on normalizer for feet forces
	beta				: Weight on smoothness between feet forces
	gamma				: Scaling factor on angular acceleration deviations

	OSQP uses an objective of the form:
							 (1/2)x'Px + qx 
	and linear ineq constraints:
								 lb < Cx < ub

	Our objective is to select foot forces that minimize the difference between the 
	realized body acceleration and a desired body acceleration.

	This is akin to:
							Minimize ||Ax-b||^2
	However, we want to prioritize angular stability of the body, so we use a quadratic norm of K:
							Minimize [(Ax-b)'K(Ax-b)]


	When expanded, it becomes:
							Minimize x'A'KAx - 2A'Kbx + b'Kb
	We can write in the OSPQ form with the following substitutions
							P = 2A'KA
							q = -2A'Kb

	We also want to minimize the norm of x and so the final objective is
							Minimize x'(A'KA + R)x -A'Kbx
							P = 2(A'KA + R)
							q = -2A'Kb

	in the code A'KA is called P0, and R is R
	"""
	
	## Params ##
	max_horz_force = 133
	max_vert_force = 133
	min_vert_force = 1

	A = AccelerationMatrix(q_feet)
	b = reference_wrench

	# Construct normalizer matrix R
	alpha_revolute	= alpha
	alpha_prismatic = alpha * 0.10
	R = np.diag([alpha_revolute, alpha_revolute, alpha_prismatic]*4)

	# Construct quadratic norm matrix K
	K = np.diag([1,1,1,gamma,gamma,gamma])

	# Construct P0
	P0 = np.matmul(A.T,np.matmul(K, A))

	# Construct the final P matrix
	P_dense = 2*(P0 + R)
	P = sparse.csc_matrix(P_dense)

	# Construct the linear term q
	q = -2*np.matmul(A.T,np.matmul(K, b))

	# Set up the inequality constraints with matrix X
	mu_pyramid = mu/(2**0.5)
	# cone: fx^2+fy^2 <= mu^2 * fz^2
	C = np.zeros((28,12))

	# Enforce friction pyramid constraints on all 4 feet
	for i in range(4):
		# fz >= 0
		C[i*5, i*3+2] = 1
		# ufz+fx >= 0
		C[i*5+1, i*3] = 1
		C[i*5+1, i*3+2] = mu
		# ufz-fx >= 0
		C[i*5+2, i*3] = -1
		C[i*5+2, i*3+2] = mu
		# ufz+fy >= 0
		C[i*5+3, i*3+1] = 1
		C[i*5+3, i*3+2] = mu
		# ufz-fy >= 0
		C[i*5+4, i*3+1] = -1
		C[i*5+4, i*3+2] = mu

	# Enforce limits on horizontal components
	for i in range(4):
		C[i*2+20, i*3] 	= 1
		C[i*2+20+1, i*3+1]= 1
		
	C = sparse.csc_matrix(C)
	lb = np.zeros((28,))
	ub = np.array([np.inf]*28)
	for i in range(4):
		lb[i*5] = min_vert_force 	if feet_contact[i] else 0
		ub[i*5] = max_vert_force 	if feet_contact[i] else 0
	
	# Enforce that the horizontal ||foot forces|| are <= 40
	lb[20:] = -max_horz_force
	ub[20:] =  max_horz_force

	# print(feet_contact, ub)

	prob = osqp.OSQP()
	prob.setup(P,q,C,lb,ub,alpha=1.0,verbose=0)
	res = prob.solve()

	if verbose>0:
		print('------------ QP Outputs ------------')
		print('Desired acceleration')
		print(b)
		print('Foot forces:')
		print(res.x)
		print('Status: %s'%res.info.status)
		print('Actual accleration')
		print(np.dot(A,res.x))


		acc_cost 	= np.matmul(res.x,	np.matmul(P0,	res.x)) + np.dot(q,res.x) + np.dot(b,b)
		force_cost 	= np.dot(res.x, 	np.dot(R, 	res.x))
		print('Accuracy cost: %s \t Force cost: %s'%(acc_cost, force_cost))
		print('\n')

	return res.x


# # generate basic foot config
# # this is for testing only
# np.set_printoptions(suppress=True)
# body_w = 0.4 #[m]
# body_l = 0.8 #[m]
# stance_h = 0.6 #[m]
# xfl = np.array([-body_w/2, body_l/2, -stance_h])
# xfr = np.array([body_w/2, body_l/2, -stance_h])
# xbl = np.array([-body_w/2, -body_l/2, -stance_h])
# xbr = np.array([body_w/2, -body_l/2, -stance_h])

# q_feet = np.concatenate((xfl,xfr,xbl,xbr))
# # generate acceleration matrix

# reference_wrench = np.array([0, 0, 9.81, 1, 0,0 ])
# SolveFeetForces(q_feet, reference_wrench)
