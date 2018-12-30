import osqp
import scipy.sparse as sparse
import numpy as np
import cvxpy as cp

np.set_printoptions(suppress=True)

l1 = 0.2 #[m]

body_w = 0.4 #[m]
body_l = 0.8 #[m]
stance_h = 0.6 #[m]

body_m = 8.0 #[kg]
n = 4 #[num feet]

def CrossProductMatrix(a):
	return np.array([[0, -a[2], a[1]], \
					 [a[2], 0, -a[0]], \
					 [-a[1], a[0], 0]])

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

# generate basic foot config
# this is for testing only
xfl = np.array([-body_w/2, body_l/2, -stance_h])
xfr = np.array([body_w/2, body_l/2, -stance_h])
xbl = np.array([-body_w/2, -body_l/2, -stance_h])
xbr = np.array([body_w/2, -body_l/2, -stance_h])

q_feet = np.concatenate((xfl,xfr,xbl,xbr))
# generate acceleration matrix
A = AccelerationMatrix(q_feet)

# verify A
# print('Map between foot force and acceleration:')
# print(A)

##################### Four-foot Test (Stand) #########################

# target body acceleration
b = np.zeros((6,))
b[2] = 10 # [m/s2]
b[3] = 1 # [rad/s2]

# ||Ax-b||^2 = xT ATA x - 2ATb x + bTb = 0.5 * xT P x + qx + constant
# P = 2 * A'A, q = -2*A'b, constant = b'b
P = 2*np.matmul(A.T,A) + np.eye(12)*0.001
# print("P")
# print(P)

q = -2*np.dot(A.T,b)
# print("q")
# print(q)

P = sparse.csc_matrix(P)
q = q

A2 = sparse.csc_matrix([[0,0,1,0,0,0,0,0,0,0,0,0],
						[0,0,0,0,0,1,0,0,0,0,0,0],
						[0,0,0,0,0,0,0,0,1,0,0,0],
						[0,0,0,0,0,0,0,0,0,0,0,1]])
lb = np.array([0,0,0,0])
ub = np.array([float('inf')]*4)

prob = osqp.OSQP()
prob.setup(P,q,A2,lb,ub,alpha=1.0,verbose=0)
res = prob.solve()

print('------------Four-foot test (stand)------------')
print('Desired acceleration')
print(b)
print('Foot forces:')
print(res.x)
print('Status: %s'%res.info.status)
print('Actual accleration')
print(np.dot(A,res.x))

# ##################### Two-foot Test (Trot) #########################
# # Find foot forces when you set the condition that only two legs are in contact with the ground
# b = np.zeros((6,1))
# b[2] = 10 # [m/s2]
# b[3] = 1 # [rad/s2]
# b[4] = -1 # [rad/s2]
# x = cp.Variable(3*n)
# objective = cp.Minimize(cp.sum_squares(A*x - b) + (1e-3)*cp.norm(x))
# constraints = [0 <= x[2], 0 <= x[5], 0 <= x[8], 0 <= x[11]]
# constraints = constraints + [0 <= x[3:6], 0 >= x[3:6], 0 <= x[6:9], 0 >= x[6:9]]
# prob = cp.Problem(objective, constraints)
# result = prob.solve()
# print('------------Two-foot test (trot)------------')
# print('Desired acceleration')
# print(b)
# print('Foot forces:')
# print(x.value)
# print(prob.status)
# print('Actual accleration')
# print(np.dot(A,x.value))

# ##################### Three-foot Test (Walk) #########################
# # Find foot forces when you set the condition that only two legs are in contact with the ground
# b = np.zeros((6,1))
# b[2] = 10 # [m/s2]
# b[3] = 1 # [rad/s2]
# b[4] = -1 # [rad/s2]
# x = cp.Variable(3*n)
# objective = cp.Minimize(cp.sum_squares(A*x - b) + (1e-3)*cp.norm(x))
# constraints = [0 <= x[2], 0 <= x[5], 0 <= x[8], 0 <= x[11]]
# constraints = constraints + [0 <= x[0:3], 0 >= x[0:3]]
# prob = cp.Problem(objective, constraints)
# result = prob.solve()
# print('------------Three-foot test (trot)------------')
# print('Desired acceleration')
# print(b)
# print('Foot forces:')
# print(x.value)
# print(prob.status)
# print('Actual accleration')
# print(np.dot(A,x.value))